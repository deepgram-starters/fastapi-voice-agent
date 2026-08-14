"""
FastAPI Voice Agent Starter - Raw WebSocket proxy to Deepgram

Key Features:
- WebSocket endpoint: /api/voice-agent
- JWT session auth for API protection
- Raw WebSocket proxy to Deepgram Agent API
"""

import os
import json
import secrets
import time
import asyncio

import jwt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import toml

from deepgram import AsyncDeepgramClient
from deepgram.core.unchecked_base_model import construct_type
from deepgram.core.api_error import ApiError
from deepgram.agent.v1.types import (
    AgentV1InjectUserMessage,
    AgentV1Settings,
    AgentV1UpdatePrompt,
    AgentV1UpdateSpeak,
)

load_dotenv(override=False)

CONFIG = {
    "port": int(os.environ.get("PORT", 8081)),
    "host": os.environ.get("HOST", "0.0.0.0"),
}

def load_api_key():
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY required")
    return api_key

API_KEY = load_api_key()

# One async SDK client, reused across connections; the browser never sees the API
# key. agent.v1.connect() targets wss://agent.deepgram.com/v1/agent/converse and
# handles the Authorization header, so the raw websockets proxy is no longer needed.
deepgram = AsyncDeepgramClient(api_key=API_KEY)


def _safe_error_detail(e):
    """Sanitize a Deepgram error before it reaches the browser or logs.

    NEVER surface str(e): a deepgram-sdk ApiError stringifies its request
    headers, which include Authorization: Token <api-key> — a bad connect
    would otherwise leak the key to the browser and the server logs.
    """
    if isinstance(e, ApiError):
        return f"Deepgram rejected the connection (HTTP {e.status_code})"
    return f"Failed to connect to Deepgram ({type(e).__name__})"

# ============================================================================
# SESSION AUTH - JWT tokens for API protection
# ============================================================================

SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
JWT_EXPIRY = 3600  # 1 hour


# Read frontend/dist/index.html for serving
_index_html_template = None
try:
    with open(os.path.join(os.path.dirname(__file__), "frontend", "dist", "index.html")) as f:
        _index_html_template = f.read()
except FileNotFoundError:
    pass  # No built frontend (dev mode)


def require_session(authorization: str = Header(None)):
    """FastAPI dependency for JWT session validation."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "MISSING_TOKEN",
                    "message": "Authorization header with Bearer token is required",
                }
            }
        )
    token = authorization[7:]
    try:
        jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Session expired, please refresh the page",
                }
            }
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Invalid session token",
                }
            }
        )


app = FastAPI(title="Deepgram Voice Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SESSION ROUTES - Auth endpoints (unprotected)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve index.html."""
    if not _index_html_template:
        raise HTTPException(status_code=404, detail="Frontend not built. Run make build first.")
    return HTMLResponse(content=_index_html_template)


@app.get("/api/session")
async def get_session():
    """Issues a JWT session token."""
    token = jwt.encode(
        {"iat": int(time.time()), "exp": int(time.time()) + JWT_EXPIRY},
        SESSION_SECRET,
        algorithm="HS256",
    )
    return JSONResponse(content={"token": token})


# ============================================================================
# WEBSOCKET ROUTE
# ============================================================================

@app.websocket("/api/voice-agent")
async def voice_agent(websocket: WebSocket):
    """Raw WebSocket proxy endpoint for voice agent"""
    # Validate JWT from subprotocol
    protocols = websocket.headers.get("sec-websocket-protocol", "")
    protocol_list = [p.strip() for p in protocols.split(",")]
    valid_proto = None
    for proto in protocol_list:
        if proto.startswith("access_token."):
            token = proto[len("access_token."):]
            try:
                jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
                valid_proto = proto
            except Exception:
                pass
            break

    if not valid_proto:
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await websocket.accept(subprotocol=valid_proto)
    print("Client connected to /api/voice-agent")

    forward_task = None

    try:
        # Connect to Deepgram's Voice Agent API through the SDK. The browser still
        # owns the agent protocol (it builds the Settings/Update frames), so the
        # backend stays a thin bridge; only the Deepgram-facing transport moved
        # from raw websockets to the SDK's agent.v1 socket client.
        print("Connecting to Deepgram Agent API...")
        async with deepgram.agent.v1.connect() as connection:
            print("✓ Connected to Deepgram Agent API")

            # Task to forward messages from Deepgram to client
            async def forward_from_deepgram():
                try:
                    async for message in connection:
                        if isinstance(message, (bytes, bytearray)):
                            await websocket.send_bytes(bytes(message))
                        elif hasattr(message, "model_dump_json"):
                            # Typed event -> re-serialize to the same wire JSON.
                            await websocket.send_text(message.model_dump_json())
                        elif isinstance(message, (dict, list)):
                            await websocket.send_text(json.dumps(message))
                        else:
                            await websocket.send_text(str(message))
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    detail = _safe_error_detail(e)
                    print(f"Error forwarding from Deepgram: {detail}")
                    await websocket.send_text(json.dumps({
                        "type": "Error",
                        "description": detail,
                        "code": "PROVIDER_ERROR"
                    }))

            # Start forwarding task
            forward_task = asyncio.create_task(forward_from_deepgram())

            # Forward messages from client to Deepgram. Binary frames are mic audio;
            # JSON frames are control messages the browser builds. construct_type is
            # the SDK's own deserializer and preserves every field (nested providers,
            # future settings), so behavior matches the previous verbatim proxy.
            try:
                while True:
                    message = await websocket.receive()

                    if message.get("type") == "websocket.disconnect":
                        break

                    data_bytes = message.get("bytes")
                    if data_bytes is not None:
                        await connection.send_media(data_bytes)
                        continue

                    text = message.get("text")
                    if text is None:
                        continue

                    try:
                        payload = json.loads(text)
                    except (ValueError, TypeError):
                        print("Ignoring non-JSON message from client")
                        continue

                    msg_type = payload.get("type")
                    if msg_type == "Settings":
                        await connection.send_settings(
                            construct_type(type_=AgentV1Settings, object_=payload)
                        )
                    elif msg_type == "UpdateSpeak":
                        await connection.send_update_speak(
                            construct_type(type_=AgentV1UpdateSpeak, object_=payload)
                        )
                    elif msg_type == "UpdatePrompt":
                        await connection.send_update_prompt(
                            construct_type(type_=AgentV1UpdatePrompt, object_=payload)
                        )
                    elif msg_type == "InjectUserMessage":
                        await connection.send_inject_user_message(
                            construct_type(type_=AgentV1InjectUserMessage, object_=payload)
                        )
                    else:
                        # Unrecognized control frame: forward verbatim so a new
                        # client message type is never silently dropped.
                        await connection._send(payload)

            except WebSocketDisconnect:
                print("Client disconnected")
            except Exception as e:
                print(f"Error forwarding to Deepgram: {_safe_error_detail(e)}")

    except Exception as e:
        detail = _safe_error_detail(e)
        print(f"WebSocket error: {detail}")
        try:
            await websocket.send_text(json.dumps({
                "type": "Error",
                "description": detail,
                "code": "CONNECTION_FAILED"
            }))
        except Exception:
            pass

    finally:
        # Cleanup
        if forward_task:
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass

        print("Connection cleanup complete")

@app.get("/api/metadata")
async def get_metadata():
    try:
        with open('deepgram.toml', 'r') as f:
            config = toml.load(f)
        return JSONResponse(content=config.get('meta', {}))
    except:
        return JSONResponse(status_code=500, content={"error": "Metadata read failed"})

if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 FastAPI Voice Agent Server: http://localhost:{CONFIG['port']}")
    print(f"   GET  /api/session")
    print(f"   WS   /api/voice-agent (auth required)")
    print(f"   GET  /api/metadata\n")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
