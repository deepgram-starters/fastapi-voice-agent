import asyncio
import json
import os
import sys
import unittest
from types import SimpleNamespace


os.environ.setdefault("DEEPGRAM_API_KEY", "test-key")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app


class FakeBrowserWebSocket:
    def __init__(self, token, messages):
        self.headers = {"sec-websocket-protocol": f"access_token.{token}"}
        self.messages = list(messages)
        self.forwarded_unknown_event = asyncio.Event()
        self.sent_bytes = []
        self.sent_text = []
        self.accepted_subprotocol = None

    async def accept(self, subprotocol):
        self.accepted_subprotocol = subprotocol

    async def close(self, code, reason):
        raise AssertionError(f"unexpected close: {code} {reason}")

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        await self.forwarded_unknown_event.wait()
        return {"type": "websocket.disconnect"}

    async def send_bytes(self, message):
        self.sent_bytes.append(message)

    async def send_text(self, message):
        self.sent_text.append(message)
        if json.loads(message).get("type") == "FutureEvent":
            self.forwarded_unknown_event.set()


class FakeAgentConnection:
    def __init__(self, received_messages):
        self.received_messages = list(received_messages)
        self.media = []
        self.settings = []
        self.update_speak = []
        self.update_prompt = []
        self.injected_messages = []
        self.raw_messages = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True

    async def recv(self):
        if self.received_messages:
            return self.received_messages.pop(0)
        await asyncio.Event().wait()

    async def send_media(self, message):
        self.media.append(message)

    async def send_settings(self, message):
        self.settings.append(message)

    async def send_update_speak(self, message):
        self.update_speak.append(message)

    async def send_update_prompt(self, message):
        self.update_prompt.append(message)

    async def send_inject_user_message(self, message):
        self.injected_messages.append(message)

    async def _send(self, message):
        self.raw_messages.append(message)


class VoiceAgentBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_typed_and_unknown_protocol_messages(self):
        settings = {
            "type": "Settings",
            "audio": {"input": {"encoding": "linear16", "sample_rate": 16000}},
        }
        update_speak = {
            "type": "UpdateSpeak",
            "speak": {"provider": {"type": "deepgram", "model": "aura-asteria-en"}},
        }
        update_prompt = {"type": "UpdatePrompt", "prompt": "Be concise."}
        injected_message = {"type": "InjectUserMessage", "content": "Hello"}
        function_response = {"type": "FunctionCallResponse", "id": "call-1", "content": "ok"}
        future_event = {"type": "FutureEvent", "value": "preserve me"}

        token = app.jwt.encode({"sub": "test"}, app.SESSION_SECRET, algorithm="HS256")
        browser = FakeBrowserWebSocket(
            token,
            [
                {"bytes": b"microphone-audio"},
                {"text": json.dumps(settings)},
                {"text": json.dumps(update_speak)},
                {"text": json.dumps(update_prompt)},
                {"text": json.dumps(injected_message)},
                {"text": json.dumps(function_response)},
            ],
        )
        connection = FakeAgentConnection([b"agent-audio", future_event])
        original_deepgram = app.deepgram
        app.deepgram = SimpleNamespace(agent=SimpleNamespace(v1=SimpleNamespace(connect=lambda: connection)))
        try:
            await app.voice_agent(browser)
        finally:
            app.deepgram = original_deepgram

        self.assertEqual(browser.accepted_subprotocol, f"access_token.{token}")
        self.assertEqual(browser.sent_bytes, [b"agent-audio"])
        self.assertEqual([json.loads(message) for message in browser.sent_text], [future_event])
        self.assertEqual(connection.media, [b"microphone-audio"])
        self.assertEqual(connection.settings[0].dict(), settings)
        self.assertEqual(connection.update_speak[0].dict(), update_speak)
        self.assertEqual(connection.update_prompt[0].dict(), update_prompt)
        self.assertEqual(connection.injected_messages[0].dict(), injected_message)
        self.assertEqual(connection.raw_messages, [function_response])
        self.assertTrue(connection.closed)
