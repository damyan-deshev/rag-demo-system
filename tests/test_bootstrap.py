import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import local_stack as stack


class BootstrapTests(unittest.TestCase):
    def test_local_key_is_scoped_reused_and_private(self):
        calls = []

        def handler(req):
            calls.append(req.url.path)
            if req.url.path.endswith("/auth/login"):
                return httpx.Response(401)
            if req.url.path.endswith("/users"):
                return httpx.Response(
                    200, json={"code": 0}, headers={"Authorization": "session"}
                )
            if req.url.path.endswith("/system/tokens"):
                self.assertEqual(req.headers["Authorization"], "Bearer session")
                return httpx.Response(200, json={"data": {"token": "test-token"}})
            self.assertEqual(req.headers["Authorization"], "Bearer test-token")
            return httpx.Response(200, json={"data": []})

        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"XDG_CONFIG_HOME": temp}
        ), patch.object(stack, "STATE", Path(temp)), httpx.Client(
            transport=httpx.MockTransport(handler)
        ) as client:
            self.assertEqual(stack.account(client), stack.account(client))
            self.assertEqual(calls.count("/api/v1/system/tokens"), 1)
            self.assertEqual(stack.saved_key("ragflow_url", "http://other"), "")
            self.assertEqual(stack.keyfile().stat().st_mode & 0o777, 0o600)

    def test_gateway_chooses_different_models_and_container_url(self):
        def handler(req):
            if req.method == "GET":
                return httpx.Response(
                    200, json={"data": [{"id": "embed"}, {"id": "chat"}]}
                )
            self.assertEqual(req.url.path, "/v1/embeddings")
            self.assertEqual(json.loads(req.content)["model"], "embed")
            return httpx.Response(200, json={"data": [{"embedding": [0.5]}]})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client, patch(
            "builtins.input", side_effect=["http://localhost:1234/v1", "1", "2", ""]
        ), patch("getpass.getpass", return_value=""):
            url, inside, _, embed, chat = stack.model_choices(client)
            self.assertEqual(
                (url, inside, embed, chat),
                (
                    "http://localhost:1234/v1",
                    "http://host.docker.internal:1234/v1",
                    "embed",
                    "chat",
                ),
            )
