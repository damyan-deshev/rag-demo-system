import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import httpx

import app
import sessions
from epub import fix_opf
from iterative import fused_candidates


class ClientTests(unittest.TestCase):
    def test_wizard_selects_discovered_models(self):
        def server(req):
            if req.url.path == "/api/v1/models":
                body = {
                    "data": [
                        dict(
                            name="encoder",
                            instance_name="instance",
                            provider_name="provider",
                            model_type=["embedding"],
                        )
                    ]
                }
            else:
                self.assertEqual(req.url.path, "/v1/models")
                body = {"data": [{"id": "chat-model"}]}
            return httpx.Response(200, json=body)

        with tempfile.TemporaryDirectory() as folder, patch.object(
            app, "CONFIG", Path(folder) / "settings.json"
        ), httpx.Client(transport=httpx.MockTransport(server)) as client:
            api = app.Backend(
                dict(
                    ragflow_url="http://rag",
                    llm_url="http://model/v1",
                    embedding_model="",
                    llm_model="",
                ),
                client,
            )
            with patch.object(
                app.Prompt, "ask", side_effect=["http://rag", "http://model/v1"]
            ), patch("builtins.input", side_effect=["1", "1"]):
                app.wizard(api)
            self.assertEqual(api.cfg["embedding_model"], "encoder@instance@provider")
            self.assertEqual(api.cfg["llm_model"], "chat-model")

    def test_session_deletion_is_scoped(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"XDG_DATA_HOME": folder}
        ):
            a, b = ("server", "a"), ("server", "b")
            sessions.save(a, "Q", [{"answer": "A"}])
            ident = sessions.list_saved(a)[0][1]
            sessions.delete(b, ident)
            self.assertEqual(sessions.read(a, ident), [{"answer": "A"}])
            sessions.delete(a, ident)
            self.assertEqual(sessions.list_saved(a), [])

    def test_epub_repair_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = [Path(folder) / name for name in ("one.epub", "two.epub")]
            with zipfile.ZipFile(src, "w") as z:
                z.writestr(
                    "META-INF/container.xml",
                    '<container><rootfile full-path="book.opf"/></container>',
                )
                z.writestr(
                    "book.opf",
                    '<?xml version="1.0" encoding="iso-8859-1"?><title>café</title>'.encode(
                        "latin1"
                    ),
                )
                z.writestr("chapter.xhtml", b"chapter")
            original = src.read_bytes()
            fix_opf(src, dst, "parse OPF: iso-8859-1 CharsetReader is nil")
            self.assertEqual(src.read_bytes(), original)
            with zipfile.ZipFile(dst) as z:
                self.assertIn("café", z.read("book.opf").decode())
                self.assertEqual(z.read("chapter.xhtml"), b"chapter")

    def test_context_budget_across_passes(self):
        batches = [
            [{"id": "first"}, {"id": "common"}],
            [{"id": "common"}, {"id": "second"}],
        ]
        self.assertEqual([r["id"] for r in fused_candidates(batches, 1)], ["common"])
