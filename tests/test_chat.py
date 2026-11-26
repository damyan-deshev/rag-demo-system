import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import app


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_index_still_calls_model_and_saves_answer(self):
        api = Mock()
        api.request.return_value = {"chunks": []}
        question = "Capital?"
        api.complete = AsyncMock(side_effect=[(question, {}), ("Paris.", {})])
        with patch.object(app, "ui"):
            turn = await app.ask(api, "book", question, 1, [], False)
        api.request.assert_called_once()
        messages = api.complete.await_args.args[0]
        self.assertEqual([m["role"] for m in messages], ["user"])
        self.assertEqual(
            json.loads(messages[0]["content"]),
            {"question": question, "retrieved_excerpts": []},
        )
        self.assertEqual(turn["answer"], "Paris.")

    async def test_original_before_rewrite_and_bounded_context(self):
        api = Mock()
        chunks = [
            {"id": "a", "content": "Original result"},
            {"id": "b", "content": "Second result"},
        ]
        api.request.side_effect = [{"chunks": chunks}, {"chunks": [chunks[1]]}]
        api.complete = AsyncMock(side_effect=[("rewritten", {}), ("answer", {})])
        with patch.object(app, "ui") as ui:
            turn = await app.ask(
                api, "book", "original", 1, [], False, 1, "Use excerpts only"
            )
        self.assertEqual(
            [c.kwargs["json"]["question"] for c in api.request.call_args_list],
            ["original", "rewritten"],
        )
        ui.print.assert_any_call("Original result")
        self.assertEqual(len(turn["sources"]), 1)
        self.assertIn("Second result", turn["sources"][0])
        self.assertIn(
            "Use excerpts only", api.complete.await_args.args[0][-1]["content"]
        )
