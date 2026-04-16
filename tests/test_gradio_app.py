import unittest
from unittest.mock import patch

from gradio_app import _build_context_from_history, _format_router_output, chat_fn


class TestGradioContextBuilder(unittest.TestCase):
    def test_empty_history(self):
        self.assertEqual(_build_context_from_history([]), "")

    def test_builds_context_from_recent_turns(self):
        history = [
            ("u1", "a1"),
            ("u2", "a2"),
            ("u3", "a3"),
        ]
        context = _build_context_from_history(history, max_memory=2)
        self.assertEqual(context, "USER: u2\nASSISTANT: a2\nUSER: u3\nASSISTANT: a3")


class TestGradioOutputFormatting(unittest.TestCase):
    def test_plain_text_passthrough(self):
        self.assertEqual(_format_router_output("hello"), "hello")

    def test_image_marker_is_cleaned(self):
        text = "Result text\nIMAGE_PATH::output.png"
        formatted = _format_router_output(text)
        self.assertIn("Result text", formatted)
        self.assertIn("Image saved to: `output.png`", formatted)
        self.assertNotIn("IMAGE_PATH::", formatted)

    def test_rag_error_format(self):
        formatted = _format_router_output({"type": "rag", "error": "No relevant docs"})
        self.assertIn("RAG Error", formatted)
        self.assertIn("No relevant docs", formatted)

    def test_rag_success_format(self):
        payload = {
            "type": "rag",
            "answer": "Answer",
            "sources_cited": "doc1.txt",
            "query_variants": ["q1", "q2"],
            "retrieved": [("doc1.txt", "chunk", 0.9)],
            "validated": [("doc1.txt", "chunk", 0.9)],
        }
        formatted = _format_router_output(payload)
        self.assertIn("Document-Based Answer", formatted)
        self.assertIn("Answer", formatted)
        self.assertIn("doc1.txt", formatted)


class TestChatFn(unittest.TestCase):
    @patch("gradio_app._route_query")
    def test_chat_fn_routes_with_context_and_formats(self, mock_route_query):
        mock_route_query.return_value = "hello"
        history = [("first", "reply"), ("second", "response")]

        result = chat_fn("hi", history)

        self.assertEqual(result, "hello")
        mock_route_query.assert_called_once_with(
            "hi",
            "USER: first\nASSISTANT: reply\nUSER: second\nASSISTANT: response",
        )


if __name__ == "__main__":
    unittest.main()
