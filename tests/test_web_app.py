"""
Tests for the Text-DIRE web API helpers.
"""

from src.scoring import _lookup_modal_function


class TestModalLookup:
    """Tests for Modal function lookup compatibility."""

    def test_lookup_modal_function_prefers_from_name(self):
        """Use Modal's deployed-function lookup when available."""

        class Function:
            @staticmethod
            def from_name(app_name, function_name):
                return ("from_name", app_name, function_name)

            @staticmethod
            def lookup(app_name, function_name):
                return ("lookup", app_name, function_name)

        class Modal:
            pass

        Modal.Function = Function

        assert _lookup_modal_function(Modal, "text-dire", "score") == (
            "from_name",
            "text-dire",
            "score",
        )

    def test_lookup_modal_function_supports_older_lookup_api(self):
        """Fall back for Modal versions before Function.from_name."""

        class Function:
            @staticmethod
            def lookup(app_name, function_name):
                return ("lookup", app_name, function_name)

        class Modal:
            pass

        Modal.Function = Function

        assert _lookup_modal_function(Modal, "text-dire", "score") == (
            "lookup",
            "text-dire",
            "score",
        )
