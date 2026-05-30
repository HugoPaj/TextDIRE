"""
Tests for package import behavior.
"""


def test_import_src_has_no_stdout(capsys):
    """Importing the package should not print or load local env files noisily."""
    import sys

    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            del sys.modules[module_name]

    import src  # noqa: F401

    captured = capsys.readouterr()
    assert captured.out == ""
