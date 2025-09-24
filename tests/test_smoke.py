"""Basic smoke tests for YOLO Dataset Studio packaging."""

def test_main_module_imports():
    import importlib

    assert importlib.import_module("main")
    assert importlib.import_module("toolkit")
    assert importlib.import_module("advanced_features")
