import importlib
import sys

from rsna.utils.predictions import PredictionsManager  # NOQA


def get_class(name):
    mod, cls = split_module_and_class(name)
    assert len(mod) > 0 and len(cls) > 0, (name, mod, cls)
    m = sys.modules[
        mod] if mod in sys.modules else importlib.import_module(mod)
    return getattr(m, cls)


def split_module_and_class(name):
    words = name.split('.')
    return '.'.join(words[:-1]), words[-1]
