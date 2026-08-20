"""Minimal torch/transformers stubs: enough to IMPORT the pipeline modules.
Nothing numeric is exercised -- this only lets the non-neural code paths run.
Any torch.* / transformers.* submodule is auto-created permissively."""
import sys, types, importlib.abc, importlib.machinery
from unittest.mock import MagicMock

# sentence_transformers deliberately NOT stubbed: letting the import fail makes
# build_default_embedding_backend() fall back to the REAL KeywordOverlapEmbeddingBackend,
# so similarity/filtering runs genuine arithmetic instead of MagicMock no-ops.
_ROOTS = ("torch", "transformers", "peft", "accelerate")


class _Module:
    def __init__(self, *a, **k): pass
    def modules(self): return []
    def named_modules(self): return []
    def parameters(self): return []
    def eval(self): return self
    def train(self, *a): return self
    def to(self, *a, **k): return self
    def __call__(self, *a, **k): return MagicMock()


class _StubModule(types.ModuleType):
    """Any attribute not explicitly set resolves to a MagicMock."""
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        v = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, v)
        return v


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in _ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


def install():
    sys.meta_path.insert(0, _Finder())
    import torch
    for n in ("float16", "bfloat16", "float32", "long", "bool"):
        setattr(torch, n, n)
    torch.Tensor = type("Tensor", (), {})
    torch.is_tensor = lambda x: False
    torch.inference_mode = lambda *a, **k: MagicMock(
        __enter__=lambda s=None: None, __exit__=lambda *x: False)
    torch.no_grad = torch.inference_mode
    torch.manual_seed = lambda *a: None

    import torch.nn as nn
    nn.Module = _Module
    nn.LayerNorm = _Module
    nn.Linear = _Module
    nn.Identity = _Module
    nn.Parameter = lambda *a, **k: MagicMock()

    import torch.cuda as cuda
    cuda.is_available = lambda: False
    cuda.empty_cache = lambda: None
    cuda.manual_seed_all = lambda *a: None

    import transformers as tr
    tr.PreTrainedModel = _Module
