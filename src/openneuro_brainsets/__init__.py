"""OpenNeuro brainsets for torch_brain.

Pipelines live in :mod:`openneuro_brainsets.pipelines` and are exposed to the
``brainsets`` CLI under the ``openneuro`` namespace. Dataset classes are imported
lazily so that running a pipeline does not require ``torch``.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openneuro_brainsets")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover

__all__ = [
    "OpenNeuroDataset",
    "OpenNeuroSplitType",
    "OpenNeuroPipeline",
    "KlinzingSleepDS005555",
    "KochiVisualNamingDS006914",
    "ShiraziHBNR1DS005505",
    "get_pipeline_dir",
    "list_brainsets",
]

_LAZY_ATTRS = {
    "OpenNeuroDataset": ".dataset",
    "OpenNeuroSplitType": ".dataset",
    "KlinzingSleepDS005555": ".dataset",
    "KochiVisualNamingDS006914": ".dataset",
    "ShiraziHBNR1DS005505": ".dataset",
    "OpenNeuroPipeline": ".pipeline",
    "get_pipeline_dir": ".pipelines",
    "list_brainsets": ".pipelines",
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        return getattr(import_module(_LAZY_ATTRS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
