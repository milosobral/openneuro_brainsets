"""Brainset pipelines provided by this plugin.

Each sub-folder ``<brainset_id>/pipeline.py`` defines a ``Pipeline`` class for the
``brainsets`` CLI. This package is registered under the ``brainsets.pipelines``
entry point group with the ``openneuro`` namespace.
"""

from importlib.resources import files
from pathlib import Path

NAMESPACE = "openneuro"


def list_brainsets() -> list[str]:
    """Return the ids of all brainsets shipped by this plugin."""
    root = files(__name__)
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir()
        and not d.name.startswith((".", "_"))
        and (d / "pipeline.py").is_file()
    )


def get_pipeline_dir(brainset_id: str) -> Path:
    """Return the directory holding ``pipeline.py`` for ``brainset_id``."""
    if brainset_id not in list_brainsets():
        raise ValueError(
            f"Unknown brainset '{brainset_id}'. Available: {', '.join(list_brainsets())}"
        )
    return Path(str(files(__name__) / brainset_id))
