"""Tests for the plugin layout expected by torch_brain's ``brainsets`` CLI."""

import importlib.util
from importlib.metadata import entry_points

import pytest
from torch_brain.pipeline import BrainsetPipeline

from openneuro_brainsets.__main__ import main
from openneuro_brainsets.pipeline import OpenNeuroPipeline
from openneuro_brainsets.pipelines import (
    NAMESPACE,
    get_pipeline_dir,
    list_brainsets,
)

EXPECTED_BRAINSETS = [
    "klinzing_sleep_ds005555",
    "kochi_visualnaming_ds006914",
    "shirazi_hbnr1_ds005505",
]

# Keys allowed in a `# /// brainset-pipeline` block by torch_brain's CLI.
ALLOWED_INLINE_METADATA_KEYS = {"python-version", "dependencies"}


def _load_pipeline_cls(brainset_id):
    path = get_pipeline_dir(brainset_id) / "pipeline.py"
    spec = importlib.util.spec_from_file_location(f"_pipeline_{brainset_id}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Pipeline


def test_list_brainsets():
    assert list_brainsets() == EXPECTED_BRAINSETS


def test_get_pipeline_dir_unknown_raises():
    with pytest.raises(ValueError, match="Unknown brainset"):
        get_pipeline_dir("does_not_exist")


@pytest.mark.parametrize("brainset_id", EXPECTED_BRAINSETS)
def test_pipeline_follows_plugin_rules(brainset_id):
    cls = _load_pipeline_cls(brainset_id)
    assert issubclass(cls, BrainsetPipeline)
    assert issubclass(cls, OpenNeuroPipeline)
    assert cls.brainset_id == brainset_id
    cls.validate_dataset_id(cls.dataset_id)
    assert cls.modality in ("eeg", "ieeg")


@pytest.mark.parametrize("brainset_id", EXPECTED_BRAINSETS)
def test_inline_metadata_is_valid(brainset_id):
    from torch_brain.pipeline._cli.cli_prepare import _read_inline_metadata

    metadata = _read_inline_metadata(get_pipeline_dir(brainset_id) / "pipeline.py")
    assert metadata is not None
    assert set(metadata) <= ALLOWED_INLINE_METADATA_KEYS


def test_entry_point_registered():
    eps = entry_points(group="brainsets.pipelines")
    matches = [ep for ep in eps if ep.name == NAMESPACE]
    assert len(matches) == 1
    assert matches[0].value == "openneuro_brainsets.pipelines"


def test_cli_list(capsys):
    main(["list"])
    out = capsys.readouterr().out.split()
    assert out == [f"{NAMESPACE}:{b}" for b in EXPECTED_BRAINSETS]


def test_cli_path(capsys):
    main(["path", "klinzing_sleep_ds005555"])
    out = capsys.readouterr().out.strip()
    assert out.endswith("klinzing_sleep_ds005555")
