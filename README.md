# openneuro_brainsets

OpenNeuro brainsets for [torch_brain](https://github.com/neuro-galaxy/torch_brain):
pipelines that download and convert EEG/iEEG datasets from
[OpenNeuro](https://openneuro.org/) into torch_brain's HDF5 format, and Dataset
classes to load them.

This is a **brainsets plugin**: an installable package that registers its pipelines
with torch_brain's `brainsets` CLI under the `openneuro` namespace. It is a
community package, maintained here rather than by the torch_brain team.

## Datasets

| Brainset | Python class | Modality | Source |
| --- | --- | --- | --- |
| `openneuro:klinzing_sleep_ds005555` | `KlinzingSleepDS005555` | EEG (PSG + headband), sleep | [ds005555](https://openneuro.org/datasets/ds005555) |
| `openneuro:kochi_visualnaming_ds006914` | `KochiVisualNamingDS006914` | iEEG (ECoG), picture naming | [ds006914](https://openneuro.org/datasets/ds006914) |
| `openneuro:shirazi_hbnr1_ds005505` | `ShiraziHBNR1DS005505` | EEG, HBN release 1 | [ds005505](https://openneuro.org/datasets/ds005505) |

## Install

No PyPI release yet; install from git (or a local clone):

```bash
uv add torch_brain
uv add "openneuro_brainsets @ git+https://github.com/milosobral/openneuro_brainsets"
# for development
git clone https://github.com/milosobral/openneuro_brainsets && cd openneuro_brainsets
uv pip install -e ".[dev]"
```

## Prepare a dataset

Configure storage once with `brainsets config set` (shared with torch_brain).

**With plugin support in torch_brain** (planned; see "Status" below):

```bash
brainsets list
brainsets prepare openneuro:klinzing_sleep_ds005555 --cores 8
```

**With current torch_brain releases**, point the CLI at the pipeline folder shipped
in this package and run it in the active environment (which already has this
package and its dependencies installed):

```bash
python -m openneuro_brainsets list
brainsets prepare "$(python -m openneuro_brainsets path klinzing_sleep_ds005555)" \
    --local --use-active-env --cores 8
```

Either way, processed files are written to
`<processed_dir>/klinzing_sleep_ds005555/<recording_id>.h5`, the same location
torch_brain's built-in pipelines used, so data prepared with older torch_brain
versions keeps loading.

OpenNeuro pipelines accept extra flags: `--redownload`, `--reprocess` and
`--on-version-mismatch {prompt,continue,abort}` (use `continue` or `abort` in
non-interactive runs).

## Load it in Python

```python
from torch_brain.samplers import RandomFixedWindowSampler
from openneuro_brainsets import KlinzingSleepDS005555, OpenNeuroDataset

ds = KlinzingSleepDS005555(root="/data/processed", split_type="intersubject")
# equivalent, for any brainset prepared by this plugin:
ds = OpenNeuroDataset(
    "/data/processed", "klinzing_sleep_ds005555", split_type="intersubject"
)

sampler = RandomFixedWindowSampler(
    sampling_intervals=ds.get_sampling_intervals("train"),
    window_length=30.0,
)
```

Split types (unchanged from torch_brain):

- `intrasession`: each recording is cut in time into train/valid/test (80/10/10 by default).
- `intersubject` / `intersession`: whole subjects (or subject+session pairs) are
  assigned to a split by a seeded hash.

Split names are `"train"`, `"valid"` and `"test"`, matching torch_brain's built-in
datasets so they mix in a `NestedDataset`. `"val"` still works but emits a
`DeprecationWarning`.

Dataset classes need `torch` (as torch_brain's do); running pipelines does not.

## Adding a dataset

See [docs/creating_a_pipeline.md](docs/creating_a_pipeline.md).

## Status

This code was ported from torch_brain (commit `e92e8d1`), where these datasets
lived under `torch_brain.pipeline.openneuro` and `torch_brain.datasets`. Changes
from the torch_brain version:

- import paths: `openneuro_brainsets.pipeline.OpenNeuroPipeline`,
  `openneuro_brainsets.OpenNeuroDataset`, `openneuro_brainsets.openneuro_api`;
- `OpenNeuroDataset(root, brainset, split_type, ...)` takes the brainset id instead
  of `dataset_dir`;
- the validation split is named `"valid"` (was `"val"`).

The `brainsets.pipelines` entry point in `pyproject.toml` is what torch_brain's CLI
will use to discover this plugin once plugin support lands there.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
