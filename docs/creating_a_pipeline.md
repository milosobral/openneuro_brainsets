# Creating an OpenNeuro pipeline

This guide shows how to add an [OpenNeuro](https://openneuro.org/) dataset to
`openneuro_brainsets`. A pipeline downloads, processes and standardizes a public
dataset into torch_brain's HDF5 format. `OpenNeuroPipeline` currently supports EEG
and iEEG.

```text
OpenNeuro S3 → Download → Process → Save to HDF5
```

`OpenNeuroPipeline` handles the whole workflow:

- discovering recordings in OpenNeuro's S3 bucket based on modality
- downloading BIDS-compliant files
- extracting the signal and metadata (channel names and types)
- storing one HDF5 file per recording, with full provenance

*Ported from torch_brain's "Creating an OpenNeuroPipeline" guide.*

## Minimal pipeline

Create `src/openneuro_brainsets/pipelines/<brainset_id>/pipeline.py`:

```python
# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

from openneuro_brainsets.pipeline import OpenNeuroPipeline


class Pipeline(OpenNeuroPipeline):
    modality = "eeg"  # or "ieeg"
    brainset_id = "my_sleep_study_ds005555"
    dataset_id = "ds005555"
    origin_version = "1.0.0"  # Check OpenNeuro for this!
    derived_version = "1.0.0"  # Version of your processing pipeline
    description = "Sleep recordings from OpenNeuro"
```

The folder name must equal `brainset_id`. The rest is inherited, but can be customized.

Working examples in this repository:

| Example | Use when | Complexity |
| --- | --- | --- |
| [`shirazi_hbnr1_ds005505`](../src/openneuro_brainsets/pipelines/shirazi_hbnr1_ds005505/pipeline.py) | All recordings have identical channels | Simple |
| [`klinzing_sleep_ds005555`](../src/openneuro_brainsets/pipelines/klinzing_sleep_ds005555/pipeline.py) | Different recordings need different channel mappings | Complex |

## The five required attributes

### 1. `modality`: EEG or iEEG?

Set `modality` to `"eeg"` for scalp/wearable EEG datasets or `"ieeg"` for
intracranial datasets. This drives BIDS recording discovery from OpenNeuro S3.

### 2. `dataset_id`: which OpenNeuro dataset?

The id must use strict OpenNeuro format: `ds` followed by exactly 6 digits
(`"ds005555"` is valid; `"5555"` and `"ds5555"` are not). Invalid ids raise an
error before data discovery.

### 3. `brainset_id`: your unique name

A descriptive, valid Python identifier, unique within this repository. Users run it
as `openneuro:<brainset_id>`, and the processed files land in
`<processed_dir>/<brainset_id>/`. Recommended naming scheme:

```python
brainset_id = "klinzing_sleep_ds005555"
#               └─ author's last name
#                        └─ short label for the main task or paradigm
#                              └─ dataset id
```

### 4. `origin_version`: the dataset version you used

Set it to the exact OpenNeuro snapshot you built and tested the pipeline with
(find it under "Snapshots" / "Version History" on the dataset page; the dataset's
`CHANGES` file describes the differences between versions). OpenNeuro datasets
evolve, so hardcoding documents which version the pipeline was validated on.

When the pipeline runs, it fetches the latest snapshot tag and compares it to
`origin_version`. If they differ, the `--on-version-mismatch` policy applies:

| Option | Behavior | When to use |
| --- | --- | --- |
| `prompt` (default) | Interactive: asks to confirm. Non-interactive: raises `ValueError` asking for `continue` or `abort`. | Local development |
| `continue` | Warns and proceeds with the latest version. | Automation / CI |
| `abort` | Raises an error on mismatch. | Strict reproducibility |

Whatever the policy, OpenNeuro S3 always serves the **latest** snapshot. The check
is informational: it does not download `origin_version`.

| Scenario | Action |
| --- | --- |
| Changes are minor or unrelated to your pipeline | Continue; the latest version is used. |
| You've tested with the new version | Update `origin_version`, test, re-run. |
| Changes break your pipeline | Adjust the pipeline, or fetch the old version manually. |

### 5. `derived_version`: your processing version

Tracks the version of your processing logic and its output. It is stored in the
brainset metadata. Increment it when you change channel mappings, filtering or any
processing logic.

## Optional attributes

- `description`: human-readable description stored in the metadata.
- `CHANNEL_NAME_REMAPPING`: `{original_name: standardized_name}`, e.g.
  `{"PSG_F3": "F3", "PSG_EOG": "EOG"}`.
- `TYPE_CHANNELS_REMAPPING`: `{type: [channel names]}`, e.g.
  `{"EEG": ["F3", "F4"], "EOG": ["EOG"]}`. Names can be original or remapped.
- `IGNORE_CHANNELS`: channel names (as in the raw file) to drop.

We recommend UPPERCASE channel names and types, in line with common EEG/iEEG
conventions.

## Per-recording customization

Override these methods when mappings depend on the recording:

```python
def get_channel_name_remapping(self, recording_id):
    if "acq-headband" in recording_id:
        return {"HB_1": "AF7", "HB_2": "AF8", "HB_PULSE": "PULSE"}
    return {"PSG_F3": "F3", "PSG_F4": "F4", "PSG_EOG": "EOG"}


def get_type_channels_remapping(self, recording_id):
    if "acq-headband" in recording_id:
        return {"EEG": ["AF7", "AF8"], "PPG": ["PULSE"]}
    return {"EEG": ["F3", "F4"], "EOG": ["EOG"]}
```

Add custom processing by overriding `process`:

```python
def process(self, download_output):
    result = self.process_common(download_output)
    if result is None:
        return  # Already processed

    data, store_path = result
    # Add your custom processing here (filters, artifact removal, ...)

    import h5py

    with h5py.File(store_path, "w") as file:
        data.to_hdf5(file)
```

## Checklist

1. Pick a dataset on [OpenNeuro](https://openneuro.org/).
2. Copy the minimal example into `pipelines/<brainset_id>/pipeline.py`.
3. Set `modality`, `dataset_id`, `brainset_id`, `origin_version`, `derived_version`.
4. Add channel name and/or type mappings if needed.
5. Optionally add a named Dataset class in `dataset.py` and export it.
6. Add the brainset id to `EXPECTED_BRAINSETS` in `tests/test_plugin.py`.
7. Run it (see the README) on one recording with `--single <recording_id>`.
