import hashlib
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from torch_brain.data import Data, Interval
from torch_brain.datasets import Dataset, MultiChannelDatasetMixin

OpenNeuroSplitType = Literal["intrasession", "intersubject", "intersession"]
OpenNeuroSplit = Literal["train", "valid", "test"]

_VALID_SPLIT_TYPES = get_args(OpenNeuroSplitType)
_VALID_SPLITS = get_args(OpenNeuroSplit)


def _get_integer_hash_from_string(string: str) -> int:
    """Compute a deterministic integer hash from a string using MD5.

    Copied from the private ``torch_brain.utils.split._get_integer_hash_from_string``
    so that split assignments stay identical to the ones torch_brain produced.
    """
    hash_obj = hashlib.md5(string.encode("utf-8"))
    return int(hash_obj.hexdigest(), 16)


def _normalize_split(split: str) -> str:
    """Map the deprecated ``"val"`` split name to ``"valid"`` and validate."""
    if split == "val":
        warnings.warn(
            "The split name 'val' is deprecated, use 'valid' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return "valid"
    if split not in _VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}. Must be one of {_VALID_SPLITS}.")
    return split


class OpenNeuroDataset(MultiChannelDatasetMixin, Dataset):
    """
    Dataset class for OpenNeuro brainsets prepared with ``openneuro_brainsets``.

    This class provides an interface for loading, representing, and manipulating
    OpenNeuro datasets using the MultiChannelDatasetMixin and the Dataset interface.
    It supports various splitting strategies for machine learning workflows, notably
    'intrasession', 'intersubject', and 'intersession' splits.

    Args:
        root: Root directory containing processed brainsets
            (the ``processed_dir`` used by ``brainsets prepare``).
        brainset: The brainset id, i.e. the sub-directory of ``root`` holding the
            processed recordings (e.g. ``"klinzing_sleep_ds005555"``).
        split_type: The split strategy to use, must be one of
            'intrasession', 'intersubject', or 'intersession'.
        recording_ids: List of recording IDs to include,
            or None to use all available recordings.
        transform: Optional sample transform.
        uniquify_channel_ids_with_subject: Whether to prefix channel IDs with
            ``subject.id`` via ``MultiChannelDatasetMixin``.
            Defaults to ``False``.
        uniquify_channel_ids_with_session: Whether to prefix channel IDs with
            ``session.id`` via ``MultiChannelDatasetMixin``.
            Defaults to ``True``.
        split_ratios: Tuple of three floats (train, valid, test) whose sum must be 1.0.
            Specifies the proportion of the dataset to use for the train, validation,
            and test splits, respectively. All ratios must be in [0, 1] and their sum must be 1.0.
            If the sum does not equal 1.0, a ValueError is raised.
        seed: The seed for the random number generator. Used for computing splits in
            intersubject and intersession mode. Defaults to 42.
    """

    def __init__(
        self,
        root: str,
        brainset: str,
        split_type: OpenNeuroSplitType,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        uniquify_channel_ids_with_subject: bool = False,
        uniquify_channel_ids_with_session: bool = True,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        **kwargs,
    ):
        if split_type not in _VALID_SPLIT_TYPES:
            raise ValueError(
                f"Invalid split_type '{split_type}'. Must be one of {_VALID_SPLIT_TYPES}."
            )
        self.brainset = brainset
        self.split_type = split_type

        super().__init__(
            dataset_dir=Path(root) / brainset,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        # Configure subject/session-based channel-id prefixing behavior.
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_subject = (
            uniquify_channel_ids_with_subject
        )
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_session = (
            uniquify_channel_ids_with_session
        )

        self.split_ratios = self._validate_split_ratios(split_ratios)

        self.seed = seed

    def _validate_split_ratios(
        self, split_ratios: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Validate the split ratios.

        Args:
            split_ratios: Tuple of three floats (train, valid, test), whose sum must be 1.0.

        Returns:
            Tuple[float, float, float]: The validated split ratios.

        Raises:
            ValueError: If any ratio is negative, or sum does not equal 1.0.
        """
        if any(ratio < 0 for ratio in split_ratios):
            raise ValueError("`split_ratios` cannot contain negative values")
        if not np.isclose(sum(split_ratios), 1.0, atol=1e-8):
            raise ValueError("The sum of `split_ratios` must be exactly 1.0")
        return split_ratios

    def get_sampling_intervals(
        self,
        split: Literal["train", "valid", "test"] | None = None,
    ) -> dict[str, Interval]:
        """
        Retrieve the sampling intervals for each recording according to the specified split.

        If `split` is None, returns the full interval domain for every recording for unrestricted sampling.
        If a split ("train", "valid", or "test") is provided, returns only the intervals (within each recording)
        eligible for sampling under the current split type. ``"val"`` is accepted as a
        deprecated alias of ``"valid"``.

        Args:
            split: One of "train", "valid", or "test" to select intervals corresponding to that split,
                or None to retrieve the entire domain for all recordings.

        Returns:
            Dictionary mapping recording IDs to their valid Interval objects for sampling in the given split
            (or full Interval domain if split is None).

        Raises:
            ValueError: If the requested `split` or the dataset's `split_type` is not recognized/supported.
        """
        if split is None:
            return super().get_sampling_intervals()

        split = _normalize_split(split)

        intervals = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            intervals[rid] = self.get_default_sampling_intervals(rec, split)
        return intervals

    def get_default_sampling_intervals(
        self,
        recording: Data,
        split: Literal["train", "valid", "test"],
    ) -> Interval:
        """
        Get the default sampling intervals for a given split. These intervals are behavior agnostic, meaning they
        do not take into account any task or behavioral (event/label) annotations when creating the train, valid,
        and test splits—interval assignment is performed solely based on session or subject, not on in-task structure.

        Notes:
        - For split_type == "intrasession", intervals are split causally into train, valid, and test based on split_ratios.
        - For split_type == "intersubject" or "intersession", only the assigned recordings are included for each split
        (using hash-based assignment); all others return an empty interval.
        """
        split = _normalize_split(split)

        if self.split_type == "intrasession":
            starts = np.asarray(recording.domain.start)
            ends = np.asarray(recording.domain.end)
            durations = ends - starts

            train_ends = starts + durations * self.split_ratios[0]
            valid_ends = train_ends + durations * self.split_ratios[1]
            test_ends = valid_ends + durations * self.split_ratios[2]

            if split == "train":
                return Interval(start=starts, end=train_ends)
            elif split == "valid":
                return Interval(start=train_ends, end=valid_ends)
            return Interval(start=valid_ends, end=test_ends)

        elif self.split_type == "intersubject" or self.split_type == "intersession":
            if self.split_type == "intersubject":
                string_id = recording.subject.id
            elif self.split_type == "intersession":
                string_id = f"{recording.subject.id}_{recording.session.id}"

            base_str = f"{string_id}_{self.seed}"
            hash_int = _get_integer_hash_from_string(base_str)
            normalized_hash = (hash_int % 10000) / 10000.0

            if normalized_hash < self.split_ratios[0]:
                assignment = "train"
            elif normalized_hash < (self.split_ratios[0] + self.split_ratios[1]):
                assignment = "valid"
            else:
                assignment = "test"

            if assignment == split:
                return recording.domain
            else:
                return Interval(start=np.array([]), end=np.array([]))

        raise ValueError(
            f"Invalid split_type '{self.split_type}'. Must be one of {_VALID_SPLIT_TYPES}."
        )


class _CuratedOpenNeuroDataset(OpenNeuroDataset):
    """Base for named datasets: same as :class:`OpenNeuroDataset` with a fixed brainset."""

    brainset_id: str

    def __init__(
        self,
        root: str,
        split_type: OpenNeuroSplitType,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            root=root,
            brainset=self.brainset_id,
            split_type=split_type,
            recording_ids=recording_ids,
            transform=transform,
            **kwargs,
        )


class KlinzingSleepDS005555(_CuratedOpenNeuroDataset):
    """
    Klinzing Sleep EEG Dataset (OpenNeuro ds005555).

    The Bitbrain Open Access Sleep (BOAS) dataset contains simultaneous recordings
    from a clinical PSG system and a wearable EEG headband across 128 nights, with
    expert-consensus sleep stage labels.

    To download and prepare this dataset, run
    ``brainsets prepare openneuro:klinzing_sleep_ds005555``.

    Args:
        root: Root directory containing processed brainsets.
        split_type: Dataset split strategy, see :class:`OpenNeuroDataset`.
        recording_ids: List of explicit recording IDs to load, or None for all.
        transform: Optional transform to apply to each sample.
        **kwargs: Additional keyword arguments forwarded to :class:`OpenNeuroDataset`.

    **References**

    Source: https://openneuro.org/datasets/ds005555
    """

    brainset_id = "klinzing_sleep_ds005555"


class KochiVisualNamingDS006914(_CuratedOpenNeuroDataset):
    """
    Kochi Visual Naming iEEG Dataset (OpenNeuro ds006914).

    A large-scale intracranial EEG (ECoG) dataset with 110 subjects and 353
    recordings during a picture naming task.

    To download and prepare this dataset, run
    ``brainsets prepare openneuro:kochi_visualnaming_ds006914``.

    Args:
        root: Root directory containing processed brainsets.
        split_type: Dataset split strategy, see :class:`OpenNeuroDataset`.
        recording_ids: List of explicit recording IDs to load, or None for all.
        transform: Optional transform to apply to each sample.
        **kwargs: Additional keyword arguments forwarded to :class:`OpenNeuroDataset`.

    **References**

    Source: https://openneuro.org/datasets/ds006914
    """

    brainset_id = "kochi_visualnaming_ds006914"


class ShiraziHBNR1DS005505(_CuratedOpenNeuroDataset):
    """
    Shirazi Healthy Brain Network Release 1 EEG Dataset (OpenNeuro ds005505).

    EEG recordings from participants performing passive and active tasks, including
    resting state, movie watching and cognitive tasks.

    To download and prepare this dataset, run
    ``brainsets prepare openneuro:shirazi_hbnr1_ds005505``.

    Args:
        root: Root directory containing processed brainsets.
        split_type: Dataset split strategy, see :class:`OpenNeuroDataset`.
        recording_ids: List of explicit recording IDs to load, or None for all.
        transform: Optional transform to apply to each sample.
        **kwargs: Additional keyword arguments forwarded to :class:`OpenNeuroDataset`.

    **References**

    Source: https://openneuro.org/datasets/ds005505
    """

    brainset_id = "shirazi_hbnr1_ds005505"
