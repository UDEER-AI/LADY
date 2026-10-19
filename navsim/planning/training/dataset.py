from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import pickle
import gzip
import os

import torch
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


def pad_sequence(sequence: torch.Tensor, target_sequence_length: int) -> (torch.Tensor, int):
    """
    Pads a sequence to a target sequence length.
    :param sequence: sequence to pad
    :return: padded sequence， and the number of padding frames added
    """
    T, C, H, W = sequence.shape
    if T < target_sequence_length:
        pad = torch.zeros(target_sequence_length - T, C, H, W, dtype=sequence.dtype)
        sequence = torch.cat([pad, sequence], dim=0)  # left padding
    return sequence, target_sequence_length - T

def pad_status(status_data: torch.Tensor, target_sequence_length: int) -> (torch.Tensor, int):
    T, hid_dim = status_data.shape
    if T < target_sequence_length:
        pad = torch.zeros(target_sequence_length - T, hid_dim, dtype=status_data.dtype)
        status_data = torch.cat([pad, status_data], dim=0)  # left padding
    return status_data, target_sequence_length - T


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
            self,
            cache_path: str,
            feature_builders: List[AbstractFeatureBuilder],
            target_builders: List[AbstractTargetBuilder],
            log_names: Optional[List[str]] = None,
            num_history: Optional[int] = None,
            enable_padding: bool = False,
            split: str = "train",
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        if enable_padding:
            assert num_history is not None, 'num history must not be none'
        self._cache_path = Path(cache_path)
        self._num_history = num_history
        self._enable_padding = enable_padding

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._debug_path = self._cache_path / 'debug'
        self._debug_file = self._cache_path / 'debug' / f"{split}_cache_paths.pkl"

        if os.path.exists(self._debug_file):
            with open(self._debug_file, 'rb') as f:
                self._valid_cache_paths = pickle.load(f)
        else:
            self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
                cache_path=self._cache_path,
                feature_builders=self._feature_builders,
                target_builders=self._target_builders,
                log_names=self.log_names,
            )
            os.makedirs(self._debug_path, exist_ok=True)
            with open(self._debug_file, 'wb') as f:
                pickle.dump(self._valid_cache_paths, f)

        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        features, targets = self._load_scene_with_token(self.tokens[idx])
        if self._enable_padding:
            features["camera_feature"], padding_len = pad_sequence(features["camera_feature"],
                                                                   self._num_history)
            features["lidar_feature"], padding_len = pad_sequence(features["lidar_feature"],
                                                                  self._num_history)
            features["status_feature"], padding_len = pad_status(features["status_feature"],
                                                                self._num_history)
            if padding_len > 0:
                logger.info(f"Padding {padding_len} frames for token {self.tokens[idx]}")
            features["valid_frame_len"] = self._num_history - padding_len

        return features, targets

    @staticmethod
    def _load_valid_caches(
            cache_path: Path,
            feature_builders: List[AbstractFeatureBuilder],
            target_builders: List[AbstractTargetBuilder],
            log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            scene_loader: SceneLoader,
            feature_builders: List[AbstractFeatureBuilder],
            target_builders: List[AbstractTargetBuilder],
            cache_path: Optional[str] = None,
            force_cache_computation: bool = False,
            num_history: Optional[int] = None,
            enable_padding: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )
        self._num_history = num_history
        self._enable_padding = enable_padding
        if enable_padding:
            assert num_history is not None, "num_history must be set when enable_padding is True"

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
            cache_path: Optional[Path],
            feature_builders: List[AbstractFeatureBuilder],
            target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                    token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
            features["camera_feature"], padding_len = pad_sequence(features["camera_feature"],
                                                                   self._num_history)
            features["lidar_feature"], padding_len = pad_sequence(features["lidar_feature"],
                                                                  self._num_history)
            features["status_feature"], padding_len = pad_status(features["status_feature"],
                                                                self._num_history)
                                                                                                                            
            if padding_len > 0:
                logger.info(f"Padding {padding_len} frames for token {token}")
            features["valid_frame_len"] = self._num_history - padding_len
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return features, targets
