from typing import List

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.lady.rwkv_config import RWKVConfig
from navsim.agents.lady.rwkv_features import RWKVFeatureBuilder, RWKVTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class LADYCacheAgent(AbstractAgent):
    """Build LADY feature/target caches without a model or metric-cache scorer."""

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self._config = config

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """No model weights are required for caching."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=list(range(self._config.cam_seq_len)))

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [RWKVFeatureBuilder(config=self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [RWKVTargetBuilder(config=self._config)]
