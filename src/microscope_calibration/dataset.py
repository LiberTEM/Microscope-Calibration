"""
Datasets with a calibration, see :class:`CalibratedDataset`
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from libertem.io.dataset.base import DataSet

from microscope_calibration.common.model import Model4DSTEM


class AbstractCalibratedDataset(ABC):
    @property
    @abstractmethod
    def model(self) -> Model4DSTEM:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataset(self) -> DataSet:
        raise NotImplementedError()

    @abstractmethod
    def calibrated(self, model: Model4DSTEM) -> AbstractCalibratedDataset:
        raise NotImplementedError()

    @property
    @abstractmethod
    def tags(self) -> Mapping[str, Any]:
        raise NotImplementedError()


class CalibratedDataset(AbstractCalibratedDataset):
    def __init__(self, dataset: DataSet, model: Model4DSTEM):
        self._ds = dataset
        self._model = model

    @property
    def dataset(self) -> DataSet:
        return self._ds

    @property
    def model(self) -> Model4DSTEM:
        return self._model

    def calibrated(self, model) -> CalibratedDataset:
        return CalibratedDataset(dataset=self.dataset, model=model)

    @property
    def tags(self):
        return {}
