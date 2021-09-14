# SYSTEM IMPORTS
from abc import ABC, abstractmethod
import numpy as np


# PYTHON PROJECT IMPORTS


class Posterior(ABC):
    def __init__(self,
                 num_params: int,
                 dtype: np.dtype = np.float32) -> None:
        self.num_params: int = num_params
        self.dtype: np.dtype = dtype
        self.num_samples_seen: int = 0

    @abstractmethod
    def update(self,
               theta: np.ndarray) -> None:
        ...

    @abstractmethod
    def finalize(self) -> None:
        ...


    @abstractmethod
    def sample(self) -> np.ndarray:
        ...

