from typing import List, Literal, Optional, Protocol, Union

import ConfigSpace.hyperparameters as CSH

import numpy as np

from scipy.stats import rankdata


EPS = 1.0e-300
NumericType = Union[float, int]
SQR2, SQR2PI = np.sqrt(2), np.sqrt(2 * np.pi)

CategoricalHPType = Union[CSH.CategoricalHyperparameter]
NumericalHPType = Union[
    CSH.UniformIntegerHyperparameter,
    CSH.UniformFloatHyperparameter,
    CSH.OrdinalHyperparameter,
]
HPType = Union[CategoricalHPType, NumericalHPType]

config2type = {
    "UniformFloatHyperparameter": float,
    "UniformIntegerHyperparameter": int,
    "OrdinalHyperparameter": float,
}

type2config = {
    float: "UniformFloatHyperparameter",
    int: "UniformIntegerHyperparameter",
    bool: "CategoricalHyperparameter",
    str: "CategoricalHyperparameter",
}


class WeightFuncType(Protocol):
    def __call__(
        self, size: int, order: np.ndarray, sorted_loss_vals: np.ndarray, lower_group: bool, threshold: Optional[float]
    ) -> np.ndarray:
        raise NotImplementedError


class WeightFuncs:
    choices: List[str] = ["uniform", "older-smaller", "older-drop", "weaker-smaller", "expected-improvement"]
    _decay_start_from = 25

    def __init__(
        self,
        choice: Literal["uniform", "older-smaller", "older-drop", "weaker-smaller", "expected-improvement"],
    ):
        self._choice = choice
        self._prior: bool
        if choice not in self.choices:
            raise ValueError(f"choice must be in {self.choices}, but got {self._choice}")

    def __call__(
        self,
        size: int,
        order: np.ndarray,
        sorted_loss_vals: np.ndarray,
        prior: bool,
        lower_group: bool = False,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        # NOTE: We always add a weight for prior and thus weights.size is always size + 1
        self._prior = prior
        if self._choice == "uniform":
            return self.uniform(size)
        elif self._choice == "older-smaller":
            return self.older_smaller(order)
        elif self._choice == "older-drop":
            return self.older_drop(order)
        elif self._choice == "weaker-smaller":
            return self.weaker_smaller(sorted_loss_vals)
        elif self._choice == "expected-improvement":
            return self.expected_improvement(sorted_loss_vals, threshold) if lower_group else self.uniform(size)
        else:
            raise NotImplementedError(f"Unknown choice {self._choice}")

    def _linear_decayed_weight(self, size: int) -> np.ndarray:
        flat = np.ones(self._decay_start_from)
        n_decays = size - self._decay_start_from
        ramp = np.linspace(1.0 / size, 1.0, num=n_decays)
        weights = np.concatenate([ramp, flat], axis=0)
        return weights / np.sum(weights)

    def uniform(self, size: int) -> np.ndarray:
        size += self._prior  # for prior
        return np.full(size, 1.0 / size)

    def older_smaller(self, order: np.ndarray) -> np.ndarray:
        if order.size < self._decay_start_from:
            return self.uniform(order.size)

        if self._prior:
            order = np.append(order, -1)  # prior is the oldest

        ages = rankdata(order, method="ordinal") - 1
        weights = self._linear_decayed_weight(order.size)
        return weights[ages]

    def older_drop(self, order: np.ndarray) -> np.ndarray:
        if order.size < self._decay_start_from:
            return self.uniform(order.size)

        if self._prior:
            order = np.append(order, -1)  # prior is the oldest

        ages = rankdata(order, method="ordinal") - 1
        weights = np.zeros(order.size, dtype=np.float32)
        weights[-self._decay_start_from:] = 1
        weights /= np.sum(weights)
        return weights[ages]

    def weaker_smaller(self, sorted_loss_vals: np.ndarray) -> np.ndarray:
        if sorted_loss_vals.size < self._decay_start_from:
            return self.uniform(sorted_loss_vals.size)

        weights = self._linear_decayed_weight(sorted_loss_vals.size + self._prior)
        return weights[::-1]  # prior is the weakest

    def expected_improvement(self, sorted_loss_vals: np.ndarray, threshold: Optional[float]) -> np.ndarray:
        if threshold is None:
            raise ValueError("threshold must be provided for expected improvement")

        if threshold == np.inf:
            return self.uniform(size=sorted_loss_vals.size)

        weights = np.maximum(1e-12, threshold - sorted_loss_vals)
        # prior is the mean of the other weights
        if self._prior:
            weights = np.append(weights, np.mean(weights))

        weights /= np.sum(weights)
        return weights


class QuantileFunc:
    choices: List[str] = ["linear", "sqrt"]
    _max_lower_size = 25

    def __init__(self, alpha: float = 0.25, choice: str = "sqrt"):
        self._alpha = alpha
        self._choice = choice

    def __call__(self, size: int) -> int:
        if self._choice == "linear":
            size = self.linear(size)
        elif self._choice == "sqrt":
            size = self.sqrt(size)
        else:
            raise ValueError(f"choice must be in {self.choices}, but got {self._choice}")

        return min(size, self._max_lower_size)

    def sqrt(self, size: int) -> int:
        return int(np.ceil(self._alpha * np.sqrt(size)))

    def linear(self, size: int) -> int:
        return int(np.ceil(self._alpha * size))
