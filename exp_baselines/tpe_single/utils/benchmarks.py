from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np


UNIMODAL = "Unimodal"
MULTIMODAL = "Multimodal"


__all__ = [
    "Ackley",
    "DifferentPower",
    "DixonPrice",
    "Griewank",
    "KTablet",
    "Langermann",
    "Levy",
    "Michalewicz",
    "Perm",
    "Powell",
    "Rastrigin",
    "Rosenbrock",
    "Schwefel",
    "Sphere",
    "Styblinski",
    "Trid",
    "WeightedSphere",
    "XinSheYang",
]


def config2array(eval_config: Dict[str, float], R: float) -> np.ndarray:
    return np.array(list(eval_config.values())) * R


class AbstractFunc(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sphere(AbstractFunc):
    _R = 5
    name = "Sphere"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: 0
        vals = config2array(eval_config, cls._R)
        return np.sum(vals**2)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return x**2 + y**2


class Styblinski(AbstractFunc):
    _R = 5
    name = "Styblinski"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: -39.166165 * DIM
        vals = config2array(eval_config, R=cls._R)
        t1 = np.sum(vals**4)
        t2 = -16 * np.sum(vals**2)
        t3 = 5 * np.sum(vals)
        return 0.5 * (t1 + t2 + t3)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        t1 = x**4 + y**4
        t2 = -16 * (x**2 + y**2)
        t3 = 5 * (x + y)
        return 0.5 * (t1 + t2 + t3)


class Rastrigin(AbstractFunc):
    _R = 5.12
    name = "Rastrigin"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: 0.0
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        t1 = 10 * dim
        t2 = np.sum(vals**2)
        t3 = -10 * np.sum(np.cos(2 * np.pi * vals))
        return t1 + t2 + t3

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        cos_term = np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)
        return 20 + x**2 + y**2 - 10 * cos_term


class Schwefel(AbstractFunc):
    _R = 500
    name = "Schwefel"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: -418.9829 * DIM
        vals = config2array(eval_config, R=cls._R)
        return -np.sum(vals * np.sin(np.sqrt(np.abs(vals))))

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        zx = x * np.sin(np.sqrt(np.abs(x)))
        zy = y * np.sin(np.sqrt(np.abs(y)))
        return -(zx + zy)


class Ackley(AbstractFunc):
    _R = 32.768
    name = "Ackley"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: 0
        vals = config2array(eval_config, R=cls._R)
        t1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(vals**2)))
        t2 = -np.exp(np.mean(np.cos(2 * np.pi * vals)))
        return 20 + np.e + t1 + t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        t1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        t2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        return 20 + np.e + t1 + t2


class Griewank(AbstractFunc):
    _R = 600
    name = "Griewank"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        t1 = np.sum(vals**2) / 4000
        t2 = -np.prod(np.cos(vals / np.sqrt(np.arange(1, dim + 1))))
        return 1 + t1 + t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        t1 = (x**2 + y**2) / 4000
        t2 = np.cos(x) * np.cos(y / np.sqrt(2))
        return 1 + t1 - t2


class Perm(AbstractFunc):
    _R = 1.0
    _beta = 1.0
    name = "Perm"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        indices = np.arange(dim) + 1
        ret = 0
        for d in indices:
            center = (1 / indices) ** d
            factor = vals**d
            ret += ((indices + cls._beta) @ (factor - center)) ** 2

        return ret

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        z = ((cls._beta + 1) * (x - 1) + (cls._beta + 2) * (y - 0.5)) ** 2 + (
            (cls._beta + 1) * (x**2 - 1) + (cls._beta + 2) * (y - 0.25)
        ) ** 2
        return z


class KTablet(AbstractFunc):
    _R = 5.12
    name = "K-Tablet"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        k = (dim + 3) // 4
        vals = config2array(eval_config, R=cls._R)
        vals[k:] *= 100
        return np.sum(vals**2)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return x**2 + 10000 * y**2


class WeightedSphere(AbstractFunc):
    _R = 5.0
    name = "Weighted sphere"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        weights = np.arange(dim) + 1
        return weights @ (vals**2)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return x**2 + 2 * y**2


class Rosenbrock(AbstractFunc):
    _R = 5.0
    name = "Rosenbrock"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R)
        t1 = np.sum(100 * (vals[1:] - vals[:-1] ** 2) ** 2)
        t2 = np.sum((vals[:-1] - 1) ** 2)
        return t1 + t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return 100 * (y - x**2) ** 2 + (x - 1) ** 2


class DifferentPower(AbstractFunc):
    _R = 1.0
    name = "Sum of different power"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        indices = np.arange(dim) + 2
        return np.sum(np.abs(vals) ** indices)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return np.abs(x) ** 2 + np.abs(y) ** 3


class XinSheYang(AbstractFunc):
    _R = 2 * np.pi
    name = "Xin-She Yang"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R)
        return np.sum(np.abs(vals)) * np.exp(-np.sum(np.sin(vals**2)))

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return (np.abs(x) + np.abs(y)) * np.exp(-(np.sin(x**2) + np.sin(y**2)))


class Levy(AbstractFunc):
    _R = 10.0
    name = "Levy"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R)
        weights = 1 + (vals - 1) / 4
        t1 = np.sin(np.pi * weights[0]) ** 2
        t2 = (weights[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * weights[-1]) ** 2)
        t3 = np.sum((weights[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * weights[:-1] + 1) ** 2))
        return t1 + t2 + t3

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        wx = 1 + (x - 1) / 4
        wy = 1 + (y - 1) / 4
        t1 = np.sin(np.pi * wx) ** 2
        t2 = (wy - 1) ** 2 * (1 + np.sin(2 * np.pi * wy) ** 2)
        t3 = (wx - 1) ** 2 * (1 + 10 * np.sin(np.pi * wx + 1) ** 2)
        return t1 + t2 + t3


class Michalewicz(AbstractFunc):
    _R = np.pi / 2
    name = "Michalewicz"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R) + cls._R
        t1 = np.sin(vals)
        t2 = np.sin((np.arange(len(eval_config)) + 1) * (vals ** 2) / np.pi) ** 20
        return - t1 @ t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X + cls._R
        y = cls._R * Y + cls._R
        t1 = - np.sin(x) * np.sin(x ** 2 / np.pi) ** 20
        t2 = - np.sin(y) * np.sin(2 * y ** 2 / np.pi) ** 20
        return t1 + t2


class Trid(AbstractFunc):
    _R = None  # D ** 2
    name = "Trid"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        R = len(eval_config) ** 2
        vals = config2array(eval_config, R=R)
        t1 = np.sum((vals - 1) ** 2)
        t2 = vals[:-1] @ vals[1:]
        return t1 - t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = 4 * X
        y = 4 * Y
        return (x - 1) ** 2 + (y - 1) ** 2 - x * y


class DixonPrice(AbstractFunc):
    _R = 10.0
    name = "DixonPrice"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R)
        d = vals.size
        t1 = (vals[0] - 1) ** 2
        t2 = np.arange(2, d + 1) @ (2 * vals[1:] ** 2 - vals[:-1]) ** 2
        return t1 + t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        t1 = (x - 1) ** 2
        t2 = 2 * (2 * x ** 2 - y) ** 2
        return t1 + t2


class Powell(AbstractFunc):
    _R = 5.0
    name = "Powell"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R)
        size = vals.size // 4
        x1 = vals[::4][:size]  # 4i - 3
        x2 = vals[1::4][:size]  # 4i - 2
        x3 = vals[2::4][:size]  # 4i - 1
        x4 = vals[3::4][:size]  # 4i
        t1 = np.sum((x1 + 10 * x2) ** 2)
        t2 = np.sum(5 * (x3 - x4) ** 2)
        t3 = np.sum((x2 + 2 * x3) ** 4)
        t4 = np.sum(10 * (x1 - x4) ** 4)
        return t1 + t2 + t3 + t4

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("2D function is not defined")


class Langermann(AbstractFunc):
    _R = 5.0
    name = "Langermann"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R) + cls._R
        dim = vals.size
        m = 5
        rng = np.random.RandomState(42)
        C = rng.randint(low=1, high=5, size=m).astype(np.float64)
        A = rng.randint(low=1, high=10, size=(m, dim)).astype(np.float64)
        exps = np.sum((vals - A) ** 2, axis=-1)
        return (C * np.exp(- 1 / np.pi * exps)) @ np.cos(np.pi * exps)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X + cls._R
        y = cls._R * Y + cls._R
        C = np.array([1, 2, 5, 2, 3], dtype=np.float64).reshape(5, 1)
        A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]], dtype=np.float64)
        exps = (x - A[:, 0, np.newaxis]) ** 2 + (y - A[:, 1, np.newaxis]) ** 2  # shape = (m, N)
        return np.sum(C * np.exp(-1 / np.pi * exps) * np.cos(np.pi * exps), axis=0)
