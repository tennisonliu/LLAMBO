from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

import ConfigSpace as CS

import numpy as np

from parzen_estimator import (
    MultiVariateParzenEstimator,
    ParzenEstimatorType,
    build_categorical_parzen_estimator,
    build_numerical_parzen_estimator,
)

from tpe.utils.constants import CategoricalHPType, NumericType, NumericalHPType, WeightFuncType, config2type


class TreeStructuredParzenEstimator:
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        quantile_func: Callable[[int], int],
        weight_func: WeightFuncType,
        n_ei_candidates: int,
        metric_name: str,
        seed: Optional[int],
        min_bandwidth_factor: float,
        min_bandwidth_factor_for_discrete: Optional[float],
        top: Optional[float],
        multivariate: bool,
        magic_clip: bool,
        magic_clip_exponent: float,
        prior: bool,
        heuristic: Optional[Literal["optuna", "hyperopt"]] = None,
    ):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            n_ei_candidates (int): The number of samplings to optimize the EI value
            config_space (CS.ConfigurationSpace): The searching space of the task
            hp_names (List[str]): The list of hyperparameter names
            metric_name (str): The name of the metric (or objective function value)
            observations (Dict[str, Any]): The storage of the observations
            sorted_observations (Dict[str, Any]): The storage of the observations sorted based on loss
            min_bandwidth_factor (float): The minimum bandwidth for continuous.
            min_bandwidth_factor_for_discrete (Optional[float]):
                The minimum bandwidth factor for discrete.
                If None, it adapts so that the factor gives 0.1 after the discrete modifications.
            top (Optional[float]):
                The hyperparam of the cateogircal kernel. It defines the prob of the top category.
                If None, it adapts the parameter in a way that Optuna does.
                top := (1 + 1/N) / (1 + c/N) where
                c is the number of choices and N is the number of observations.
            is_categoricals (Dict[str, bool]): Whether the given hyperparameter is categorical
            is_ordinals (Dict[str, bool]): Whether the given hyperparameter is ordinal
            quantile_func (Callable[[np.ndarray], int]):
                The function that returns the number of a better group based on the total number of evaluations.
            magic_clip (bool):
                Whether to use the magic clip in TPE.
        """
        self._multivariate = multivariate
        self._rng = np.random.RandomState(seed)
        self._n_ei_candidates = n_ei_candidates
        self._config_space = config_space
        self._hp_names = list(config_space._hyperparameters.keys())
        self._metric_name = metric_name
        self._n_lower = 0
        self._quantile = 0
        self._min_bandwidth_factor = min_bandwidth_factor
        self._min_bandwidth_factor_for_discrete = min_bandwidth_factor_for_discrete
        self._magic_clip = magic_clip
        self._magic_clip_exponent = magic_clip_exponent
        self._heuristic = heuristic
        self._prior = prior
        self._top = top
        self._size = 0

        self._observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._sorted_observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._observations[self._metric_name] = np.array([])
        self._sorted_observations[self._metric_name] = np.array([])
        self._order = np.array([])

        self._quantile_func = quantile_func
        self._weight_func = weight_func

        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "CategoricalHyperparameter"
            for hp_name in self._hp_names
        }

        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "OrdinalHyperparameter"
            for hp_name in self._hp_names
        }
        self._mvpe_lower: MultiVariateParzenEstimator
        self._mvpe_upper: MultiVariateParzenEstimator

    def _insert_observations(
        self, key: str, insert_loc: int, val: Any, is_categorical: bool, dtype: Optional[Type[np.number]] = None
    ) -> None:
        data, sorted_data = self._observations, self._sorted_observations

        if is_categorical:
            data[key] = np.append(data[key], val)
            if sorted_data[key].size == 0:  # cannot cast str to float!
                sorted_data[key] = np.array([val], dtype="U32")
            else:
                sorted_data[key] = np.insert(sorted_data[key], insert_loc, val)
        else:
            if dtype is None:
                raise ValueError("dtype must be specified, but got None")

            data[key] = np.append(data[key], val).astype(dtype)
            sorted_data[key] = np.insert(sorted_data[key], insert_loc, val).astype(dtype)

    def update_observations(self, eval_config: Dict[str, NumericType], loss: float) -> None:
        """
        Update the observations for the TPE construction

        Args:
            eval_config (Dict[str, NumericType]): The configuration to evaluate (after conversion)
            loss (float): The loss value as a result of the evaluation
        """
        sorted_losses, losses = (
            self._sorted_observations[self._metric_name],
            self._observations[self._metric_name],
        )
        insert_loc = np.searchsorted(sorted_losses, loss, side="right")
        self._observations[self._metric_name] = np.append(losses, loss)
        self._sorted_observations[self._metric_name] = np.insert(sorted_losses, insert_loc, loss)
        self._order = np.insert(self._order, insert_loc, self.size)

        self._size += 1
        self._n_lower = self._quantile_func(self.size)
        self._quantile = self._n_lower / self.size

        for hp_name in self._hp_names:
            is_categorical = self._is_categoricals[hp_name]
            config_type = self._config_space.get_hyperparameter(hp_name).__class__.__name__
            self._insert_observations(
                key=hp_name,
                insert_loc=insert_loc,
                val=eval_config[hp_name],
                is_categorical=is_categorical,
                dtype=config2type[config_type] if not is_categorical else None,
            )

        self._update_parzen_estimators()

    def _calculate_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        sorted_loss_vals = self._sorted_observations[self._metric_name]
        n_lower, n_upper = self._n_lower, sorted_loss_vals.size - self._n_lower
        lower_vals, upper_vals = sorted_loss_vals[:n_lower], sorted_loss_vals[n_lower:]
        threshold = upper_vals[0] if upper_vals.size > 0 else np.inf
        weights_lower = self._weight_func(
            size=n_lower,
            order=self._order[:n_lower],
            sorted_loss_vals=lower_vals,
            lower_group=True,
            threshold=threshold,
            prior=self.prior,
        )
        weights_upper = self._weight_func(
            size=n_upper,
            order=self._order[n_lower:],
            sorted_loss_vals=upper_vals,
            lower_group=False,
            prior=self.prior,
        )
        return weights_lower, weights_upper

    def _update_parzen_estimators(self) -> None:
        n_lower = self._n_lower
        pe_lower_dict: Dict[str, ParzenEstimatorType] = {}
        pe_upper_dict: Dict[str, ParzenEstimatorType] = {}
        weights_lower, weights_upper = self._calculate_weights()
        for hp_name in self._hp_names:
            is_categorical = self._is_categoricals[hp_name]
            sorted_observations = self._sorted_observations[hp_name]

            # split observations
            lower_vals = sorted_observations[:n_lower]
            upper_vals = sorted_observations[n_lower:]

            pe_lower_dict[hp_name], pe_upper_dict[hp_name] = self._get_parzen_estimator(
                lower_vals=lower_vals,
                upper_vals=upper_vals,
                weights_lower=weights_lower,
                weights_upper=weights_upper,
                hp_name=hp_name,
                is_categorical=is_categorical,
            )

        self._mvpe_lower = MultiVariateParzenEstimator(pe_lower_dict)
        self._mvpe_upper = MultiVariateParzenEstimator(pe_upper_dict)

    def get_config_candidates(self) -> Dict[str, np.ndarray]:
        """
        Since we compute the probability improvement of each objective independently,
        we need to sample the configurations in advance.

        Args:
            n_samples (int):
                The number of samples.
                If None is provided, we use n_ei_candidates.

        Returns:
            config_cands (Dict[str, np.ndarray]):
                A dict of arrays of candidates in each dimension
        """
        return self._mvpe_lower.sample(
            n_samples=self._n_ei_candidates, rng=self._rng, dim_independent=True, return_dict=True
        )

    def compute_probability_improvement(self, config_cands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute the probability improvement given configurations

        Args:
            config_cands (Dict[str, np.ndarray]):
                The dict of candidate values for each dimension.
                The length is the number of dimensions and
                each array has the length of n_ei_candidates.

        Returns:
            config_ll_ratio (np.ndarray):
                The log of the likelihood ratios of each configuration.
                The shape is (n_ei_candidates, ) if multivariate.
                If univariate, the shape is (dim, n_ei_candidates).
        """
        dim = len(self._hp_names)
        n_samples = next(iter(config_cands.values())).size
        if self._multivariate:
            density_ratios = np.zeros((n_samples,))
            config_ll_lower = self._mvpe_lower.log_pdf(config_cands)
            config_ll_upper = self._mvpe_upper.log_pdf(config_cands)
            density_ratios = config_ll_lower - config_ll_upper
        else:
            density_ratios = np.zeros((dim, n_samples))
            pdf_lower = self._mvpe_lower.dimension_wise_pdf(config_cands)
            pdf_upper = self._mvpe_upper.dimension_wise_pdf(config_cands)

            for d in range(dim):
                density_ratios[d] = pdf_lower[d] / pdf_upper[d]

        return density_ratios

    def _get_parzen_estimator(
        self,
        lower_vals: np.ndarray,
        upper_vals: np.ndarray,
        weights_lower: np.ndarray,
        weights_upper: np.ndarray,
        hp_name: str,
        is_categorical: bool,
    ) -> Tuple[ParzenEstimatorType, ParzenEstimatorType]:
        """
        Construct parzen estimators for the lower and the upper groups and return them

        Args:
            lower_vals (np.ndarray): The array of the values in the lower group
            upper_vals (np.ndarray): The array of the values in the upper group
            weights_lower (np.ndarray): The weights for the values in the lower group
            weights_upper (np.ndarray): The weights for the values in the upper group
            hp_name (str): The name of the hyperparameter
            is_categorical (bool): Whether the given hyperparameter is categorical

        Returns:
            pe_lower (ParzenEstimatorType): The parzen estimator for the lower group
            pe_upper (ParzenEstimatorType): The parzen estimator for the upper group
        """
        pe_lower: ParzenEstimatorType
        pe_upper: ParzenEstimatorType

        config = self._config_space.get_hyperparameter(hp_name)
        config_type = config.__class__.__name__
        is_ordinal = self._is_ordinals[hp_name]
        kwargs = dict(config=config, prior=self.prior)

        if is_categorical:
            top_lower = self._top if self._top is not None else self._calculate_adapted_top(config, lower_vals.size)
            top_upper = self._top if self._top is not None else self._calculate_adapted_top(config, upper_vals.size)
            kwargs.update(top=top_lower)
            pe_lower = build_categorical_parzen_estimator(vals=lower_vals, weights=weights_lower, **kwargs)
            kwargs.update(top=top_upper)
            pe_upper = build_categorical_parzen_estimator(vals=upper_vals, weights=weights_upper, **kwargs)
        else:
            kwargs.update(
                dtype=config2type[config_type],
                is_ordinal=is_ordinal,
                default_min_bandwidth_factor=self._min_bandwidth_factor,
                # default_min_bandwidth_factor_for_discrete=(
                #     self._calculate_adapted_bw_factor(config)
                #     if self._min_bandwidth_factor_for_discrete is None
                #     else self._min_bandwidth_factor_for_discrete
                # ),
                default_min_bandwidth_factor_for_discrete=self._min_bandwidth_factor_for_discrete,
                magic_clip=self._magic_clip,
                magic_clip_exponent=self._magic_clip_exponent,
                heuristic=self._heuristic,
                space_dim=len(self._hp_names),
            )
            pe_lower = build_numerical_parzen_estimator(vals=lower_vals, weights=weights_lower, **kwargs)
            pe_upper = build_numerical_parzen_estimator(vals=upper_vals, weights=weights_upper, **kwargs)

        return pe_lower, pe_upper

    def _calculate_adapted_bw_factor(self, config: NumericalHPType) -> float:
        """
        Calculate the adapted min_bandwidth_factor for discrete.

        Args:
            config (CategoricalHPType):
                The numerical parameter to be input.

        Return:
            adapted_bw_factor (float):
                n_grids * 0.1.
                It adapts the bandwidth to be range * 0.1.
        """
        lb, ub, q = config.lower, config.upper, config.q
        q = q if q is not None else 1
        n_grids = int((ub - lb) / q) + 1
        return n_grids * 0.1

    def _calculate_adapted_top(self, config: CategoricalHPType, n_observations: int) -> float:
        """
        Calculate the top in the Optuna way

        Args:
            config (CategoricalHPType):
                The categorical parameter to be input.
            n_observations (int):
                The number of observations.

        Return:
            adapted_top (float):
                top := (1 + 1/N) / (1 + c/N) where
                c is the number of choices and N is the number of observations + 1.
        """
        n_choices, N = len(config.choices), n_observations + 1
        return (1 + 1 / N) / (1 + n_choices / N)

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return {hp_name: vals.copy() for hp_name, vals in self._observations.items()}

    @property
    def size(self) -> int:
        return self._size

    @property
    def prior(self) -> bool:
        # Avoid size error in the update when n_observations = 1
        n_lower = self._n_lower
        n_upper = self.size - n_lower
        return self._prior or n_lower == 0 or n_upper == 0
