import sys
sys.path.append('tpe_single')
from typing import Any, Callable, Dict, Optional, Union

import ConfigSpace as CS
import turbo
import numpy as np

from turbo.turbo import Turbo1
from bo_models.bo_tpe import bo_tpe
import pandas as pd

def get_bounds(
    config_space: CS.ConfigurationSpace
) -> Union[np.ndarray, np.ndarray]:
    hp_names = []
    lb, ub = [], []
    for hp_name in config_space:
        hp_names.append(hp_name)
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.CategoricalHyperparameter):
            for i in range(len(hp.choices)):
                lb.append(0)
                ub.append(1)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            lb.append(np.log(hp.lower) if hp.log else hp.lower)
            ub.append(np.log(hp.upper) if hp.log else hp.upper)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            lb.append(np.log(hp.lower) if hp.log else hp.lower - 0.5 + 1e-12)
            ub.append(np.log(hp.upper) if hp.log else hp.upper + 0.5 - 1e-12)
    return np.asarray(lb), np.asarray(ub)

def convert(
    X: np.ndarray,
    config_space: CS.ConfigurationSpace,
) -> Dict[str, Any]:
    config = {}
    cur = 0
    for hp_name in config_space:
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.UniformFloatHyperparameter):
            config[hp_name] = np.exp(X[cur]) if hp.log else X[cur]
            cur += 1
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            config[hp_name] = int(np.round(np.exp(X[cur])) if hp.log else np.round(X[cur]))
            cur += 1
        elif isinstance(hp, CS.CategoricalHyperparameter):
            config[hp_name] = int(np.argmax([X[cur + i] for i in range(len(hp.choices))]))
            cur += len(hp.choices)
    return config

def wrapper_func(bench: Callable, config_space: CS.ConfigurationSpace) -> Callable:
    def func(X: np.ndarray) -> float:
        eval_config = convert(X, config_space)
        return bench(eval_config)
    return func

def get_init_config(fun_to_evaluate, config_space, n_init, seed, config_init):
    init_data = bo_tpe(fun_to_evaluate, config_space, 0, n_init, seed, config_init, just_fetch_information = True, reset_info = False)
    init_X = []
    for hp_name in config_space:
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.CategoricalHyperparameter):
            init_choices = [int(c) for c in init_data[hp_name]]
            assert len(init_choices) == 10
            one_hot = np.zeros((len(hp.choices), 10))
            for idx, c in enumerate(init_choices):
                one_hot[c, idx] = 1.0
            init_X.extend(one_hot.tolist())
        else:
            init_X.append(np.log(init_data[hp_name]) if hp.log else init_data[hp_name])
    return np.asarray(init_X).T, np.asarray(init_data["loss"])

def bo_turbo(fun_to_evaluate, config_space, n_runs, n_init, seed = 0, config_init = None, order_list = None):
    if hasattr(fun_to_evaluate, "reseed"):
        fun_to_evaluate.reseed(seed)
    fun_to_evaluate.reset_results()
    X_init, fX_init = get_init_config(fun_to_evaluate, config_space, n_init, seed, config_init)
    lb, ub = get_bounds(config_space)
    opt = Turbo1(
        f=wrapper_func(fun_to_evaluate, config_space),
        lb=lb,
        ub=ub,
        n_init = n_init,
        max_evals = n_runs + n_init,
        seed=seed,
        verbose=False,
    )
    opt.optimize(fixed_X_init=X_init, fixed_fX_init=fX_init)
    return opt.fX.flatten().tolist(), pd.DataFrame(fun_to_evaluate.all_results)
