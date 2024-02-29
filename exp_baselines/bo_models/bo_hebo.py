import time
from typing import Callable
import ConfigSpace as CS

import numpy as np
import pandas as pd

from bo_models.bo_tpe import bo_tpe

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

def wrapper_func(bench: Callable) -> Callable:
    def func(configs: pd.DataFrame) -> np.ndarray:
        eval_config = {hp_name: configs[hp_name].iloc[0] for hp_name in configs.columns}
        return np.asarray([[bench(eval_config)]])
    return func

def get_init_configs(fun_to_evaluate, config_space, n_init, seed = 0, config_init = None):

    init_data = bo_tpe(fun_to_evaluate, config_space, 0, n_init, seed, config_init, just_fetch_information = True, reset_info = False)
    vals        = init_data["loss"]
    init_configs = pd.DataFrame([{
            hp_name: init_data[hp_name][i] for hp_name in config_space
        } for i in range(vals.size)
    ])

    list_vals  = list(vals)
    array_vals = np.array(list_vals)
    uniques    =  np.unique(array_vals)
    non_noise_vals = None
    # We use this in the paper to avoid many of the runs fail because matrix inversion problem.
    # if len(uniques) == 1:
    #     array_vals *= np.random.normal(0, 1, size = array_vals.shape) * 0.1/3 * array_vals[0]
    #     non_noise_vals = list_vals
    return init_configs, list(array_vals), non_noise_vals

def extract_space(config_space: CS.ConfigurationSpace):
    config_info = []
    for hp in config_space.get_hyperparameters():
        info = {"name": hp.name}
        if isinstance(hp, CS.CategoricalHyperparameter):
            info["type"] = "cat"
            info["categories"] = hp.choices
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            log = hp.log
            info["type"] = "pow" if log else "num"
            info["lb"], info["ub"] = hp.lower, hp.upper
            if log:
                info["base"] = 10
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            info["type"] = "int"
            info["lb"], info["ub"] = hp.lower, hp.upper
        else:
            raise TypeError(f"{type(type(hp))} is not supported")

        config_info.append(info)
    return DesignSpace().parse(config_info)

def bo_hebo(fun_to_evaluate, config_space, n_runs, n_init, seed = 0, config_init = None, order_list = None):
    if hasattr(fun_to_evaluate, "reseed"):
        fun_to_evaluate.reseed(seed)
    fun_to_evaluate.reset_results()
    #np.float = np.float64

    init_configs, vals, non_noise_vals = get_init_configs(fun_to_evaluate, config_space, n_init, seed, config_init)
    obj                = wrapper_func(fun_to_evaluate)

    opt = HEBO(extract_space(config_space), rand_sample = 0)
    opt.observe(init_configs, np.asarray(vals).reshape(-1, 1))
    for i in range(n_runs):
        if (i + 1) % 10 == 0:
            print(f"{i + 11} evaluations at {time.time()}")

        config = opt.suggest(n_suggestions=1)
        y = obj(config)
        opt.observe(config, y)
        vals.append(float(y[0][0]))

    print(vals)
    if non_noise_vals is not None:
        for i in range(n_init):
            vals[i] = non_noise_vals[i]

    return vals, pd.DataFrame(fun_to_evaluate.all_results)

