
from typing import Callable
import ConfigSpace as CS
import numpy as np

import pandas as pd
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from bo_models.bo_tpe import bo_tpe

def wrapper_func(bench: Callable) -> Callable:
    def func(config: CS.Configuration) -> float:
        return bench(config.get_dictionary())
    return func

def update_config_space_default_and_get_init_config(fun_to_evaluate, config_space, n_init, seed = 0, config_init = None):
    init_data = bo_tpe(fun_to_evaluate, config_space, 0, n_init, seed, config_init, just_fetch_information = True, reset_info = False)
    data = {k: v.tolist() for k, v in init_data.items()}
    for hp in config_space.get_hyperparameters():
        hp.default_value = data[hp.name][-1]

    n_init = len(data["loss"])
    assert n_init == n_init
    init_configs = [
        CS.Configuration(
            config_space,
            values={
                hp_name: data[hp_name][i] for hp_name in config_space
            })
        for i in range(n_init - 1)
    ]
    return config_space, init_configs

def bo_smac(fun_to_evaluate, config_space, n_runs, n_init, seed = 0, config_init = None, order_list = None):
    if hasattr(fun_to_evaluate, "reseed"):
        fun_to_evaluate.reseed(seed)
    fun_to_evaluate.reset_results()
    config_space, init_configs = update_config_space_default_and_get_init_config(fun_to_evaluate, config_space, n_init, seed, config_init)
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": n_runs + n_init,
        "cs": config_space,
    })
    if hasattr(fun_to_evaluate, "reseed"):
        # We need to reseed again because SMAC doubly evaluates the init configs
        fun_to_evaluate.reseed(seed)
    opt = SMAC4HPO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper_func(fun_to_evaluate),
        initial_configurations=init_configs,
        initial_design=None,
    )
    opt.optimize()
    vals = [float(v.cost) for v in opt.runhistory.data.values()]
    return vals, pd.DataFrame(fun_to_evaluate.all_results)