import numpy as np
import pandas as pd
from bo_models.bo_tpe import bo_tpe
from skopt import gp_minimize
from sampler import obtain_space

def wrapper_func(bench, order_list):
    def func(config) -> float:
        #new_dict = {key: config[idx] for idx, key in enumerate(order_list)}
        new_dict = {key: config[idx] for idx, key in enumerate(order_list)}
        return bench(new_dict)
    return func

def get_init_configs(fun_to_evaluate, config_space, n_init, order_list, seed = 0, config_init = None):
    init_data = bo_tpe(fun_to_evaluate, config_space, 0, n_init, seed, config_init, just_fetch_information = True, reset_info = False)
    y0  = list(init_data["loss"])
    x0  = []
    for idx in range(n_init):
        this_x    = config_init[idx]
        new_array = []
        for key in order_list:
            new_array += [this_x[key]]
        x0 += [new_array]
    return x0, y0

def bo_skopt(fun_to_evaluate, config_space, n_runs, n_init, seed = 0, config_init = None, order_list = None):
    if hasattr(fun_to_evaluate, "reseed"):
        fun_to_evaluate.reseed(seed)
    fun_to_evaluate.reset_results()
    np.int = np.int32
    np.float = np.float64
    x0, y0  = get_init_configs(fun_to_evaluate, config_space, n_init, order_list, seed, config_init)
    obj     = wrapper_func(fun_to_evaluate, order_list)
    space   = obtain_space(config_space, order_list)

    # Perform Bayesian optimization
    result = gp_minimize(
        obj, # the objective function to minimize
        space,      # the search space
        x0 = x0,      # fixed initial points for x (two dimensions)
        y0 = y0,      # fixed initial points for y (objective values)
        n_calls = n_runs, # number of optimization steps
        n_initial_points = 0, 
        random_state = seed,
        acq_func='EI',
        acq_optimizer='lbfgs' 
    )
    return result, pd.DataFrame(fun_to_evaluate.all_results)