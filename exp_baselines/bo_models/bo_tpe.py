import sys
sys.path.append('exp_baselines/tpe_single')
from tpe.optimizer import TPEOptimizer
from typing import Dict
import pandas as pd
from typing import Any, Dict
from tpe.utils.constants import QuantileFunc

class CustomTPEOptimizer(TPEOptimizer):
    def __init__(self, config_init, **kwargs):
        super().__init__(**kwargs)
        self.config_init   = config_init
        self.n_config_init = 0

    def initial_sample(self) -> Dict[str, Any]:
        self.n_config_init += 1
        return self.config_init[self.n_config_init - 1]

def bo_tpe(fun_to_evaluate, config_space, n_runs, n_init, seed = 0, config_init = None, just_fetch_information = False, reset_info = True, order_list = None):
    #fun_to_evaluate  = partial(func_run, -1, data_dict)
    CONFIG_SPACE   = config_space
    MAX_EVALS      = n_runs + n_init
    N_INIT         = n_init
    LINEAR         = "linear"
    if reset_info:
        fun_to_evaluate.reset_results()

    opt = CustomTPEOptimizer(
        config_init  = config_init,
        obj_func     = fun_to_evaluate,
        config_space = CONFIG_SPACE,
        max_evals    = MAX_EVALS,
        n_init       = N_INIT,
        weight_func_choice="expected-improvement",
        quantile_func = QuantileFunc(choice=LINEAR, alpha=0.15),
        seed = seed,
        resultfile = "temp",
        magic_clip = True,
        magic_clip_exponent = 2.0,
        heuristic = "hyperopt",
        min_bandwidth_factor = 0.03,
    )
    opt.optimize()
    if not just_fetch_information:
        return opt.fetch_observations()["loss"].tolist(), pd.DataFrame(fun_to_evaluate.all_results.copy()   )

    else:
        return opt.fetch_observations()