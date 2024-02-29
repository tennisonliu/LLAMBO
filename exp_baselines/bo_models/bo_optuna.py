from typing import Callable
import ConfigSpace as CS

import pandas as pd
from bo_models.bo_tpe import bo_tpe
import optuna

def wrapper_func(bench: Callable, config_space: CS.ConfigurationSpace) -> Callable:
    def func(trial: optuna.Trial) -> float:
        eval_config = {}
        for hp in config_space.get_hyperparameters():
            name = hp.name
            if isinstance(hp, CS.CategoricalHyperparameter):
                eval_config[name] = trial.suggest_categorical(name=name, choices=hp.choices)
            elif isinstance(hp, CS.UniformFloatHyperparameter):
                eval_config[name] = trial.suggest_float(name=name, low=hp.lower, high=hp.upper, log=hp.log)
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                eval_config[name] = trial.suggest_int(name=name, low=hp.lower, high=hp.upper)
            else:
                raise TypeError(f"{type(type(hp))} is not supported")
        return bench(eval_config)
    return func


def add_init_configs_to_study(
    study,
    bench,
    config_space,
    seed,
    n_init,
    config_init):

    #df.to_dict(orient='records') df.to_dict(orient='records')
    data = bo_tpe(bench, config_space, 0, n_init, seed, config_init, just_fetch_information = True, reset_info = False)

    distributions, names = dict(), []
    for hp in config_space.get_hyperparameters():
        name = hp.name
        names.append(name)
        if isinstance(hp, CS.CategoricalHyperparameter):
            distributions[name] = optuna.distributions.CategoricalDistribution(choices=hp.choices)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            distributions[name] = optuna.distributions.FloatDistribution(low=hp.lower, high=hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            distributions[name] = optuna.distributions.IntDistribution(low=hp.lower, high=hp.upper)
        else:
            raise TypeError(f"{type(type(hp))} is not supported")

    study.add_trials([
        optuna.create_trial(
            params={name: data[name][i] for name in names},
            distributions = distributions,
            value = data["loss"][i],
        )
        for i in range(n_init)
    ])


def bo_optuna(fun_to_evaluate, config_space, n_runs, n_init, seed = 0, config_init = None, order_list = None):
    if hasattr(fun_to_evaluate, "reseed"):
        fun_to_evaluate.reseed(seed)
    fun_to_evaluate.reset_results()
    sampler = optuna.samplers.TPESampler(multivariate = True, seed = seed, n_startup_trials = n_init,
                                        n_ei_candidates = 20)
    study   = optuna.create_study(sampler = sampler)
    add_init_configs_to_study(study = study, bench = fun_to_evaluate, config_space=config_space,
                              seed=seed, n_init = n_init, config_init = config_init)

    study.optimize(wrapper_func(bench = fun_to_evaluate, config_space = config_space), n_trials = n_runs)
    vals = [trial.value for trial in study.trials]

    print(vals)
    return vals, pd.DataFrame(fun_to_evaluate.all_results)

