import optuna
import torch
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def sample_from_GP(hp_constraints, lower_is_better, num_candidates, observed_configs, observed_fvals):
    '''Sample from GP'''
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # get hp_constraints as bounds
    bounds = []
    for _, hp_info in hp_constraints.items():
        _, _, [hp_min, hp_max] = hp_info
        bounds.append([hp_min, hp_max])
    
    bounds = np.array(bounds)
    bounds = bounds.T

    assert bounds.shape == (2, len(hp_constraints)), 'bounds shape incorrect'
    
    # min-max normalize observed_configs
    observed_configs = (observed_configs - bounds[0]) / (bounds[1] - bounds[0])

    # standardize observed_fvals
    observed_fvals = observed_fvals - observed_fvals.mean()
    observed_fvals = observed_fvals / observed_fvals.std()

    train_X = torch.tensor(observed_configs.values)
    train_Y = torch.tensor(observed_fvals.values)

    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # bounds is just [0, 1] after min-max
    bounds_ = np.array([[0.0, 1.0] for _ in range(len(hp_constraints))])
    bounds_ = bounds_.T
    bounds_ = torch.tensor(bounds_, dtype=torch.float)

    best_f = train_Y.min() if lower_is_better else train_Y.max()

    EI = ExpectedImprovement(gp, best_f=best_f, maximize=(not lower_is_better))
    new_x, _ = optimize_acqf(
                    acq_function=EI,
                    bounds=bounds_,
                    q=1,
                    num_restarts=num_candidates,
                    raw_samples=num_candidates,  # for initializating the optimization
                    return_best_only=False,
                    options={"batch_limit": 5, "maxiter": 100}
                )
    
    new_x = new_x.numpy().squeeze()
    new_x = new_x * (bounds[1] - bounds[0])
    new_x = new_x + bounds[0]

    candidates = []
    for i in range(num_candidates):
        candidate = {}
        for j, hp_name in enumerate(hp_constraints.keys()):
            candidate[hp_name] = new_x[i, j]
        candidates.append(candidate)

    return candidates




def custom_gamma(x: int) -> int:
    return min(int(np.ceil(0.3 * x)), 25)

assert custom_gamma(3) == 1, 'gamma ratio incorrect'

def sample_from_TPESampler(hp_constraints, lower_is_better, num_candidates, model_covariance,
                    observed_configs, observed_fvals):
    '''Sample from independent TPE sampler'''
    direction = 'minimize' if lower_is_better else 'maximize'
    sampler =  TPESampler(multivariate=model_covariance, # doing independent sampling if False, else multivariate
                          consider_endpoints=True, # clip to hyp range
                          n_startup_trials=0, # no random sampling
                          gamma=custom_gamma, # custom gamma
                          n_ei_candidates=1, # only sample 1 candidate, we will calculate EI manually
                          seed=42)
    
    study = optuna.create_study(direction=direction, sampler=sampler)

    # create distribution to sample from
    distribution = {}
    for hp_name, hp_info in hp_constraints.items():
        type, transform, [hp_min, hp_max] = hp_info
        assert transform in ['log', 'logit', 'linear']
        if type == 'int':
            if transform == ['log']:
                distribution[hp_name] = optuna.distributions.IntDistribution(hp_min, hp_max, log=True)
            else:
                distribution[hp_name] = optuna.distributions.IntDistribution(hp_min, hp_max, log=False)
        elif type == 'float':
            if transform in ['log']:
                distribution[hp_name] = optuna.distributions.FloatDistribution(hp_min, hp_max, log=True)
            else:
                distribution[hp_name] = optuna.distributions.FloatDistribution(hp_min, hp_max, log=False)

    # add observed trials
    for i in range(len(observed_configs)):
        study.add_trial(
            optuna.trial.create_trial(
                params=observed_configs.iloc[i].to_dict(),
                distributions=distribution,
                value=observed_fvals.iloc[i].values[0]
            )
        )
    
    # sample candidates, 1 at a time
    candidates = []
    for i in range(num_candidates):
        trial = study.ask(distribution)
        candidates.append(trial.params)

    return candidates

    
def sample_from_RandomSampler(hp_constraints, num_candidates, seed):
    '''Sample from independent Random sampler'''
    sampler = RandomSampler(seed=42+seed)
    study = optuna.create_study(sampler=sampler)
    
    # create distribution to sample from
    distribution = {}
    for hp_name, hp_info in hp_constraints.items():
        type, transform, [hp_min, hp_max] = hp_info
        assert transform in ['log', 'logit', 'linear']
        if type == 'int':
            if transform == ['log']:
                distribution[hp_name] = optuna.distributions.IntDistribution(hp_min, hp_max, log=True)
            else:
                distribution[hp_name] = optuna.distributions.IntDistribution(hp_min, hp_max, log=False)
        elif type == 'float':
            if transform in ['log']:
                distribution[hp_name] = optuna.distributions.FloatDistribution(hp_min, hp_max, log=True)
            else:
                distribution[hp_name] = optuna.distributions.FloatDistribution(hp_min, hp_max, log=False)
    candidates = []
    for i in range(num_candidates):
        trial = study.ask(distribution)
        candidates.append(trial.params)

    return candidates

    

