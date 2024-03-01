import numpy as np
import optuna
from optuna.samplers import TPESampler

def fit_and_predict_with_GP(hp_constraints, X_train, y_train, X_test):
    '''Return predictions on surrogate model from GP.'''
    # NOTE: X_train, X_test are hyperparameter configurations, y_train is the corresponding validation error

    from smac.model.gaussian_process import GaussianProcess
    from ConfigSpace import ConfigurationSpace, Float, Integer
    from smac.model.gaussian_process.kernels import ConstantKernel, MaternKernel, ProductKernel
    from sklearn.preprocessing import StandardScaler

    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()

    # The ConfigSpace doesn't matter here, we just need to create a ConfigurationSpace object to use the GP
    # https://automl.github.io/SMAC3/main/_modules/smac/model/gaussian_process/gaussian_process.html#GaussianProcess
    cs = ConfigurationSpace(seed=42)
    for hp_name, hp_info in hp_constraints.items():
        type, transform, [hp_min, hp_max] = hp_info
        if type == 'int':
            if transform == 'log':
                cs.add_hyperparameter(Integer(hp_name, (hp_min, hp_max), log=True))
            else:
                cs.add_hyperparameter(Integer(hp_name, (hp_min, hp_max), log=False))
        elif type == 'float':
            if transform == 'log':
                cs.add_hyperparameter(Float(hp_name, (hp_min, hp_max), log=True))
            else:
                cs.add_hyperparameter(Float(hp_name, (hp_min, hp_max), log=False))

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()

    # standardize X_train, X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # kernel is based on default used here: 
    # https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
    # length_scale = 1 since we have standardized the data, \nu=2.5 is good heuristic starting point
    kernel = ProductKernel(ConstantKernel(), MaternKernel(length_scale=1.0, nu=2.5))

    if X_train.shape[0] > 10:
        gp = GaussianProcess(cs, kernel, normalize_y=False, n_restarts=20*int(X_train.shape[0]/10), seed=0)
    else:
        gp = GaussianProcess(cs, kernel, normalize_y=False, seed=0)
    gp.train(X_train, y_train)
    y_pred, y_std = gp.predict(X_test, covariance_type='std')

    return y_pred, y_std

def fit_and_predict_with_SMAC(hp_constraints, X_train, y_train, X_test):
    '''Return predictions on surrogate model from SMAC.'''

    from smac.model.random_forest import RandomForest
    from ConfigSpace import ConfigurationSpace, Float, Integer

    config_space_dict = {}

    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()

    cs = ConfigurationSpace(seed=42)
    for hp_name, hp_info in hp_constraints.items():
        type, transform, [hp_min, hp_max] = hp_info
        if type == 'int':
            if transform == 'log':
                cs.add_hyperparameter(Integer(hp_name, (hp_min, hp_max), log=True))
            else:
                cs.add_hyperparameter(Integer(hp_name, (hp_min, hp_max), log=False))
        elif type == 'float':
            if transform == 'log':
                cs.add_hyperparameter(Float(hp_name, (hp_min, hp_max), log=True))
            else:
                cs.add_hyperparameter(Float(hp_name, (hp_min, hp_max), log=False))

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()

    if X_train.shape[0] < 5:
        # for when we have <= 5 samples, to stop RF from just predicting a constant
        rf = RandomForest(cs, seed=0, min_samples_leaf=1, min_samples_split=1)
    else:
        rf = RandomForest(cs, seed=0)

    rf.train(X_train, y_train)
    y_pred, y_var = rf.predict(X_test, covariance_type='diagonal')  # RF only works with diagonal, need to calculate std manually
    y_std = np.sqrt(y_var)

    return y_pred, y_std




def custom_gamma(x: int) -> int:
    return min(int(np.ceil(0.25 * x)), 25)


def fit_and_predict_with_TPE(hp_constraints, X_train, y_train, X_test, top_pct, multivariate, lower_is_better):
    '''Sample from independent TPE sampler'''

    def custom_gamma(x: int) -> int:
        return min(int(np.ceil(top_pct * x)), 25)

    direction = 'minimize' if lower_is_better else 'maximize'
    sampler =  TPESampler(multivariate=multivariate, # doing independent sampling if False, else multivariate
                          consider_prior=True,
                          consider_endpoints=False, # clip to hyp range
                          n_startup_trials=0,
                          gamma=custom_gamma, # custom gamma
                          n_ei_candidates=1,
                          seed=69)
    
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
    for i in range(len(X_train)):
        study.add_trial(
            optuna.trial.create_trial(
                params=X_train.iloc[i].to_dict(),
                distributions=distribution,
                value=y_train.iloc[i].values[0]
            )
        )

    rel_search_space = sampler.infer_relative_search_space(study, None)

    X_test = X_test.to_dict(orient='records')
    X_test_ = {k: [] for k in X_test[0].keys()}
    for d in X_test:
        for k, v in d.items():
            X_test_[k].append(v)
            
    for k, v in X_test_.items():
        X_test_[k] = np.array(v)

    if multivariate:
        score = sampler._evaluate_relative(study, None, rel_search_space, X_test_)
    else:
        score = sampler.evaluate_independent(study, distribution, X_test_)

    return score




