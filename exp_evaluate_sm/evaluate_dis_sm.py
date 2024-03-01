import os
import argparse
import json
import optuna
import logging
import pickle
import asyncio
import numpy as np
import pandas as pd
from scipy.stats import norm
from bayesmark.bbox_utils import get_bayesmark_func
from llambo.discriminative_sm import LLM_DIS_SM
from llambo.rate_limiter import RateLimiter
from sklearn.metrics import get_scorer
from sklearn.metrics import mean_squared_error, r2_score
from uncertainty_toolbox import metrics_calibration as cal
from exp_evaluate_sm.evaluate_sm_utils import fit_and_predict_with_GP, fit_and_predict_with_SMAC
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

rate_limiter = RateLimiter(max_tokens=100000, time_frame=60)

logger = logging.getLogger(__name__)

def setup_logging(log_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def obtain_n_configurations(hp_constraints, n, dataset, model, task_metric, task_type, lower_is_better):
    # run random sampled hyperaparameter configurations with optuna
    def objective(trial):
        config = {}
        for hp_name, hp_info in hp_constraints.items():
            use_log = hp_info[1] in ['log', 'logit']
            if hp_info[0] == 'int':
                config[hp_name] = trial.suggest_int(hp_name, hp_info[2][0], hp_info[2][1], log=use_log)
            elif hp_info[0] == 'float':
                config[hp_name] = trial.suggest_float(hp_name, hp_info[2][0], hp_info[2][1], log=use_log)
            else:
                raise ValueError(f'Unknown hyperparameter type: {hp_info[0]}')
            
        model_ = get_bayesmark_func(model, task_type)

        train_x = dataset['train_x']
        test_x = dataset['test_x']
        
        if task_type == 'regression':
            # standardize y
            y_mean = dataset['train_y'].mean()
            y_std = dataset['train_y'].std()
            train_y = (dataset['train_y'] - y_mean) / y_std
            test_y = (dataset['test_y'] - y_mean) / y_std
        else:
            train_y = dataset['train_y']
            test_y = dataset['test_y']

        predictor = model_(**config, random_state=42)
        predictor.fit(train_x, train_y)
        scorer = get_scorer(task_metric)
        score = scorer(predictor, test_x, test_y)
        return score
    
    configs = []
    scores = []
    for i in range(5):
        direction = 'minimize' if lower_is_better else 'maximize'
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42+i), direction=direction)
        study.optimize(objective, n_trials=n)

        # get all configurations and scores
        for trial in study.trials:
            configs.append(trial.params)
            if task_metric == 'neg_mean_squared_error':
                scores.append(-trial.value)
            else:
                scores.append(trial.value)


    configs = pd.DataFrame(configs)
    scores = pd.DataFrame(scores, columns=['score'])

    return configs, scores


def sample_n_configurations(configs, scores, n, seed, existing_config=None):
    '''Sample n configurations from configs and scores'''
    number_sampled = 0
    iter_i = 0

    sampled_configs = pd.DataFrame()
    sampled_scores = pd.DataFrame()
    
    # get all unique values in scores
    unique_scores = scores['score'].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_scores)

    scores['score_rank'] = scores['score'].apply(lambda x: np.where(unique_scores == x)[0][0])

    # avoid execessive duplication of configs - makes prediction task trivial!
    while number_sampled < n:
        # randomly sample from each unique score
        sample_index = scores.groupby('score_rank').apply(lambda x: x.sample(1, random_state=seed+iter_i)).index.get_level_values(1)
        # get sampled configs and scores
        sampled_configs = pd.concat([sampled_configs, configs.iloc[sample_index]], axis=0)
        sampled_scores = pd.concat([sampled_scores, scores[['score']].iloc[sample_index]], axis=0)
        sampled_configs = sampled_configs.reset_index(drop=True)
        sampled_scores = sampled_scores.reset_index(drop=True)

        if existing_config is not None:
            drop_index = []
            for i in range(sampled_configs.shape[0]):
                row = sampled_configs.iloc[i, :]
                if (existing_config == row).all(1).any():
                    drop_index.append(i)
            
            sampled_configs = sampled_configs.drop(drop_index)
            sampled_scores = sampled_scores.drop(drop_index)
            sampled_configs = sampled_configs.reset_index(drop=True)
            sampled_scores = sampled_scores.reset_index(drop=True)

        # remove duplicates
        duplicate_index = sampled_configs[sampled_configs.duplicated()].index
        sampled_configs = sampled_configs.drop(duplicate_index)
        sampled_scores = sampled_scores.drop(duplicate_index)
        sampled_configs = sampled_configs.reset_index(drop=True)
        sampled_scores = sampled_scores.reset_index(drop=True)

        iter_i += 1
        number_sampled = len(sampled_configs)

    sampled_configs = sampled_configs.head(n)
    sampled_scores = sampled_scores.head(n)

    return sampled_configs, sampled_scores


def evaluate_posterior(fval_pred, fval_pred_std, fval_true, 
                       f_best, lower_is_better):
    '''Calculate RMSE, NLL, MACE, regret to evaluate posterior prediction and uncertainty'''
    assert type(fval_pred) == type(fval_pred_std) == type(fval_true) == np.ndarray
    
    if fval_pred.shape != 1:
        fval_pred = fval_pred.squeeze()
    if fval_pred_std.shape != 1:
        fval_pred_std = fval_pred_std.squeeze()
    if fval_true.shape != 1:
        fval_true = fval_true.squeeze()
        
    assert len(fval_pred.shape) == 1 and len(fval_pred_std.shape) == 1 and len(fval_true.shape) == 1
    
    # calculate normalized RMSE
    rmse = mean_squared_error(fval_true, fval_pred, squared=False)
    rmse /= np.abs(fval_true.max() - fval_true.min())

    # calculate r^2
    r2 = r2_score(fval_true, fval_pred)
    
    # calculate log predictive density - catch explosive values
    fval_pred_std[fval_pred_std < 1e-12] = 1e-12
    nll = 0.5 * np.log(2 * np.pi * fval_pred_std**2) + 0.5 * ((fval_true - fval_pred) / fval_pred_std)**2
    # nll = np.mean(nll[nll<10])
    nll = np.mean(nll[nll<100])
    
    # calculate empirical coverage
    alpha = 0.68 # for 1 sigma
    z = np.abs(np.percentile(np.random.randn(1000000), (1-alpha)*100/2))
    lower_bound = fval_pred - z * fval_pred_std
    upper_bound = fval_pred + z * fval_pred_std
    in_interval = np.sum((fval_true >= lower_bound) & (fval_true <= upper_bound))
    observed_coverage = in_interval / fval_true.shape[0]

    # calculate MACE (this is very noisy at low sample sizes)
    mace = cal.mean_absolute_calibration_error(fval_pred, fval_pred_std, fval_true)

    # calculate sharpness
    sharpness = cal.sharpness(fval_pred_std)
    
    # compute expected improvement (EI)
    if lower_is_better:
        delta = -1*(fval_pred - f_best)
    else:
        delta = fval_pred - f_best
    with np.errstate(divide='ignore'):
        z = delta / fval_pred_std
    ei = np.where(fval_pred_std > 0, delta * norm.cdf(z) + fval_pred_std * norm.pdf(z), 0)
    idx = np.argmax(ei)

    # calculate normalized regret
    if lower_is_better:
        regret = fval_true[idx] - fval_true.min()
    else:
        regret = fval_true.max() - fval_true[idx]

    regret /= np.abs( max(fval_true.max(), observed_fvals.max().item()) -
                     min(fval_true.min(), observed_fvals.min().item()))
                     
    return rmse, r2, nll, mace, sharpness, observed_coverage, regret



TASK_MAP = {
    'breast': ['classification', 'accuracy'],
    'digits': ['classification', 'accuracy'],
    'wine': ['classification', 'accuracy'],
    'iris': ['classification', 'accuracy'],
    'diabetes': ['regression', 'neg_mean_squared_error'],
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_observed', type=int)
    parser.add_argument('--num_seeds', type=int)
    parser.add_argument('--engine', type=str)

    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    num_observed = args.num_observed
    num_seeds = args.num_seeds
    engine = args.engine

    # load hyperparameter config space
    with open(f'hp_configurations/bayesmark.json', 'r') as f:
        hp_constraints = json.load(f)[model]

    task_map = TASK_MAP[dataset]
    task_type = task_map[0]
    task_metric = task_map[1]


    # define result save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_fpath = f'{script_dir}/results/evaluate_dis_sm/{dataset}/{model}/{num_observed}.json'
    if not os.path.exists(os.path.dirname(save_res_fpath)):
        os.makedirs(os.path.dirname(save_res_fpath))
    # define logging directory
    logging_fpath = f'{script_dir}/logs/evaluate_dis_sm/{dataset}/{model}/{num_observed}.log'
    if not os.path.exists(os.path.dirname(logging_fpath)):
        os.makedirs(os.path.dirname(logging_fpath))
    setup_logging(logging_fpath)

    logger.info('='*200)
    logger.info(f'Evaluating Disriminative SM performance on {dataset} with {model} and {num_observed} observed configurations... Running {num_seeds} runs.')
    logger.info('='*200)

    # load dataset
    pickle_fpath = f'bayesmark/data/{dataset}.pickle'
    with open(pickle_fpath, 'rb') as f:
        dataset = pickle.load(f)

    results = {}
    results['GP'] = {'rmse': [], 'r2': [], 
                     'nll': [], 'mace': [], 'sharpness': [], 'observed_coverage': [],
                     'regret': [], 'y_pred': [], 'y_std': [], 'y_true': []}
    results['SMAC'] = {'rmse': [], 'r2': [], 
                     'nll': [], 'mace': [], 'sharpness': [], 'observed_coverage': [],
                     'regret': [], 'y_pred': [], 'y_std': [], 'y_true': []}
    results['LLAMBO'] = {'rmse': [], 'r2': [], 
                     'nll': [], 'mace': [], 'sharpness': [], 'observed_coverage': [],
                     'regret': [], 'y_pred': [], 'y_std': [], 'y_true': [],
                     'llm_query_cost': [], 'llm_query_time': []}
    results['LLAMBO_VANILLA'] = {'rmse': [], 'r2': [], 
                     'nll': [], 'mace': [], 'sharpness': [], 'observed_coverage': [],
                     'regret': [], 'y_pred': [], 'y_std': [], 'y_true': [],
                     'llm_query_cost': [], 'llm_query_time': []}
    lower_is_better = False if task_metric == 'accuracy' else True

    logger.info(f'Collecting configurations - this might take a while...')
    sampled_configs, sampled_scores = obtain_n_configurations(hp_constraints, 100, dataset, model, 
                                                              task_metric=task_metric, task_type=task_type, lower_is_better=lower_is_better)


    tot_llm_cost = 0
    for seed in range(num_seeds):
        logger.info('='*200)
        logger.info(f'Evaluating SM with seed {seed}...')

        observed_configs, observed_fvals = sample_n_configurations(sampled_configs, sampled_scores, num_observed, seed=seed)

        candidate_configs, candidate_fvals = sample_n_configurations(sampled_configs, sampled_scores, 10, 
                                                                     seed=42)
        
        f_best = observed_fvals.min().item() if lower_is_better else observed_fvals.max().item()

        # evaluate GP
        y_pred, y_std = fit_and_predict_with_GP(hp_constraints, observed_configs, observed_fvals, candidate_configs)
        scores = evaluate_posterior(y_pred, y_std, candidate_fvals.to_numpy(), f_best, lower_is_better)
        rmse, r2, nll, mace, sharpness, observed_coverage, regret = scores
        logger.info(f"[GP] RMSE: {rmse:.4f}, R2 score: {r2:.4f}, NLL: {nll:.4f}, "
                    f"Coverage: {observed_coverage:.4f}, MACE: {mace:.4f}, Sharpness: {sharpness:.4f}, Regret: {regret:.4f}")
        results['GP']['rmse'].append(rmse)
        results['GP']['r2'].append(r2)
        results['GP']['nll'].append(nll)
        results['GP']['mace'].append(mace)
        results['GP']['sharpness'].append(sharpness)
        results['GP']['observed_coverage'].append(observed_coverage)
        results['GP']['regret'].append(regret)
        results['GP']['y_pred'].append(y_pred.squeeze().tolist())
        results['GP']['y_std'].append(y_std.squeeze().tolist())
        results['GP']['y_true'].append(candidate_fvals.to_numpy().squeeze().tolist())


        # evaluate SMAC
        y_pred, y_std = fit_and_predict_with_SMAC(hp_constraints, observed_configs, observed_fvals, candidate_configs)
        scores = evaluate_posterior(y_pred, y_std, candidate_fvals.to_numpy(), f_best, lower_is_better)
        rmse, r2, nll, mace, sharpness, observed_coverage, regret = scores
        logger.info(f"[SMAC] RMSE: {rmse:.4f}, R2 score: {r2:.4f}, NLL: {nll:.4f}, "
                    f"Coverage: {observed_coverage:.4f}, MACE: {mace:.4f}, Sharpness: {sharpness:.4f}, Regret: {regret:.4f}")
        results['SMAC']['rmse'].append(rmse)
        results['SMAC']['r2'].append(r2)
        results['SMAC']['nll'].append(nll)
        results['SMAC']['mace'].append(mace)
        results['SMAC']['sharpness'].append(sharpness)
        results['SMAC']['observed_coverage'].append(observed_coverage)
        results['SMAC']['regret'].append(regret)
        results['SMAC']['y_pred'].append(y_pred.squeeze().tolist())
        results['SMAC']['y_std'].append(y_std.squeeze().tolist())
        results['SMAC']['y_true'].append(candidate_fvals.to_numpy().squeeze().tolist())


        # evaluate LLAMBO - calibrated
        task_context = {}
        task_context['model'] = model
        task_context['task'] = task_type
        task_context['tot_feats'] = dataset['train_x'].shape[1]
        task_context['cat_feats'] = 0
        task_context['num_feats'] = dataset['train_x'].shape[1]
        task_context['n_classes'] = len(np.unique(dataset['train_y']))
        task_context['metric'] = 'mean squared error' if task_metric == 'neg_mean_squared_error' else task_metric
        task_context['num_samples'] = dataset['train_x'].shape[0]
        task_context['hyperparameter_constraints'] = hp_constraints


        LLM_SM = LLM_DIS_SM(task_context, n_gens=10, lower_is_better=lower_is_better, 
                                bootstrapping=False, n_templates=5, use_recalibration=False,
                                verbose=False, rate_limiter=rate_limiter, chat_engine=engine)
        y_mean, y_std, cost, time_taken = asyncio.run(LLM_SM._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs))
        scores = evaluate_posterior(y_mean, y_std, candidate_fvals.to_numpy(), f_best, lower_is_better)
        rmse, r2, nll, mace, sharpness, observed_coverage, regret = scores
        logger.info(f"[LLAMBO] RMSE: {rmse:.4f}, R2 score: {r2:.4f}, NLL: {nll:.4f}, "
                    f"Coverage: {observed_coverage:.4f}, MACE: {mace:.4f}, Sharpness: {sharpness:.4f}, Regret: {regret:.4f} | Cost: ${cost:.4f}, Time: {time_taken:.4f}s")
        results['LLAMBO']['rmse'].append(rmse)
        results['LLAMBO']['r2'].append(r2)
        results['LLAMBO']['nll'].append(nll)
        results['LLAMBO']['mace'].append(mace)
        results['LLAMBO']['sharpness'].append(sharpness)
        results['LLAMBO']['observed_coverage'].append(observed_coverage)
        results['LLAMBO']['regret'].append(regret)
        results['LLAMBO']['y_pred'].append(y_mean.squeeze().tolist())
        results['LLAMBO']['y_std'].append(y_std.squeeze().tolist())
        results['LLAMBO']['y_true'].append(candidate_fvals.to_numpy().squeeze().tolist())
        results['LLAMBO']['llm_query_cost'].append(cost)
        results['LLAMBO']['llm_query_time'].append(time_taken)

        tot_llm_cost += cost


        # evaluate LLAMBO - vanilla
        LLM_SM = LLM_DIS_SM(task_context, n_gens=10, lower_is_better=lower_is_better, 
                                bootstrapping=False, n_templates=1, use_recalibration=False,
                                verbose=False, rate_limiter=rate_limiter, chat_engine=engine)
        y_mean, y_std, cost, time_taken = asyncio.run(LLM_SM._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs))
        scores = evaluate_posterior(y_mean, y_std, candidate_fvals.to_numpy(), f_best, lower_is_better)
        rmse, r2, nll, mace, sharpness, observed_coverage, regret = scores
        logger.info(f"[LLAMBO_VANILLA] RMSE: {rmse:.4f}, R2 score: {r2:.4f}, NLL: {nll:.4f}, "
                    f"Coverage: {observed_coverage:.4f}, MACE: {mace:.4f}, Sharpness: {sharpness:.4f}, Regret: {regret:.4f} | Cost: ${cost:.4f}, Time: {time_taken:.4f}s")
        results['LLAMBO_VANILLA']['rmse'].append(rmse)
        results['LLAMBO_VANILLA']['r2'].append(r2)
        results['LLAMBO_VANILLA']['nll'].append(nll)
        results['LLAMBO_VANILLA']['mace'].append(mace)
        results['LLAMBO_VANILLA']['sharpness'].append(sharpness)
        results['LLAMBO_VANILLA']['observed_coverage'].append(observed_coverage)
        results['LLAMBO_VANILLA']['regret'].append(regret)
        results['LLAMBO_VANILLA']['y_pred'].append(y_mean.squeeze().tolist())
        results['LLAMBO_VANILLA']['y_std'].append(y_std.squeeze().tolist())
        results['LLAMBO_VANILLA']['y_true'].append(candidate_fvals.to_numpy().squeeze().tolist())
        results['LLAMBO_VANILLA']['llm_query_cost'].append(cost)
        results['LLAMBO_VANILLA']['llm_query_time'].append(time_taken)
        
        tot_llm_cost += cost


        # save results
        with open(save_res_fpath, 'w') as f:
            json.dump(results, f, indent=4)

    logger.info('='*200)
    logger.info(f'[LLAMBO] {num_seeds} evaluation runs complete! Total cost: ${tot_llm_cost:.4f}')
