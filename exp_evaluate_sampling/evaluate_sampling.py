import os
import argparse
import json
import optuna
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from bayesmark.bbox_utils import get_bayesmark_func
from llambo.acquisition_function import LLM_ACQ
from llambo.rate_limiter import RateLimiter
from exp_evaluate_sampling.evaluate_sampling_utils import sample_from_TPESampler, sample_from_RandomSampler, sample_from_GP
from exp_evaluate_sampling.metrics_utils import calculate_mahalanobis_dist, calculate_gen_var, calculate_loglikelihood
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)
rate_limiter = RateLimiter(max_tokens=100000, time_frame=60)

logger = logging.getLogger(__name__)

def setup_logging(log_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def obtain_n_configurations(hp_constraints, n, dataset, model, 
                            task_metric, task_type, lower_is_better):
    # run random sampled hyperaparameter configurations with optuna
    def objective(trial):
        config = {}
        for hp_name, hp_info in hp_constraints.items():
            use_log = hp_info[1] in ['log']
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

    # avoid execessive duplication of configs - means observed configurations are not diverse enough
    # this leads to rank deficiency in covariance matrix
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


def evaluate_proposals(candidates, observed, model, task_type, task_metric, 
                       dataset, lower_is_better, hp_constraints=None):

    model_ = get_bayesmark_func(model, task_type)
    candidate_scores = []
    for config in candidates:

        if hp_constraints is not None:
            for hyperparam, value in config.items():
                if hp_constraints[hyperparam][0] == 'int':
                    config[hyperparam] = int(value)


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
        candidate_scores.append(score)

    if task_metric == 'neg_mean_squared_error':
        candidate_scores = [-score for score in candidate_scores]
    
    # calculate average regret
    candidate_scores = np.array(candidate_scores)
    if lower_is_better:
        av_regret = np.mean(candidate_scores - global_best_score)
    else:
        av_regret = np.mean(global_best_score - candidate_scores)
    
    # normalize average regret
    av_regret /= np.abs(global_worst_score - global_best_score)

    # calculate best regret
    if lower_is_better:
        best_regret = candidate_scores.min() - global_best_score
    else:
        best_regret = global_best_score - candidate_scores.max()

    # normalize best regret
    best_regret = 0 if best_regret < 0 else best_regret
    best_regret /= np.abs(global_worst_score - global_best_score)

    # calculate spread in performance
    perf_spread = np.std(candidate_scores)

    # calculate mahalanobis distance
    ml_dist = calculate_mahalanobis_dist(hp_constraints, candidates, observed)

    # calculate determinant of covariance matrix
    gen_var = calculate_gen_var(hp_constraints, candidates, observed)

    # calculate nll
    ll = calculate_loglikelihood(hp_constraints, candidates, observed)

    return av_regret, best_regret, perf_spread, ml_dist, gen_var, ll, candidate_scores




TASK_MAP = {
    'breast': ['classification', 'accuracy'],
    'digits': ['classification', 'accuracy'],
    'wine': ['classification', 'accuracy'],
    'iris': ['classification', 'accuracy'],
    'diabetes': ['regression', 'neg_mean_squared_error'],
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RandomForest')
    parser.add_argument('--dataset', type=str, default='breast')
    parser.add_argument('--num_observed', type=int, default=10)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--engine', type=str)

    args = parser.parse_args()
    model = args.model
    dataset_name = args.dataset
    num_observed = args.num_observed
    num_seeds = args.num_seeds
    engine = args.engine

    # load hyperparameter config space
    with open(f'hp_configurations/bayesmark.json', 'r') as f:
        hp_constraints = json.load(f)[model]

    task_map = TASK_MAP[dataset_name]
    task_type = task_map[0]
    task_metric = task_map[1]


    # define result save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_fpath = f'{script_dir}/results/evaluate_sampling/{dataset_name}/{model}/{num_observed}.json'
    if not os.path.exists(os.path.dirname(save_res_fpath)):
        os.makedirs(os.path.dirname(save_res_fpath))
    # define logging directory
    logging_fpath = f'{script_dir}/logs/evaluate_sampling/{dataset_name}/{model}/{num_observed}.log'
    if not os.path.exists(os.path.dirname(logging_fpath)):
        os.makedirs(os.path.dirname(logging_fpath))
    setup_logging(logging_fpath)

    logger.info('='*200)
    logger.info(f'Evaluating sampling performance on {dataset_name} with {model} and {num_observed} observed configurations... Executing {num_seeds} runs.')
    logger.info('='*200)

    # load dataset
    pickle_fpath = f'bayesmark/data/{dataset_name}.pickle'
    with open(pickle_fpath, 'rb') as f:
        dataset = pickle.load(f)

    results = {}
    results['TPE_IN'] = {'av_regret': [], 'best_regret': [], 'score_spread': [], 'ml_dist': [], 'gen_var': [], 'll': [],
                      'proposed_points': [], 'proposed_points_evaluation': []}
    results['TPE_MULTI'] = {'av_regret': [], 'best_regret': [], 'score_spread': [], 'ml_dist': [], 'gen_var': [], 'll': [],
                      'proposed_points': [], 'proposed_points_evaluation': []}
    results['RANDOM'] = {'av_regret': [], 'best_regret': [], 'score_spread': [], 'ml_dist': [], 'gen_var': [], 'll': [],
                        'proposed_points': [], 'proposed_points_evaluation': []}
    results['LLAMBO'] = {'av_regret': [], 'best_regret': [], 'score_spread': [], 'ml_dist': [], 'gen_var': [], 'll': [],
                        'proposed_points': [], 'proposed_points_evaluation': []}

    lower_is_better = False if task_metric == 'accuracy' else True


    logger.info(f'Collecting configurations - this might take a while...')
    sampled_configs, sampled_scores = obtain_n_configurations(hp_constraints, 100, dataset, model, 
                                                              task_metric=task_metric, task_type=task_type, lower_is_better=lower_is_better)
    

    with open('bayesmark/data/global_perf.json', 'r') as f:
        global_perf = json.load(f)
    
    global_best_score = global_perf[dataset_name]['global_min'] if lower_is_better else global_perf[dataset_name]['global_max']
    global_worst_score = global_perf[dataset_name]['global_max'] if lower_is_better else global_perf[dataset_name]['global_min']
    logger.info(f'Global best score: {global_best_score:.4f}, global worst score: {global_worst_score:.4f}')

    tot_llm_cost = 0
    for seed in range(num_seeds):
        logger.info('='*200)
        logger.info(f'Evaluating sampling with seed {seed}...')

        observed_configs, observed_fvals = sample_n_configurations(sampled_configs, sampled_scores, num_observed, seed=seed)

        # sample candidates from Independent TPE
        model_covariance = False
        candidates = sample_from_TPESampler(hp_constraints, lower_is_better, 20, 
                                            model_covariance, observed_configs, observed_fvals)
        # evaluate candidates
        evaluation = evaluate_proposals(candidates, observed_configs, model, task_type, task_metric, dataset, lower_is_better, hp_constraints)
        av_regret, best_regret, perf_spread, ml_dist, gen_var, ll, candidate_scores = evaluation
        logger.info(f'[TPE_IN] Average regret: {av_regret:.4f}, Best regret: {best_regret:.4f}, '
                    f'Score spread: {perf_spread:.4f}, Mahalanobis distance: {ml_dist:.4f}, '
                    f'Gen var: {gen_var:.4f}, Log likelihood: {ll:.4f}')
        results['TPE_IN']['av_regret'].append(av_regret)
        results['TPE_IN']['best_regret'].append(best_regret)
        results['TPE_IN']['score_spread'].append(perf_spread)
        results['TPE_IN']['ml_dist'].append(ml_dist)
        results['TPE_IN']['gen_var'].append(gen_var)
        results['TPE_IN']['ll'].append(ll)
        results['TPE_IN']['proposed_points'].append(candidates)
        results['TPE_IN']['proposed_points_evaluation'].append(candidate_scores.tolist())



        # sample candidates from Multivariate TPE
        model_covariance = True
        candidates = sample_from_TPESampler(hp_constraints, lower_is_better, 20,
                                            model_covariance, observed_configs, observed_fvals)
        # evaluate candidates
        evaluation = evaluate_proposals(candidates, observed_configs, model, task_type, task_metric, dataset, lower_is_better, hp_constraints)
        av_regret, best_regret, perf_spread, ml_dist, gen_var, ll, candidate_scores = evaluation
        logger.info(f'[TPE_MULTI] Average regret: {av_regret:.4f}, Best regret: {best_regret:.4f}, '
                    f'Score spread: {perf_spread:.4f}, Mahalanobis distance: {ml_dist:.4f}, '
                    f'Gen var: {gen_var:.4f}, Log likelihood: {ll:.4f}')
        results['TPE_MULTI']['av_regret'].append(av_regret)
        results['TPE_MULTI']['best_regret'].append(best_regret)
        results['TPE_MULTI']['score_spread'].append(perf_spread)
        results['TPE_MULTI']['ml_dist'].append(ml_dist)
        results['TPE_MULTI']['gen_var'].append(gen_var)
        results['TPE_MULTI']['ll'].append(ll)
        results['TPE_MULTI']['proposed_points'].append(candidates)
        results['TPE_MULTI']['proposed_points_evaluation'].append(candidate_scores.tolist())
        


        # sample candidates from Random sampler
        candidates = sample_from_RandomSampler(hp_constraints, 20, seed=seed)
        # evaluate candidates
        evaluation = evaluate_proposals(candidates, observed_configs, model, task_type, task_metric, dataset, lower_is_better, hp_constraints)
        av_regret, best_regret, perf_spread, ml_dist, gen_var, ll, candidate_scores = evaluation
        logger.info(f'[RANDOM] Average regret: {av_regret:.4f}, Best regret: {best_regret:.4f}, '
                    f'Score spread: {perf_spread:.4f}, Mahalanobis distance: {ml_dist:.4f}, '
                    f'Gen var: {gen_var:.4f}, Log likelihood: {ll:.4f}')
        results['RANDOM']['av_regret'].append(av_regret)
        results['RANDOM']['best_regret'].append(best_regret)
        results['RANDOM']['score_spread'].append(perf_spread)
        results['RANDOM']['ml_dist'].append(ml_dist)
        results['RANDOM']['gen_var'].append(gen_var)
        results['RANDOM']['ll'].append(ll)
        results['RANDOM']['proposed_points'].append(candidates)
        results['RANDOM']['proposed_points_evaluation'].append(candidate_scores.tolist())


        # sample candidates from LLM sampler
        task_context = {}
        task_context['model'] = model
        task_context['task'] = task_type
        task_context['tot_feats'] = dataset['train_x'].shape[1]
        task_context['cat_feats'] = 0
        task_context['num_feats'] = dataset['train_x'].shape[1]
        task_context['n_classes'] = len(np.unique(dataset['train_y']))
        task_context['metric'] = task_metric
        task_context['num_samples'] = dataset['train_x'].shape[0]
        task_context['hyperparameter_constraints'] = hp_constraints
        # sample candidates
        LLM_Sampler = LLM_ACQ(task_context, n_candidates=20, n_templates=5, lower_is_better=lower_is_better, rate_limiter=rate_limiter, chat_engine=engine)
        candidates, tot_cost, time_taken = LLM_Sampler.get_candidate_points(observed_configs, observed_fvals, alpha=-0.2)
        # evaluate candidates
        candidates = candidates.to_dict('records')
        evaluation = evaluate_proposals(candidates, observed_configs, model, task_type, task_metric, dataset, lower_is_better, hp_constraints)
        av_regret, best_regret, perf_spread, ml_dist, gen_var, ll, candidate_scores = evaluation
        logger.info(f'[LLAMBO] Average regret: {av_regret:.4f}, Best regret: {best_regret:.4f}, '
                    f'Score spread: {perf_spread:.4f}, Mahalanobis distance: {ml_dist:.4f}, '
                    f'Gen var: {gen_var:.4f}, Log likelihood: {ll:.4f}')
        results['LLAMBO']['av_regret'].append(av_regret)
        results['LLAMBO']['best_regret'].append(best_regret)
        results['LLAMBO']['score_spread'].append(perf_spread)
        results['LLAMBO']['ml_dist'].append(ml_dist)
        results['LLAMBO']['gen_var'].append(gen_var)
        results['LLAMBO']['ll'].append(ll)
        results['LLAMBO']['proposed_points'].append(candidates)
        results['LLAMBO']['proposed_points_evaluation'].append(candidate_scores.tolist())
        # track costs
        tot_llm_cost += tot_cost

        # save results
        with open(save_res_fpath, 'w') as f:
            json.dump(results, f, indent=4)

    logger.info('='*200)
    logger.info(f'[LLAMBO] {seed+1} evaluation runs complete! Total cost: ${tot_llm_cost:.4f}')
