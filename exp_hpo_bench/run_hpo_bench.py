import os
import pickle
import openml
import json
import argparse
import logging
import pandas as pd
import numpy as np
from llambo.llambo import LLAMBO
from hpo_bench.tabular_benchmarks import HPOBench


import warnings
# ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)

def setup_logging(log_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
DATASET_MAP = {
    "credit_g": [0, 31],    # [dataset id, openml task id]
    "vehicle": [1, 53],
    "kc1": [2, 3917],
    "phoneme": [3, 9952],
    "blood_transfusion": [4, 10101],
    "australian": [5, 146818],
    "car": [6, 146821],
    "segment": [7, 146822],
}

MODEL_MAP = {
    'rf': 'Random Forest',
    'nn': 'Multilayer Perceptron',
    'xgb': 'XGBoost'
}


class HPOExpRunner:
    def __init__(self, model, dataset, seed):
        self.hpo_bench = HPOBench(model, dataset)
        self.seed = seed
        self.config_path = f'hpo_bench/configs/{model}/config{seed}.json'
    
    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: list of dictionaries, each dictionary is a point to be evaluated
        '''
        # load initial configs
        with open(self.config_path, 'r') as f:
            configs = json.load(f)

        assert isinstance(configs, list)
        init_configs = []
        for i, config in enumerate(configs):
            assert isinstance(config, dict)
            
            if i < n_samples:
                init_configs.append(self.hpo_bench.ordinal_to_real(config))
        
        assert len(init_configs) == n_samples

        return init_configs
    
    def _find_nearest_neighbor(self, config):
        discrete_grid = self.hpo_bench._value_range
        nearest_config = {}
        for key in config:
            if key in discrete_grid:
                # Find the nearest value in the grid for the current key
                nearest_value = min(discrete_grid[key], key=lambda x: abs(x - config[key]))
                nearest_config[key] = nearest_value
            else:
                raise ValueError(f"Key '{key}' not found in the discrete grid.")
        return nearest_config
        
    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        Example fval:
        fvals = {
            'score': float,
            'generalization_score': float
        }
        '''
        # find nearest neighbor
        nearest_config = self._find_nearest_neighbor(candidate_config)
        # evaluate nearest neighbor
        fvals = self.hpo_bench.complete_call(nearest_config)
        return nearest_config, fvals


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=0)
    parser.add_argument('--engine', type=str) # temporary fix to run multiple in parallel
    parser.add_argument('--sm_mode', type=str)
    parser.add_argument('--use_input_warping', action='store_true')

    args = parser.parse_args()
    dataset_name = args.dataset
    dataset = DATASET_MAP[dataset_name][0]
    model = args.model
    seed = args.seed
    num_seeds = args.num_seeds
    chat_engine = args.engine
    sm_mode = args.sm_mode
    use_input_warping = args.use_input_warping

    if num_seeds == 0:
        seeds_to_run = [seed]
    else:
        seeds_to_run = list(range(seed, num_seeds+seed))



    # Describe task context
    task_context = {}
    task_context['model'] = MODEL_MAP[model]
    task_context['task'] = 'classification' # hpo_bech datasets are all classification

    task = openml.tasks.get_task(DATASET_MAP[dataset_name][1])
    dataset_ = task.get_dataset()
    X, y, categorical_mask, _ = dataset_.get_data(target=dataset_.default_target_attribute)

    task_context['tot_feats'] = X.shape[1]
    task_context['cat_feats'] = len(categorical_mask)
    task_context['num_feats'] = X.shape[1] - len(categorical_mask)
    task_context['n_classes'] = len(np.unique(y))
    task_context['metric'] = "accuracy"
    task_context['lower_is_better'] = False
    task_context['num_samples'] = X.shape[0]
    with open('hp_configurations/hpobench.json', 'r') as f:
        task_context['hyperparameter_constraints'] = json.load(f)[model]

    # define result save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_dir = f'{script_dir}/results_{sm_mode}/{dataset_name}/{model}'
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    # define logging directory
    logging_fpath = f'{script_dir}/logs_{sm_mode}/{dataset_name}/{model}.log'
    if not os.path.exists(os.path.dirname(logging_fpath)):
        os.makedirs(os.path.dirname(logging_fpath))
    setup_logging(logging_fpath)

    tot_llm_cost = 0

    for seed in seeds_to_run:
        logger.info('='*200)
        logger.info(f'Executing LLAMBO ({sm_mode}) to tune {model} on {dataset_name} with seed {seed+1} / {num_seeds}...')
        logger.info(f'Task context: {task_context}')

        # instantiate benchmark
        benchmark = HPOExpRunner(model, dataset, seed)

        llambo = LLAMBO(task_context, sm_mode, n_candidates=10, n_templates=2, n_gens=10, 
                        alpha=0.1, n_initial_samples=5, n_trials=25, init_f=benchmark.generate_initialization,
                        bbox_eval_f=benchmark.evaluate_point, chat_engine=chat_engine, use_input_warping=use_input_warping)
        llambo.seed = seed


        configs, fvals = llambo.optimize()
        logger.info(f'[LLAMBO] Query cost: {sum(llambo.llm_query_cost):.4f}')
        logger.info(f'[LLAMBO] Query time: {sum(llambo.llm_query_time):.4f}')
        tot_llm_cost += sum(llambo.llm_query_cost)

        # save search history
        search_history = pd.concat([configs, fvals], axis=1)
        search_history.to_csv(f'{save_res_dir}/{seed}.csv', index=False)

        
        logger.info(search_history)
        logger.info(f'[LLAMBO] RUN COMPLETE, saved results to {save_res_dir}...')

        # save search info
        search_info = {
            'llm_query_cost_breakdown': llambo.llm_query_cost,
            'llm_query_time_breakdown': llambo.llm_query_time,
            'llm_query_cost': sum(llambo.llm_query_cost),
            'llm_query_time': sum(llambo.llm_query_time),
        }
        with open(f'{save_res_dir}/{seed}_search_info.json', 'w') as f:
            json.dump(search_info, f)

    logger.info('='*200)
    logger.info(f'[LLAMBO] {seed+1} evaluation runs complete! Total cost: ${tot_llm_cost:.4f}')