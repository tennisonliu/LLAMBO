# Import necessary libraries
import sys
# Extend Python's search path to include the 'exp_warmstarting' directory, allowing the import of modules from it
sys.path.append('exp_warmstarting')
from utils_templates import RandomTemplate
import argparse

# Define lists of dataset names for classification, regression, and HPOBench benchmarking
LIST_DATASETS_CLF      = ['breast', 'wine', 'digits', 'iris', 'cutract', 'maggic', 'seer']
LIST_DATASETS_REG      = ['diabetes', 'Griewank', 'KTablet', 'Rosenbrock']
LIST_DATASETS_HPOBENCH = [0, 1, 2, 3, 4, 5, 6, 7]

# Define lists of metric names for accuracy and regression tasks
LIST_METRICS_ACC       = ['acc']
LIST_METRICS_REG       = ['mse']

# Define lists of model names for use in Bayesmark and HPOBench benchmarking
LIST_MODELS_BAYESMARK  = ['RF', 'SVM', 'DT', 'MLP-sgd','ada']
LIST_MODELS_HPOBENCH   = ['rf', 'xgb','nn']

# Initialize a list of Bayesian Optimization (BO) model configurations

ALL_BO_MODELS   = [{'template_object': RandomTemplate(), 'name_experiment': 'TPE (Random init)', 'bo_type': 'bo_tpe'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'GP (Random init)', 'bo_type': 'bo_gp'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'GP DKL (Random init)', 'bo_type': 'bo_dkl'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'SMAC (Random init)', 'bo_type': 'bo_smac'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'SKOPT (Random init)', 'bo_type': 'bo_skopt'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'Turbo (Random init)', 'bo_type': 'bo_turbo'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'Optuna (Random init)', 'bo_type': 'bo_optuna'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'Random (Random init)', 'bo_type': 'bo_random'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'HEBO (Random init)', 'bo_type': 'bo_hebo'}]
ALL_BO_MODELS  += [{'template_object': RandomTemplate(), 'name_experiment': 'DNGO (Random init)', 'bo_type': 'bo_dngo'}]

# Define constants for experiment configuration
NUM_SEEDS       = 1 # Number of seeds for experiment repetition
NUM_INIT        = 5 # Number of initial points for BO
NUM_RUNS        = 25
#NUM_RUNS        = 7 # Number of runs for the experiment

if __name__ == '__main__':
    # Initialize an empty list to store objects (not used in the provided snippet)
    all_objects = []
    # Setup command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str)
    args = parser.parse_args()
    run_config = args.run_config
    # Execute the appropriate benchmarking task based on the run configuration
    if run_config in ['bayesmark_clf', 'bayesmark_reg']:
        # Import the function to run Bayesmark experiments
        from tasks import run_bayesmark
        # Configure datasets and metrics based on whether it's a classification or regression task
        if run_config == 'bayesmark_clf':
            LIST_DATASETS = LIST_DATASETS_CLF
            LIST_METRICS  = LIST_METRICS_ACC
        elif run_config == 'bayesmark_reg':
            LIST_DATASETS = LIST_DATASETS_REG
            LIST_METRICS  = LIST_METRICS_REG
        # Run the Bayesmark experiment with the specified configurations
        run_bayesmark(LIST_MODELS_BAYESMARK, LIST_DATASETS, LIST_METRICS, all_dict_templates = ALL_BO_MODELS,  n_repetitions = NUM_SEEDS, n_runs = NUM_RUNS, n_init = NUM_INIT)

    elif run_config == 'hpobench':
        # Import the function to run HPOBench experiments
        from tasks import run_tabular
        # Run the HPOBench experiment with the specified configurations
        run_tabular(LIST_MODELS_HPOBENCH, LIST_DATASETS_HPOBENCH, all_dict_templates = ALL_BO_MODELS, n_repetitions = NUM_SEEDS, n_runs = NUM_RUNS, n_init = NUM_INIT)
