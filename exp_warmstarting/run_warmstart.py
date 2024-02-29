# Import necessary modules from Python's standard library and custom modules.
import sys
sys.path.append('exp_baselines') # Add the 'exp_baselines' directory to the Python path to find custom modules.
from utils_templates import FullTemplate, RandomTemplate
from tasks import run_bayesmark

# Define lists of configurations for the experiments.
LIST_INIT     = ["random", "sobol", "lhs", "Partial_Context", "No_Context", "Full_Context"]
LIST_SM       = ["bo_tpe", "bo_skopt"]
LIST_DATASETS = [ 'breast', 'wine', 'digits', 'cutract', 'maggic', 'seer']

LIST_MODELS   = ['RF', 'SVM', 'DT']
LIST_METRICS  = ['acc']
NUM_SEEDS     = 10
NUM_INIT      = 5
NUM_RUNS      = 25

if __name__ == '__main__':
    # parse input arguments
    all_objects = []
    for sm_model in LIST_SM:
        for config in LIST_INIT:
            if config == 'random':
                template_object = RandomTemplate()
            else:
                if config in ['sobol', 'lhs']:
                    template_object = RandomTemplate(config)
                else:
                    template_object = FullTemplate(context = config, provide_ranges = True)
            all_objects += [{'template_object': template_object, 'name_experiment': f'{config}_{sm_model}', 'bo_type': sm_model}]

    run_bayesmark(LIST_MODELS, LIST_DATASETS, LIST_METRICS, all_dict_templates = all_objects,  n_repetitions = NUM_SEEDS,
                                                                                                 n_runs = NUM_RUNS, n_init = NUM_INIT)