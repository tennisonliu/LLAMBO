import argparse
from utils_templates import RandomTemplate, FullTemplate
from sampler import write_random_bayesmark

# Predefined lists containing datasets, models, and metrics to be used in the experiments
LIST_DATASETS = ['cutract', 'maggic', 'seer', 'breast', 'wine', 'digits']
#LIST_DATASETS = ['cutract']

LIST_MODELS   = ['RF', 'SVM', 'DT']
#LIST_MODELS   = ['RF']
LIST_METRICS  = ['acc']
NUM_SEEDS     = 10 # Number of seeds for randomization, ensuring reproducibility across runs
NUM_INIT      = 5 # Number of initial points to start the optimization

if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = args.config

    if config == 'random':
        template_object = None
    else:
        if config in ['sobol', 'lhs']:
            template_object = RandomTemplate(config)
        else:
            template_object = FullTemplate(context = config, provide_ranges = True)
    
    write_random_bayesmark(LIST_MODELS, LIST_DATASETS, LIST_METRICS, n_init = NUM_INIT, num_seeds = NUM_SEEDS, template_object = template_object)
    import time
    time.sleep(30)