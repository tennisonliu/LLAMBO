import numpy as np

RANDOM_STATE = 0 # Global variable to set the seed for reproducibility

# Function to create a directory path for saving results
def obtain_path_to_write_results(task_name, context_type = None):
        import os
        path_results  = "results"
        # Create the main results directory if it doesn't exist        
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        # Create a subdirectory for the specific task
        path_task     = '%s/%s' % (path_results, task_name)
        if not os.path.exists(path_task):
            os.makedirs(path_task)
        # If a context type is provided, create an additional subdirectory
        if context_type is not None:
            path_context  = '%s/%s' % (path_task, context_type) 
            if not os.path.exists(path_context):
                os.makedirs(path_context)
            return path_context
        else:
            return path_task

# Function to save a dictionary as a pickle file
def save_pickle_dict(this_dict, path, name_file):
    import pickle
    with open('%s/%s.pickle' % (path, name_file), 'wb') as handle:
        pickle.dump(this_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Function to set the seed for various libraries
def seed_everything():
    seed = RANDOM_STATE
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to initialize and return a Bayesian optimization object based on the specified type
def obtain_bo_object(bo_type, fun_to_evaluate, config_space, order_list, seed):
    if bo_type == 'bo_gp':
        from bo_models.bo_gp_class import BO_optimization 
        BO_object = BO_optimization(fun_to_evaluate, config_space, order_list, seed = seed)
    elif bo_type == 'bo_dkl':
        from bo_models.bo_dkgp import BO_optimization_DKL
        BO_object = BO_optimization_DKL(fun_to_evaluate, config_space, order_list, seed = seed)
    elif bo_type == 'bo_random':
        from bo_models.bo_gp_class import BO_optimization_random
        BO_object = BO_optimization_random(fun_to_evaluate, config_space, order_list, seed = seed)
    elif bo_type == 'bo_gp_turbo':
        from bo_models.bo_gp_turbo import BO_optimization_Turbo
        BO_object = BO_optimization_Turbo(fun_to_evaluate, config_space, order_list, seed = seed)
    elif bo_type == 'bo_dngo':
        from bo_models.bo_dngo import BO_optimization_DNGO
        BO_object = BO_optimization_DNGO(fun_to_evaluate, config_space, order_list, seed = seed)
    elif bo_type == 'bo_sto':
        from bo_models.bo_sto import BO_optimization_sto
        BO_object = BO_optimization_sto(fun_to_evaluate, config_space, order_list, seed = seed)
    return BO_object

# Function to obtain a Bayesian optimization object for other specified types
def obtain_bo_other(bo_type):
    if bo_type   == 'bo_hebo':
        from bo_models.bo_hebo import bo_hebo as bo_used
    elif bo_type == 'bo_turbo':
        from bo_models.bo_turbo import bo_turbo as bo_used
    elif bo_type == 'bo_tpe':
        from bo_models.bo_tpe import bo_tpe as bo_used
    elif bo_type == 'bo_smac':
        from bo_models.bo_smac import bo_smac as bo_used
    elif bo_type == 'bo_skopt':
        from bo_models.bo_skopt import bo_skopt as bo_used
    elif bo_type == 'bo_optuna':
        from bo_models.bo_optuna import bo_optuna as bo_used
    return bo_used

# Main loop for running Bayesian optimization
def bo_loop(bo_type, n_repetitions, fun_to_evaluate, config_space, order_list, n_runs, n_init, list_init_config):
    all_final_y    = []
    all_metrics_pd = []
    for idx in range(n_repetitions):
        # Different handling based on the type of Bayesian optimization
        if bo_type in ['bo_hebo', 'bo_turbo', 'bo_tpe', 'bo_smac', 'bo_skopt', 'bo_optuna']:
            bo_used = obtain_bo_other(bo_type)
            final_y, all_metrics = bo_used(fun_to_evaluate, config_space, n_runs = n_runs, n_init = n_init,
                                            seed = idx, config_init = list_init_config[idx], order_list = order_list)
        else:
            BO_object            = obtain_bo_object(bo_type, fun_to_evaluate, config_space, order_list, idx)
            final_y, all_metrics = BO_object.optimize(n_runs, n_init, config_init = list_init_config[idx])

        print(final_y)
        print("generalization , ", all_metrics['generalization_score'].to_numpy())

        all_final_y    += [np.array(final_y)]
        all_metrics_pd += [all_metrics]
    return all_final_y, all_metrics_pd

# Function to run all models for a given evaluation function and set of templates
def run_all_models(fun_to_evaluate, all_dict_templates, n_repetitions, n_runs, n_init, config_space, order_list, path_name):
        for dict_template in all_dict_templates:
                name_exp              = dict_template['name_experiment']
                template_obj          = dict_template['template_object']

                path_spec_results     = obtain_path_to_write_results(path_name, name_exp)
                path_spec_configs     = template_obj.obtain_path(path_name)
                list_config           = template_obj.read_all_config(path_spec_configs, num_configs = n_repetitions)
                _, all_metrics_pd     = bo_loop(dict_template['bo_type'],
                                                        n_repetitions,
                                                        fun_to_evaluate,
                                                        config_space,
                                                        order_list,
                                                        n_runs,
                                                        n_init,
                                                        list_config)
                save_pickle_dict(all_metrics_pd, path_spec_results, 'metrics')

# Function to run the Bayesian optimization process for models evaluated with Bayesmark
def run_bayesmark(list_model, list_data, list_metric, all_dict_templates, n_repetitions, n_runs, n_init):
        from bayesmark.sklearn_funcs import SklearnModelCustom
        for this_metric in list_metric:
            for this_loader in list_data:
                for this_model in list_model:
                    print('metric', this_metric, 'dataloader', this_loader, 'this_model', this_model)
                    smc_object                   =  SklearnModelCustom(this_model, this_loader, this_metric)
                    path_name                    =  smc_object.path_name
                    config_space, order_list     =  smc_object.get_config_space()    
                    fun_to_evaluate              =  smc_object.obtain_evaluate(smc_object.evaluate)
                    run_all_models(fun_to_evaluate, all_dict_templates, n_repetitions, n_runs, n_init, config_space, order_list, path_name)

# Function to run the Bayesian optimization process for models evaluated with Bayesmark
def run_tabular(list_model, list_data, all_dict_templates, n_repetitions, n_runs, n_init):
        from tabular_benchmarks.tabular_benchmarks import HPOBench
        for data_id in list_data:
            for this_model in list_model:
                print("this_model: ", this_model)
                this_bench        = HPOBench(model_name = this_model, dataset_id = data_id)
                config_space      = this_bench.config_space
                path_name         = this_bench.dataset_name
                order_list        = this_bench.order_list
                fun_to_evaluate   = this_bench
                run_all_models(fun_to_evaluate, all_dict_templates, n_repetitions, n_runs, n_init, config_space, order_list, path_name)