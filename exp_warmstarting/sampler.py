
from typing import Any, Dict, List, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import openai
import os
import sys
import json
sys.path.append('exp_baselines')  # Adding a directory to the Python path for importing modules

import numpy as np

# Type aliases for better readability
NumericType = Union[float, int]

# Mapping of hyperparameter class names to their corresponding Python data types
config2type = {
    "UniformFloatHyperparameter": float,
    "UniformIntegerHyperparameter": int,
    "OrdinalHyperparameter": float,
}

# Function to check if a value falls within the specified range of a hyperparameter
def check_value_range(hp_name: str, config: CSH.Hyperparameter, val: NumericType) -> None:
    if val < config.lower or val > config.upper:
        raise ValueError(f"The sampled value for {hp_name} must be [{config.lower},{config.upper}], but got {val}.")

def save_json(path_json, data):
    # Convert numpy.int64 values to int for JSON serialization
    for item in data:
        for key, value in item.items():
            if isinstance(value, np.int64):
                item[key] = int(value)
    # Write the JSON data to the file
    with open(path_json, "w") as json_file:
        json.dump(data, json_file, indent=4) 

# Reverts evaluation configurations back to their original format
def revert_eval_config(
    eval_config: Dict[str, NumericType],
    config_space: CS.ConfigurationSpace,
    is_categoricals: Dict[str, bool],
    is_ordinals: Dict[str, bool],
    hp_names: List[str],
) -> Dict[str, Any]:

    converted_eval_config: Dict[str, Any] = {}
    for hp_name in hp_names:
        # Determine if the hyperparameter is categorical or ordinal
        is_categorical, is_ordinal = is_categoricals[hp_name], is_ordinals[hp_name]
        config = config_space.get_hyperparameter(hp_name)
        val = eval_config[hp_name]

        # Convert the configuration based on its type
        if is_categorical:
            # For categorical, select the choice directly
            converted_eval_config[hp_name] = config.choices[val]
        elif is_ordinal:
            # For ordinal, find the closest value in the sequence
            if config.meta is None:
                raise ValueError("The meta information of the ordinal hyperparameter must be provided")

            log  = config.meta.get("log", False)
            vals = np.log(config.sequence) if log else np.array(config.sequence)
            diff = np.abs(vals - val)
            converted_eval_config[hp_name] = config.sequence[diff.argmin()]
        else:
            dtype = config2type[config.__class__.__name__]
            q = config.q
            if config.log:
                val = np.exp(val)
            if q is not None or dtype is int:
                lb = config.lower
                q = 1 if q is None and dtype is int else q
                val = np.round((val - lb) / q) * q + lb

            check_value_range(hp_name=hp_name, config=config, val=val)

            converted_eval_config[hp_name] = dtype(val)

    return converted_eval_config

# Samples a random value for a given hyperparameter
def get_random_sample(
    hp_name: str,
    is_categorical: bool,
    is_ordinal: bool,
    rng: np.random.RandomState,
    config_space: CS.ConfigurationSpace,
) -> NumericType:

    config = config_space.get_hyperparameter(hp_name)

    if is_categorical:
        choices = config.choices
        sample = rng.randint(len(choices))
    elif is_ordinal:
        if config.meta is None:
            raise ValueError("The meta information of the ordinal hyperparameter must be provided")

        log = config.meta.get("log", False)
        seq = config.sequence
        sample = seq[rng.randint(len(seq))]
        sample = np.log(sample) if log else sample
    else:
        lb = np.log(config.lower) if config.log else config.lower
        ub = np.log(config.upper) if config.log else config.upper
        sample = rng.uniform() * (ub - lb) + lb

    return sample

# Samples initial configurations for a set of hyperparameters
def sample_initial_configuration(config_space, hp_names, is_categoricals, is_ordinals, rng):
    eval_config = {}
    for hp_name in hp_names:
        # Sample a configuration for each hyperparameter
        eval_config[hp_name] = get_random_sample(hp_name = hp_name,
                                                 rng = rng,
                                                 config_space = config_space,
                                                 is_categorical = is_categoricals[hp_name],
                                                 is_ordinal = is_ordinals[hp_name])    
    # Convert the sampled configuration back to its original format
    return revert_eval_config(eval_config = eval_config,
                              config_space = config_space,
                              is_categoricals = is_categoricals,
                              is_ordinals = is_ordinals,
                              hp_names = hp_names)

# Defines the configuration space based on specified options and types
def get_config_space(kw_opt, kw_type, hyp_log = None):
    import ConfigSpace as CS
    CONFIG_SPACE = CS.ConfigurationSpace()
    for key in kw_type.keys():
        # Determine the type of hyperparameter and add it to the space
        is_log = hyp_log[key] if hyp_log is not None else False
        if kw_type[key] == 'float':
            CONFIG_SPACE.add_hyperparameters([CS.UniformFloatHyperparameter(name=key,
                                                lower = kw_opt[key][0], upper = kw_opt[key][1], log = is_log)])
        elif kw_type[key] == 'int':
            CONFIG_SPACE.add_hyperparameters([CS.UniformIntegerHyperparameter(name=key,
                                                lower = kw_opt[key][0], upper = kw_opt[key][1], log = is_log)])
        elif type(kw_type[key]) == list:
            if type(kw_type[key][0]) == bool:
                CONFIG_SPACE.add_hyperparameters([CS.CategoricalHyperparameter(key, [0,1])])
            else:
                CONFIG_SPACE.add_hyperparameters([CS.CategoricalHyperparameter(key, kw_type[key])])
    return CONFIG_SPACE

# Samples a number of configurations from the configuration space
def sample_configurations(config_space, seed, num_samples):
    hp_names = list(config_space._hyperparameters.keys())
    rng = np.random.RandomState(seed)
    is_categoricals = {
        hp_name: config_space.get_hyperparameter(hp_name).__class__.__name__ == "CategoricalHyperparameter"
        for hp_name in hp_names}
    is_ordinals = {
        hp_name: config_space.get_hyperparameter(hp_name).__class__.__name__ == "OrdinalHyperparameter"
        for hp_name in hp_names}
    list_config = []
    for i in range(num_samples):
        list_config += [sample_initial_configuration(config_space, hp_names, is_categoricals, is_ordinals, rng)]
    return list_config

################################################## Skopt sampler #########################################################
from skopt import gp_minimize
import time  # Import the time module for timing executions

# Function to run the minimization process
def run(minimizer, bounds, num_samples, random_state, initial_point_generator = 'random', n_repeats=1):
    # Dummy function to optimize, always returns 0 (placeholder for real objective function)
    def func(x):
        return 0
    # Perform the minimization, repeating the process n_repeats times
    return [minimizer(func, bounds, n_initial_points = num_samples,
                      initial_point_generator = initial_point_generator,
                      n_calls = num_samples, random_state = random_state)
            for n in range(n_repeats)]

# Function to measure the execution time of the minimization process
def run_measure(initial_point_generator, bounds, random_state = 0, num_samples =10):
    start = time.time()  # Record the start time
    n_repeats = 1  # Number of repeats for the minimization process
    # Execute the run function and store results
    res = run(gp_minimize, bounds, num_samples, random_state,
              initial_point_generator = initial_point_generator, n_repeats=n_repeats)
    return res  # Return the results of the minimization process

# Function to convert configuration space into skopt space
def obtain_space(config_space, order_list):
    from skopt.space import Real, Categorical, Integer  # Import space definitions from skopt
    import ConfigSpace as CS  # Import ConfigSpace for configuration space definition
    new_space = []  # Initialize the list for the new space
    # Loop through each key in order_list to convert into skopt space
    for key in order_list:
        hp = config_space.get_hyperparameter(key)  # Get hyperparameter from config space
        # Convert different types of hyperparameters into skopt space definitions
        if isinstance(hp, CS.UniformFloatHyperparameter):
            prior = 'log-uniform' if hp.log else 'uniform'
            new_space += [Real(hp.lower, hp.upper, prior = prior, name = hp.name, transform = "identity")]
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            prior = 'log-uniform' if hp.log else 'uniform'
            new_space += [Integer(hp.lower, hp.upper, prior = prior, name = hp.name, transform = "identity")]
        else:
            new_space += [Categorical(hp.choices, name = hp.name, transform = "identity")]
    return new_space # Return the converted skopt space

# Function to transform skopt optimization results into a list of dictionaries
def transform_skotp_to_list(obj, order_list):
    new_list_config = [] # Initialize list for new configurations
    # Loop through each iteration result in obj
    for elem in obj[0].x_iters:
        new_dict = {}  # Initialize dictionary for current configuration
        # Populate dictionary with the results, mapping them to the correct keys
        for idx, key in enumerate(order_list):
            new_dict[key] = elem[idx]
        new_list_config += [new_dict]  # Add the dictionary to the list of configurations
    return new_list_config  # Return the list of configurations

# Function to obtain initial configurations as a dictionary
def obtain_dict_init(name_init, config_space, order_list, random_state = 0, num_samples = 10):
    new_space = obtain_space(config_space, order_list) # Convert config space to skopt space
    # Run the minimization measure to obtain results
    obj       = run_measure(name_init, new_space, random_state = random_state, num_samples = num_samples)
    # Transform skopt optimization results into a list of dictionaries
    return transform_skotp_to_list(obj, order_list)

############################################################################

def chat_gpt(input_text):
    # Set up OpenAI API parameters from environment variables
    openai.api_type    = os.environ["OPENAI_API_TYPE"]
    openai.api_version = os.environ["OPENAI_API_VERSION"]
    openai.api_base    = os.environ["OPENAI_API_BASE"]
    openai.api_key     = os.environ["OPENAI_API_KEY"]
    ENGINE             = os.environ["OPENAI_API_ENGINE"]

    # Initialize the message payload with system and user roles
    message = []
    message.append({"role":"system","content":"You are an AI assistant that helps people find information."})
    message.append({"role":"user", "content":input_text})
    # Request a chat completion from the OpenAI API with specified parameters
    resp = openai.ChatCompletion.create(
        engine = ENGINE,
        messages = message,
        temperature = 0.7,
        top_p = 0.95,
        n = 30,
        request_timeout = 100
    )
    return resp

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# Function to check if a dictionary is valid according to a given configuration space
def is_dict_valid_in_config_space(d, config_space):
    try:
        # Attempt to create a Configuration object with the given dictionary and config space
        config = CS.Configuration(config_space, values=d)
        return True
    except:
        # Return False if the dictionary is not valid
        return False    
# Function to check if all dictionaries in a list are valid in the given configuration space
def check_all_list(parsed_dicts, config_space):    
    for idx, d in enumerate(parsed_dicts):
        if not is_dict_valid_in_config_space(d, config_space):
            return False
    return True
# Function to parse and validate all list of dictionaries from GPT response
def obtain_all_list_valid(resp, config_space):
    all_resp = []
    import re
    import ast
    for i in range(len(resp["choices"])):
        try:
            text = resp["choices"][i]["message"]["content"]
            print("##################################")
            print(text)
            print("##################################")
            re.findall(r'\{[^}]+\}', text)
            dict_list = re.findall(r'\d+\.\s+({.*?})', text, re.DOTALL)
            parsed_dicts = [ast.literal_eval(d) for d in dict_list]
            if check_all_list(parsed_dicts, config_space):
                all_resp += [parsed_dicts]
        except:
            print("fail")
    return all_resp

# Function to generate initial configurations using GPT and a given template object
def obtain_gpt_init(template_object, config_space, n_init, task_dict):
    input_text = template_object.add_context(config_space, num_recommendation = n_init, task_dict = task_dict)
    resp       = chat_gpt(input_text)
    return obtain_all_list_valid(resp, config_space)

#################### Writting random configurations #########################

# Function to write all configurations to files in a specified path
def write_all_config_random(list_config, path_name, template_object):
    path_results = template_object.obtain_path(path_name)
    for idx in range(len(list_config)):
        this_path = '%s/config%s.json' % (path_results, idx)
        #if not os.path.exists(this_path):
        save_json(this_path, list_config[idx])
        print("this_path: %s, PASS: %s" % (this_path, idx))

#random_init = ["random", "sobol", "halton", "hammersly", "lhs", "grid"]
# Function to generate and write random configurations for a Bayesian optimization benchmark
def write_random_bayesmark(list_model, list_data, list_metric, n_init, num_seeds, template_object = None):
        from bayesmark.sklearn_funcs import SklearnModelCustom
        from utils_templates import RandomTemplate, FullTemplate
        template_object = RandomTemplate() if template_object is None else template_object
        all_dict = {}
        for this_model in list_model:
            idx = 0
            for this_metric in list_metric:
                for this_loader in list_data:
                    print('metric', this_metric, 'dataloader', this_loader, 'this_model', this_model)
                    smc_object                   =  SklearnModelCustom(this_model, this_loader, this_metric)
                    path_name                    =  smc_object.path_name
                    config_space, order_list     =  smc_object.get_config_space()

                    if isinstance(template_object, FullTemplate):
                        task_dict = smc_object.get_task_dict()
                        list_config = obtain_gpt_init(template_object, config_space, n_init, task_dict)
                    elif idx == 0:
                        if template_object is None:
                            list_config = [sample_configurations(config_space, i, n_init) for i in range(num_seeds)]

                        else:
                            list_config = [obtain_dict_init(template_object.name, config_space, order_list, 
                                                random_state = i, num_samples = n_init) for i in range(num_seeds)]
                        all_dict[this_model] = list_config
                    idx += 1
                    write_all_config_random(list_config, path_name, template_object)
        return all_dict
                 
# Function to generate and write random configurations for tabular benchmarks
def write_random_tabular(list_model, list_data, n_init, num_seeds, random_init = None):
        from tabular_benchmarks.tabular_benchmarks import HPOBench
        from utils_templates import RandomTemplate
        template_object = RandomTemplate() if random_init is None else RandomTemplate(add_path = '-%s' % random_init)
        for data_id in list_data:
            for this_model in list_model:
                this_bench        = HPOBench(model_name = this_model, dataset_id = data_id)
                config_space      = this_bench.config_space
                path_name         = this_bench.dataset_name
                order_list        = this_bench.order_list
                print('data model object')
                if random_init is None:
                    list_config = [sample_configurations(config_space, i, n_init) for i in range(num_seeds)]
                else:
                    list_config = [obtain_dict_init(random_init, config_space, order_list, 
                                        random_state = i, num_samples = n_init) for i in range(num_seeds)]

                write_all_config_random(list_config, path_name, template_object)                    
