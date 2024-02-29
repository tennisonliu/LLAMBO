import numpy as np
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate

def _count_decimal_places(n):
    '''Count the number of decimal places in a number.'''
    s = format(n, '.10f')
    if '.' not in s:
        return 0
    num_dp = len(s.split('.')[1].rstrip('0')) 
    if num_dp == 0:
        return 2
    else:
        return num_dp

def prepare_configurations(
    hyperparameter_constraints,
    lower_is_better, 
    top_pct, 
    observed_configs, 
    observed_fvals=None, 
    seed=None
):
    '''Prepare and possible (shuffle) the configurations for prompt templates.'''
    examples = []
    
    hyperparameter_names = observed_configs.columns
    observed_configs_ = observed_configs.copy()
    observed_configs = observed_configs_
    
    # shuffle indices to reduce permutation sensitivity
    if seed is not None:
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(observed_configs.index)
        observed_configs = observed_configs.loc[shuffled_indices]
        if observed_fvals is not None:
            observed_fvals = observed_fvals.loc[shuffled_indices]

    # reset index
    observed_configs = observed_configs.reset_index(drop=True)
    if observed_fvals is not None:
        observed_fvals = observed_fvals.reset_index(drop=True)
    
    if observed_fvals is not None:
        if lower_is_better:
            labels = (observed_fvals < np.percentile(observed_fvals, int(top_pct*100))).astype(int)
        else:
            labels = (observed_fvals > np.percentile(observed_fvals, int(100 - top_pct*100))).astype(int)
        
    # serialize the k-shot examples
    for index, row in observed_configs.iterrows():
        row_string = ''
        for i in range(len(row)):
            lower_bound = hyperparameter_constraints[hyperparameter_names[i]][2]
            n_dp = _count_decimal_places(lower_bound) + 2 # number of decimal places
            row_string += f'{hyperparameter_names[i]}: ' + f'{row[i]:.{n_dp}f}' \
                    if isinstance(row[i], float) and not row[i]%1 ==0 else f'{hyperparameter_names[i]}: ' + str(row[i])
            if i != len(row)-1:
                row_string += ', '
        example = {'Q': row_string}
        if observed_fvals is not None:
            row_index = observed_fvals.index.get_loc(index)
            label = f'## {labels.values[row_index][0]} ##'
            example['A'] = label
        examples.append(example)
        
    return examples


def gen_prompt_tempates(
        task_context, 
        observed_configs, 
        observed_fvals, 
        candidate_configs, 
        lower_is_better, 
        top_pct, 
        n_prompts=1
):
    # contextual information about the task
    model = task_context['model']
    task = task_context['task']
    tot_feats = task_context['tot_feats']
    cat_feats = task_context['cat_feats']
    num_feats = task_context['num_feats']
    n_classes = task_context['n_classes']
    n_samples = task_context['num_samples']
    metric = task_context['metric']

    if metric == 'neg_mean_squared_error':
        metric = 'mean squared error'
    
    '''Generate prompt templates for the few-shot learning task.'''
    all_prompt_templates = []
    for i in range(n_prompts):
        few_shot_examples = prepare_configurations(task_context['hyperparameter_constraints'], lower_is_better, top_pct,
                                                              observed_configs, observed_fvals, seed=i)
        
        example_template = """
Hyperparameter configuration: {Q}
Classification: {A}"""
        
        example_prompt = PromptTemplate(
            input_variables=["Q", "A"],
            template=example_template
        )

        prefix = f"The following are examples of hyperparameter configurations for a {model} and the corresponding performance classification."
        if task == 'classification':
            prefix += f" The model is evaluated on a tabular {task} task and the label contains {n_classes} classes."
        elif task == 'regression':
            prefix += f" The model is evaluated on a tabular {task} task."
        else:
            raise Exception
        prefix += f" The tabular dataset contains {n_samples} samples and {tot_feats} features ({cat_feats} categorical, {num_feats} numerical). "
        prefix += f" The performance classification is 1 if the configuarion is in the best-performing {top_pct*100}% of all configurations and 0 otherwise. "
        prefix += f" Your response should only contain the predicted performance classification in the format ## performance classification ##."

        suffix = """
Hyperparameter configuration: {Q}
Classification: """

        few_shot_prompt = FewShotPromptTemplate(
            examples=few_shot_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["Q"],
            example_separator=""
        )
        all_prompt_templates.append(few_shot_prompt)

    query_examples = prepare_configurations(task_context['hyperparameter_constraints'], 
                                                    lower_is_better, top_pct, candidate_configs, seed=None)
    return all_prompt_templates, query_examples