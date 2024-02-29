import numpy as np
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate

def _count_decimal_places(n):
    '''Count the number of decimal places in a number.'''
    s = format(n, '.10f')
    if '.' not in s:
        return 0
    num_dp = len(s.split('.')[1].rstrip('0')) 
    return num_dp

def prepare_configurations(
        hyperparameter_constraints, 
        observed_configs, 
        observed_fvals=None, 
        seed=None, 
        bootstrapping=False, 
        use_feature_semantics=True,
        shuffle_features=False,
        apply_warping=False
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

    # shuffle columns
    if shuffle_features:
        np.random.seed(0)
        shuffled_indices = np.random.permutation(len(hyperparameter_names))
        observed_configs = observed_configs[hyperparameter_names[shuffled_indices]]

    # bootstrap resampling
    if bootstrapping:
        observed_configs = observed_configs.sample(frac=1, replace=True, random_state=seed)
        if observed_fvals is not None:
            observed_fvals = observed_fvals.loc[observed_configs.index]

    # reset index
    observed_configs = observed_configs.reset_index(drop=True)
    if observed_fvals is not None:
        observed_fvals = observed_fvals.reset_index(drop=True)
    
    # serialize the k-shot examples
    for index, row in observed_configs.iterrows():
        row_string = ''
        for i in range(len(row)):
            hyp_type = hyperparameter_constraints[hyperparameter_names[i]][0]
            hyp_trans = hyperparameter_constraints[hyperparameter_names[i]][1]
            if hyp_type in ['int', 'float']:
                lower_bound = hyperparameter_constraints[hyperparameter_names[i]][2][0]
            else:
                lower_bound = hyperparameter_constraints[hyperparameter_names[i]][2][1]
            n_dp = _count_decimal_places(lower_bound) # number of decimal places
            if use_feature_semantics:
                row_string += f'{hyperparameter_names[i]} is ' 
            else:
                row_string += f'X{i+1} is '

            if apply_warping:
                if hyp_type == 'int' and hyp_trans != 'log':
                    row_string += str(int(row[i]))
                elif hyp_type == 'float' or hyp_trans == 'log':
                    row_string += f'{row[i]:.{n_dp}f}'
                elif hyp_type == 'ordinal':
                    row_string += f'{row[i]:.{n_dp}f}'
                else:
                    row_string += row[i]

            else:
                if hyp_type == 'int':
                    row_string += str(int(row[i]))
                elif hyp_type == 'float':
                    row_string += f'{row[i]:.{n_dp}f}'
                elif hyp_type == 'ordinal':
                    row_string += f'{row[i]:.{n_dp}f}'
                else:
                    row_string += row[i]


            if i != len(row)-1:
                row_string += ', '
        example = {'Q': row_string}
        if observed_fvals is not None:
            row_index = observed_fvals.index.get_loc(index)
            perf = f'## {observed_fvals.values[row_index][0]:.6f} ##'
            example['A'] = perf
        examples.append(example)
        
    return examples



def gen_prompt_tempates(
        task_context, 
        observed_configs, 
        observed_fvals, 
        candidate_configs, 
        n_prompts=1, 
        bootstrapping=False,
        use_context='full_context', 
        use_feature_semantics=True,
        shuffle_features=False,
        apply_warping=False
):
    '''Generate prompt templates for the few-shot learning task.'''

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

    if use_context == 'no_context' or not use_feature_semantics:
        metric = 'a metric'
    
    all_prompt_templates = []
    for i in range(n_prompts):
        few_shot_examples = prepare_configurations(task_context['hyperparameter_constraints'], observed_configs, observed_fvals, 
                                                              seed=i, bootstrapping=bootstrapping, use_feature_semantics=use_feature_semantics, 
                                                              shuffle_features=shuffle_features, apply_warping=apply_warping)

        example_template = """
Hyperparameter configuration: {Q}
Performance: {A}"""
        
        example_prompt = PromptTemplate(
            input_variables=["Q", "A"],
            template=example_template
        )

        prefix = ""
        prefix = f"The following are hyperparameter configurations for a {model} and the corresponding performance measured in {metric}."
        if use_context == 'full_context':
            if task == 'classification':
                prefix += f" The model is evaluated on a tabular {task} task and the label contains {n_classes} classes."
            elif task == 'regression':
                prefix += f" The model is evaluated on a tabular {task} task."
            else:
                raise Exception
            prefix += f" The tabular dataset contains {n_samples} samples and {tot_feats} features ({cat_feats} categorical, {num_feats} numerical). "
        prefix += f" Your response should only contain the predicted {metric} in the format ## performance ##."

        suffix = """
Hyperparameter configuration: {Q}
Performance: """

        few_shot_prompt = FewShotPromptTemplate(
            examples=few_shot_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["Q"],
            example_separator=""
        )
        all_prompt_templates.append(few_shot_prompt)

    query_examples = prepare_configurations(task_context['hyperparameter_constraints'], candidate_configs, 
                                                       seed=None, bootstrapping=False, use_feature_semantics=use_feature_semantics, 
                                                       shuffle_features=shuffle_features, apply_warping=apply_warping)
    return all_prompt_templates, query_examples


