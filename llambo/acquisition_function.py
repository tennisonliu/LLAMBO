import os
import random
import math
import time
import openai
import asyncio
import numpy as np
import pandas as pd
from aiohttp import ClientSession
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
from llambo.rate_limiter import RateLimiter

openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]


class LLM_ACQ:
    def __init__(self, task_context, n_candidates, n_templates, lower_is_better, 
                 jitter=False, rate_limiter=None, warping_transformer=None, chat_engine=None, 
                 prompt_setting=None, shuffle_features=False):
        '''Initialize the LLM Acquisition function.'''
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = int(n_candidates/n_templates)
        self.lower_is_better = lower_is_better
        self.apply_jitter = jitter
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=40000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter
        if warping_transformer is None:
            self.warping_transformer = None
            self.apply_warping = False
        else:
            self.warping_transformer = warping_transformer
            self.apply_warping = True
        self.chat_engine = chat_engine
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

        assert type(self.shuffle_features) == bool, 'shuffle_features must be a boolean'


    def _jitter(self, desired_fval):
        '''Add jitter to observed fvals to prevent duplicates.'''

        if not self.apply_jitter:
            return desired_fval

        assert hasattr(self, 'observed_best'), 'observed_best must be set before calling _jitter'
        assert hasattr(self, 'observed_worst'), 'observed_worst must be set before calling _jitter'
        assert hasattr(self, 'alpha'), 'alpha must be set before calling _jitter'

        jittered = np.random.uniform(low=min(desired_fval, self.observed_best), 
                                        high=max(desired_fval, self.observed_best), 
                                        size=1).item()

        return jittered


    def _count_decimal_places(self, n):
        '''Count the number of decimal places in a number.'''
        s = format(n, '.10f')
        if '.' not in s:
            return 0
        n_dp = len(s.split('.')[1].rstrip('0'))
        return n_dp

    def _prepare_configurations_acquisition(
        self,
        observed_configs=None, 
        observed_fvals=None, 
        seed=None,
        use_feature_semantics=True,
        shuffle_features=False
    ):
        '''Prepare and (possibly shuffle) few-shot examples for prompt templates.'''
        examples = []
        
        if seed is not None:
            # if seed is provided, shuffle the observed configurations
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(observed_configs.index)
            observed_configs = observed_configs.loc[shuffled_indices]
            if observed_fvals is not None:
                observed_fvals = observed_fvals.loc[shuffled_indices]
        else:
            # if no seed is provided, sort the observed configurations by fvals
            if type(observed_fvals) == pd.DataFrame:
                if self.lower_is_better:
                    observed_fvals = observed_fvals.sort_values(by=observed_fvals.columns[0], ascending=False)
                else:
                    observed_fvals = observed_fvals.sort_values(by=observed_fvals.columns[0], ascending=True)
                observed_configs = observed_configs.loc[observed_fvals.index]

        if shuffle_features:
            # shuffle the columns of observed configurations
            np.random.seed(0)
            shuffled_columns = np.random.permutation(observed_configs.columns)
            observed_configs = observed_configs[shuffled_columns]
            
        # serialize the k-shot examples
        if observed_configs is not None:
            hyperparameter_names = observed_configs.columns
            for index, row in observed_configs.iterrows():
                row_string = '## '
                for i in range(len(row)):
                    hyp_type = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][0]
                    hyp_transform = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][1]

                    if use_feature_semantics:
                        row_string += f'{hyperparameter_names[i]}: '
                    else:
                        row_string += f'X{i+1}: '

                    if hyp_type in ['int', 'float']:
                        lower_bound = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][2][0]
                    else:
                        lower_bound = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][2][1]
                    n_dp = self._count_decimal_places(lower_bound)
                    value = row[i]
                    if self.apply_warping:
                        if hyp_type == 'int' and hyp_transform != 'log':
                            row_string += str(int(value))
                        elif hyp_type == 'float' or hyp_transform == 'log':
                            row_string += f'{value:.{n_dp}f}'
                        elif hyp_type == 'ordinal':
                            row_string += f'{value:.{n_dp}f}'
                        else:
                            row_string += value

                    else:
                        if hyp_type == 'int':
                            row_string += str(int(value))
                        elif hyp_type in ['float', 'ordinal']:
                            row_string += f'{value:.{n_dp}f}'
                        else:
                            row_string += value

                    if i != len(row)-1:
                        row_string += ', '
                row_string += ' ##'
                example = {'Q': row_string}
                if observed_fvals is not None:
                    row_index = observed_fvals.index.get_loc(index)
                    perf = f'{observed_fvals.values[row_index][0]:.6f}'
                    example['A'] = perf
                examples.append(example)
        elif observed_fvals is not None:
            examples = [{'A': f'{observed_fvals:.6f}'}]
        else:
            raise Exception
            
        return examples
    

    def _gen_prompt_tempates_acquisitions(
        self,
        observed_configs, 
        observed_fvals, 
        desired_fval,
        n_prompts=1,
        use_context='full_context',
        use_feature_semantics=True,
        shuffle_features=False
    ):
        '''Generate prompt templates for acquisition function.'''
        all_prompt_templates = []
        all_query_templates = []

        for i in range(n_prompts):
            few_shot_examples = self._prepare_configurations_acquisition(observed_configs, observed_fvals,seed=i, use_feature_semantics=use_feature_semantics)           # need to update seed?
            jittered_desired_fval = self._jitter(desired_fval)

            # contextual information about the task
            task_context = self.task_context
            model = task_context['model']
            task = task_context['task']
            tot_feats = task_context['tot_feats']
            cat_feats = task_context['cat_feats']
            num_feats = task_context['num_feats']
            n_classes = task_context['n_classes']
            metric = 'mean squared error' if task_context['metric'] == 'neg_mean_squared_error' else task_context['metric']
            num_samples = task_context['num_samples']
            hyperparameter_constraints = task_context['hyperparameter_constraints']
            
            example_template = """
Performance: {A}
Hyperparameter configuration: {Q}"""
            
            example_prompt = PromptTemplate(
                input_variables=["Q", "A"],
                template=example_template
            )

            prefix = f"The following are examples of performance of a {model} measured in {metric} and the corresponding model hyperparameter configurations."
            if use_context == 'full_context':
                if task == 'classification':
                    prefix += f" The model is evaluated on a tabular {task} task containing {n_classes} classes."
                elif task == 'regression':
                    prefix += f" The model is evaluated on a tabular {task} task."
                else:
                    raise Exception
                prefix += f" The tabular dataset contains {num_samples} samples and {tot_feats} features ({cat_feats} categorical, {num_feats} numerical)."
            prefix += f" The allowable ranges for the hyperparameters are:\n"
            for i, (hyperparameter, constraint) in enumerate(hyperparameter_constraints.items()):
                if constraint[0] == 'float':
                    # number of decimal places!!
                    n_dp = self._count_decimal_places(constraint[2][0])
                    if constraint[1] == 'log' and self.apply_warping:
                        lower_bound = np.log10(constraint[2][0])
                        upper_bound = np.log10(constraint[2][1])
                    else:
                        lower_bound = constraint[2][0]
                        upper_bound = constraint[2][1]

                    if use_feature_semantics:
                        prefix += f"- {hyperparameter}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                    else:
                        prefix += f"- X{i+1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"

                    if constraint[1] == 'log' and self.apply_warping:
                        prefix += f" (log scale, precise to {n_dp} decimals)"
                    else:
                        prefix += f" (float, precise to {n_dp} decimals)"
                elif constraint[0] == 'int':
                    if constraint[1] == 'log' and self.apply_warping:
                        lower_bound = np.log10(constraint[2][0])
                        upper_bound = np.log10(constraint[2][1])
                        n_dp = self._count_decimal_places(lower_bound)
                    else:
                        lower_bound = constraint[2][0]
                        upper_bound = constraint[2][1]
                        n_dp = 0

                    if use_feature_semantics:
                        prefix += f"- {hyperparameter}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                    else:
                        prefix += f"- X{i+1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                    
                    if constraint[1] == 'log' and self.apply_warping:
                        prefix += f" (log scale, precise to {n_dp} decimals)"
                    else:
                        prefix += f" (int)"

                elif constraint[0] == 'ordinal':
                    if use_feature_semantics:
                        prefix += f"- {hyperparameter}: "
                    else:
                        prefix += f"- X{i+1}: "
                    prefix += f" (ordinal, must take value in {constraint[2]})"

                else:
                    raise Exception('Unknown hyperparameter value type') 

                prefix += "\n"
            prefix += f"Recommend a configuration that can achieve the target performance of {jittered_desired_fval:.6f}. "
            if use_context in ['partial_context', 'full_context']:
                prefix += "Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with highest possible precision, as requested by the allowed ranges. "
            prefix += f"Your response must only contain the predicted configuration, in the format ## configuration ##.\n"

            suffix = """
Performance: {A}
Hyperparameter configuration:"""

            few_shot_prompt = FewShotPromptTemplate(
                examples=few_shot_examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=["A"],
                example_separator=""
            )
            all_prompt_templates.append(few_shot_prompt)

            query_examples = self._prepare_configurations_acquisition(observed_fvals=jittered_desired_fval, seed=None, shuffle_features=shuffle_features)
            all_query_templates.append(query_examples)

        return all_prompt_templates, all_query_templates
    
    async def _async_generate(self, user_message):
        '''Generate a response from the LLM async.'''
        message = []
        message.append({"role": "system","content": "You are an AI assistant that helps people find information."})
        message.append({"role": "user", "content": user_message})

        MAX_RETRIES = 3

        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)

            resp = None
            for retry in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    self.rate_limiter.add_request(request_text=user_message, current_time=start_time)
                    resp = await openai.ChatCompletion.acreate(
                        engine=self.chat_engine,
                        messages=message,
                        temperature=0.8,
                        max_tokens=500,
                        top_p=0.95,
                        n=self.n_gens,
                        request_timeout=10
                    )
                    self.rate_limiter.add_request(request_token_count=resp['usage']['total_tokens'], current_time=start_time)
                    break
                except Exception as e:
                    print(f'[AF] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                    print(resp)
                    print(e)

        await openai.aiosession.get().close()

        if resp is None:
            return None

        tot_tokens = resp['usage']['total_tokens']
        tot_cost = 0.0015*(resp['usage']['prompt_tokens']/1000) + 0.002*(resp['usage']['completion_tokens']/1000)

        return resp, tot_cost, tot_tokens


    async def _async_generate_concurrently(self, prompt_templates, query_templates):
        '''Perform concurrent generation of responses from the LLM async.'''

        coroutines = []
        for (prompt_template, query_template) in zip(prompt_templates, query_templates):
            coroutines.append(self._async_generate(prompt_template.format(A=query_template[0]['A'])))

        # coroutines = [self._async_generate(prompt_template.format(A=query_example['A'])) for prompt_template in prompt_templates]
        tasks = [asyncio.create_task(c) for c in coroutines]

        # assert len(tasks) == int(self.n_candidates/self.n_gens)
        assert len(tasks) == int(self.n_templates)

        results = [None]*len(coroutines)

        llm_response = await asyncio.gather(*tasks)

        for idx, response in enumerate(llm_response):
            if response is not None:
                resp, tot_cost, tot_tokens = response
                results[idx] = (resp, tot_cost, tot_tokens)

        return results  # format [(resp, tot_cost, tot_tokens), None, (resp, tot_cost, tot_tokens)]
    
    def _convert_to_json(self, response_str):
        '''Parse LLM response string into JSON.'''
        pairs = response_str.split(',')
        response_json = {}
        for pair in pairs:
            key, value = [x.strip() for x in pair.split(':')]
            response_json[key] = float(value)
            
        return response_json
    
    def _filter_candidate_points(self, observed_points, candidate_points, precision=8):
        '''Filter candidate points that already exist in observed points. Also remove duplicates.'''
        # drop points that already exist in observed points
        rounded_observed = [{key: round(value, precision) for key, value in d.items()} for d in observed_points]
        rounded_candidate = [{key: round(value, precision) for key, value in d.items()} for d in candidate_points]
        filtered_candidates = [x for i, x in enumerate(candidate_points) if rounded_candidate[i] not in rounded_observed]

        def is_within_range(value, allowed_range):
            """Check if a value is within an allowed range."""
            value_type, transform, search_range = allowed_range
            if value_type == 'int':
                [min_val, max_val] = search_range
                if transform == 'log' and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                    return min_val <= value <= max_val
                else:
                    return min_val <= value <= max_val and int(value) == value
            elif value_type == 'float':                         # THIS MIGHT NEED TO CHANGE, RIGHT NOW IT CAN"T SIT ON THE BOUNDARY
                [min_val, max_val] = search_range
                if transform == 'log' and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                return min_val <= value <= max_val
            elif value_type == 'ordinal':
                # check that value is in allowed range up to 2 decimal places
                return any(math.isclose(value, x, abs_tol=1e-2) for x in allowed_range[2])
            else:
                raise Exception('Unknown hyperparameter value type')

        def is_dict_within_ranges(d, ranges_dict):
            """Check if all values in a dictionary are within their respective allowable ranges."""
            return all(key in ranges_dict and is_within_range(value, ranges_dict[key]) for key, value in d.items())

        def filter_dicts_by_ranges(dict_list, ranges_dict):
            """Return only those dictionaries where all values are within their respective allowable ranges."""
            return [d for d in dict_list if is_dict_within_ranges(d, ranges_dict)]


        # check that constraints are satisfied
        hyperparameter_constraints = self.task_context['hyperparameter_constraints']
        filtered_candidates = filter_dicts_by_ranges(filtered_candidates, hyperparameter_constraints)

        filtered_candidates = pd.DataFrame(filtered_candidates)
        # drop duplicates
        filtered_candidates = filtered_candidates.drop_duplicates()
        # reset index
        filtered_candidates = filtered_candidates.reset_index(drop=True)
        return filtered_candidates

    
    def get_candidate_points(self, observed_configs, observed_fvals, 
                             use_feature_semantics=True, use_context='full_context', alpha=-0.2):
        '''Generate candidate points for acquisition function.'''
        assert alpha >= -1 and alpha <= 1, 'alpha must be between -1 and 1'
        if alpha == 0:
            alpha = -1e-3 # a little bit of randomness never hurt anyone
        self.alpha = alpha

        if self.prompt_setting is not None:
            use_context = self.prompt_setting
        # generate prompt templates
        start_time = time.time()

        # get desired f_val for candidate points
        range = np.abs(np.max(observed_fvals.values) - np.min(observed_fvals.values))

        if range == 0:
            # sometimes there is no variability in y :')
            range = 0.1*np.abs(np.max(observed_fvals.values))
        alpha_range = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, 1e-1]

        if self.lower_is_better:
            self.observed_best = np.min(observed_fvals.values)
            self.observed_worst = np.max(observed_fvals.values)
            desired_fval = self.observed_best - alpha*range

            while desired_fval <= .00001:  # score can't be negative
                # try first alpha in alpha_range that is lower than current alpha
                for alpha_ in alpha_range:
                    if alpha_ < alpha:
                        alpha = alpha_  # new alpha
                        desired_fval = self.observed_best - alpha*range
                        break
            print(f'Adjusted alpha: {alpha} | [original alpha: {self.alpha}], desired fval: {desired_fval:.6f}')
        else:
            self.observed_best = np.max(observed_fvals.values)
            self.observed_worst = np.min(observed_fvals.values)
            desired_fval = self.observed_best + alpha*range

            while desired_fval >= .9999:  # accuracy can't be greater than 1
                for alpha_ in alpha_range:
                    if alpha_ < alpha:
                        alpha = alpha_  # new alpha
                        desired_fval = self.observed_best + alpha*range
                        break

            print(f'Adjusted alpha: {alpha} | [original alpha: {self.alpha}], desired fval: {desired_fval:.6f}')

        self.desired_fval = desired_fval

        if self.warping_transformer is not None:
            observed_configs = self.warping_transformer.warp(observed_configs)

        prompt_templates, query_templates = self._gen_prompt_tempates_acquisitions(observed_configs, observed_fvals, desired_fval, n_prompts=self.n_templates, use_context=use_context, use_feature_semantics=use_feature_semantics, shuffle_features=self.shuffle_features)

        print('='*100)
        print('EXAMPLE ACQUISITION PROMPT')
        print(f'Length of prompt templates: {len(prompt_templates)}')
        print(f'Length of query templates: {len(query_templates)}')
        print(prompt_templates[0].format(A=query_templates[0][0]['A']))
        print('='*100)

        number_candidate_points = 0
        filtered_candidate_points = pd.DataFrame()

        retry = 0
        while number_candidate_points < 5:
            llm_responses = asyncio.run(self._async_generate_concurrently(prompt_templates, query_templates))

            candidate_points = []
            tot_cost = 0
            tot_tokens = 0
            # loop through n_coroutine async calls
            for response in llm_responses:
                if response is None:
                    continue
                # loop through n_gen responses
                for response_message in response[0]['choices']:
                        response_content = response_message['message']['content']
                        try:
                            response_content = response_content.split('##')[1].strip()
                            candidate_points.append(self._convert_to_json(response_content))
                        except:
                            print(response_content)
                            continue
                tot_cost += response[1]
                tot_tokens += response[2]

            proposed_points = self._filter_candidate_points(observed_configs.to_dict(orient='records'), candidate_points)
            filtered_candidate_points = pd.concat([filtered_candidate_points, proposed_points], ignore_index=True)
            number_candidate_points = filtered_candidate_points.shape[0]

            print(f'Attempt: {retry}, number of proposed candidate points: {len(candidate_points)}, ',
                  f'number of accepted candidate points: {filtered_candidate_points.shape[0]}')


            retry += 1
            if retry > 3:
                print(f'Desired fval: {desired_fval:.6f}')
                print(f'Number of proposed candidate points: {len(candidate_points)}')
                print(f'Number of accepted candidate points: {filtered_candidate_points.shape[0]}')
                if len(candidate_points) > 5:
                    filtered_candidate_points = pd.DataFrame(candidate_points)
                    break
                else:
                    raise Exception('LLM failed to generate candidate points')

        if self.warping_transformer is not None:
            filtered_candidate_points = self.warping_transformer.unwarp(filtered_candidate_points)


        end_time = time.time()
        time_taken = end_time - start_time

        return filtered_candidate_points, tot_cost, time_taken