import os
import time
import openai
import asyncio
import numpy as np
import pandas as pd
from aiohttp import ClientSession
from llambo.rate_limiter import RateLimiter
from llambo.generative_sm_utils import gen_prompt_tempates

openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]


class LLM_GEN_SM:
    def __init__(self, task_context, n_gens, lower_is_better, top_pct,
                 n_templates=1, rate_limiter=None, 
                 verbose=False, chat_engine=None):
        '''Initialize the forward LLM surrogate model. This is modelling p(y|x) as in GP/SMAC etc.'''
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.top_pct = top_pct
        self.n_templates = n_templates
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=240000, time_frame=60, max_requests=2900)
        else:
            self.rate_limiter = rate_limiter
        self.recalibrator = None
        self.chat_engine = chat_engine
        self.verbose = verbose

    async def _async_generate(self, few_shot_template, query_example, query_idx):
        '''Generate a response from the LLM async.'''
        prompt = few_shot_template.format(Q=query_example['Q'])

        MAX_RETRIES = 3

        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)

            resp = None
            n_preds = int(self.n_gens/self.n_templates)
            for retry in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    self.rate_limiter.add_request(request_text=prompt, current_time=start_time)
                    resp = await openai.Completion.acreate(
                        model="gpt-3.5-turbo-instruct",
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=8,
                        top_p=0.95,
                        n=max(n_preds, 3),            # e.g. for 5 templates, get 2 generations per template
                        request_timeout=10,
                        logprobs=5,
                    )
                    self.rate_limiter.add_request(request_token_count=resp['usage']['total_tokens'], current_time=time.time())
                    break
                except Exception as e:
                    print(f'[SM] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                    print(resp)
                    if retry == MAX_RETRIES-1:
                        await openai.aiosession.get().close()
                        raise e
                    pass

        await openai.aiosession.get().close()

        if resp is None:
            return None

        tot_tokens = resp['usage']['total_tokens']
        tot_cost = 0.0015*(resp['usage']['prompt_tokens']/1000) + 0.002*(resp['usage']['completion_tokens']/1000)

        return query_idx, resp, tot_cost, tot_tokens



    async def _generate_concurrently(self, few_shot_templates, query_examples):
        '''Perform concurrent generation of responses from the LLM async.'''

        coroutines = []
        for template in few_shot_templates:
            for query_idx, query_example in enumerate(query_examples):
                coroutines.append(self._async_generate(template, query_example, query_idx))

        tasks = [asyncio.create_task(c) for c in coroutines]

        results = [[] for _ in range(len(query_examples))]      # nested list

        llm_response = await asyncio.gather(*tasks)

        for response in llm_response:
            if response is not None:
                query_idx, resp, tot_cost, tot_tokens = response
                results[query_idx].append([resp, tot_cost, tot_tokens])

        return results  # format [(resp, tot_cost, tot_tokens), None, (resp, tot_cost, tot_tokens)]

    def process_response(self, all_raw_response):
        all_pred_probs = [] # p(s<\tau | h)
        for raw_response in all_raw_response:
            tokens = raw_response['tokens']
            logprobs = raw_response['top_logprobs']
            pred_index = min((tokens.index(val) for val in ["0", "1"] if val in tokens), default=None)
            if pred_index is None:
                all_pred_probs.append(np.nan)
            else:
                try:
                    prob_1 = logprobs[pred_index]["1"]
                    prob_0 = logprobs[pred_index]["0"]
                    prob_1 = np.exp(prob_1)/(np.exp(prob_1) + np.exp(prob_0))
                    all_pred_probs.append(prob_1)
                except:
                    all_pred_probs.append(np.nan)

        return all_pred_probs

    
    async def _predict(self, all_prompt_templates, query_examples):
        start = time.time()
        all_preds = []
        tot_tokens = 0
        tot_cost = 0

        bool_pred_returned = []

        # make predictions in chunks of 5, for each chunk make concurent calls
        for i in range(0, len(query_examples), 5):
            query_chunk = query_examples[i:i+5]
            chunk_results = await self._generate_concurrently(all_prompt_templates, query_chunk)
            bool_pred_returned.extend([1 if x is not None else 0 for x in chunk_results])                # track effective number of predictions returned

            for _, sample_response in enumerate(chunk_results):
                if not sample_response:     # if sample prediction is an empty list :(
                    sample_preds = [np.nan] * self.n_gens
                else:
                    all_raw_response = [x['logprobs'] for template_response in sample_response for x in template_response[0]['choices'] ]        # fuarr this is some high level programming
                    sample_preds = self.process_response(all_raw_response)
                    tot_cost += sum([x[1] for x in sample_response])
                    tot_tokens += sum([x[2] for x in sample_response])
                all_preds.append(sample_preds)
        
        end = time.time()
        time_taken = end - start

        success_rate = sum(bool_pred_returned)/len(bool_pred_returned)

        pred_probs = np.array(all_preds).astype(float)
        mean_probs = np.nanmean(pred_probs, axis=1)

        return mean_probs, success_rate, tot_cost, tot_tokens, time_taken
    
    async def _evaluate_candidate_points(self, observed_configs, observed_fvals, candidate_configs):
        '''Evaluate candidate points using the LLM model.'''
        all_run_cost = 0
        all_run_time = 0

        all_prompt_templates, query_examples = gen_prompt_tempates(self.task_context, observed_configs, observed_fvals, candidate_configs, 
                                                                   self.lower_is_better, self.top_pct, n_prompts=self.n_templates)
        
        print('*'*100)
        print(f'Number of all_prompt_templates: {len(all_prompt_templates)}')
        print(f'Number of query_examples: {len(query_examples)}')
        print(all_prompt_templates[0].format(Q=query_examples[0]['Q']))
        # print(freeze)


        response = await self._predict(all_prompt_templates, query_examples)

        pred_probs, success_rate, tot_cost, tot_tokens, time_taken = response

        all_run_cost += tot_cost
        all_run_time += time_taken

        return pred_probs, all_run_cost, all_run_time


    def _warp_candidate_points(self, configurations):
        '''Warp candidate points to log scale if necessary.'''
        warped_configs = configurations.copy().to_dict(orient='records')
        hyperparameter_constraints = self.task_context['hyperparameter_constraints']
        for config in warped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == 'log':
                    config[hyperparameter] = np.log10(config[hyperparameter])

        warped_configs = pd.DataFrame(warped_configs)
        return warped_configs
    

    def _unwarp_candidate_points(self, configurations):
        '''Unwarp candidate points from log scale if necessary.'''
        unwarped_configs = configurations.copy().to_dict(orient='records')
        hyperparameter_constraints = self.task_context['hyperparameter_constraints']
        for config in unwarped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == 'log':
                    config[hyperparameter] = 10**config[hyperparameter]

        unwarped_configs = pd.DataFrame(unwarped_configs)
        return unwarped_configs
    

    def select_query_point(self, observed_configs, observed_fvals, candidate_configs, return_raw_preds=False):
        '''Select the next query point using expected improvement.'''
        # warp candidate points
        observed_configs = self._warp_candidate_points(observed_configs)
        candidate_configs = self._warp_candidate_points(candidate_configs)

        pred_probs, cost, time_taken = asyncio.run(self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs))

        best_point_index = np.argmax(pred_probs)

        # unwarp candidate points
        candidate_configs = self._unwarp_candidate_points(candidate_configs)

        best_point = candidate_configs.iloc[[best_point_index], :]  # return selected point as dataframe not series

        if return_raw_preds:
            return best_point, pred_probs, cost, time_taken
        else:
            return best_point, cost, time_taken



