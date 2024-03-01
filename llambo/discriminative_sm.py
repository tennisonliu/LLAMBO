import os
import time
import openai
import asyncio
import re
import numpy as np
from scipy.stats import norm
from aiohttp import ClientSession
from llambo.rate_limiter import RateLimiter
from llambo.discriminative_sm_utils import gen_prompt_tempates


openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]


class LLM_DIS_SM:
    def __init__(self, task_context, n_gens, lower_is_better, 
                 bootstrapping=False, n_templates=1, 
                 use_recalibration=False,
                 rate_limiter=None, warping_transformer=None,
                 verbose=False, chat_engine=None, 
                 prompt_setting=None, shuffle_features=False):
        '''Initialize the forward LLM surrogate model. This is modelling p(y|x) as in GP/SMAC etc.'''
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.bootstrapping = bootstrapping
        self.n_templates = n_templates
        assert not (bootstrapping and use_recalibration), 'Cannot do recalibration and boostrapping at the same time' 
        self.use_recalibration = use_recalibration
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=100000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter
        if warping_transformer is not None:
            self.warping_transformer = warping_transformer
            self.apply_warping = True
        else:
            self.warping_transformer = None
            self.apply_warping = False
        self.recalibrator = None
        self.chat_engine = chat_engine
        self.verbose = verbose
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

        assert type(self.shuffle_features) == bool, 'shuffle_features must be a boolean'


    async def _async_generate(self, few_shot_template, query_example, query_idx):
        '''Generate a response from the LLM async.'''
        message = []
        message.append({"role": "system","content": "You are an AI assistant that helps people find information."})
        user_message = few_shot_template.format(Q=query_example['Q'])
        message.append({"role": "user", "content": user_message})

        MAX_RETRIES = 3

        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)

            resp = None
            n_preds = int(self.n_gens/self.n_templates) if self.bootstrapping else int(self.n_gens)
            for retry in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    self.rate_limiter.add_request(request_text=user_message, current_time=start_time)
                    resp = await openai.ChatCompletion.acreate(
                        engine=self.chat_engine,
                        messages=message,
                        temperature=0.7,
                        max_tokens=8,
                        top_p=0.95,
                        n=max(n_preds, 3),            # e.g. for 5 templates, get 2 generations per template
                        request_timeout=10
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
                    sample_preds = []
                    all_gens_text = [x['message']['content'] for template_response in sample_response for x in template_response[0]['choices'] ]        # fuarr this is some high level programming
                    for gen_text in all_gens_text:
                        gen_pred = re.findall(r"## (-?[\d.]+) ##", gen_text)
                        if len(gen_pred) == 1:
                            sample_preds.append(float(gen_pred[0]))
                        else:
                            sample_preds.append(np.nan)
                            
                    while len(sample_preds) < self.n_gens:
                        sample_preds.append(np.nan)

                    tot_cost += sum([x[1] for x in sample_response])
                    tot_tokens += sum([x[2] for x in sample_response])
                all_preds.append(sample_preds)
        
        end = time.time()
        time_taken = end - start

        success_rate = sum(bool_pred_returned)/len(bool_pred_returned)

        all_preds = np.array(all_preds).astype(float)
        y_mean = np.nanmean(all_preds, axis=1)
        y_std = np.nanstd(all_preds, axis=1)

        # Capture failed calls - impute None with average predictions
        y_mean[np.isnan(y_mean)]  = np.nanmean(y_mean)
        y_std[np.isnan(y_std)]  = np.nanmean(y_std)
        y_std[y_std<1e-5] = 1e-5  # replace small values to avoid division by zero

        return y_mean, y_std, success_rate, tot_cost, tot_tokens, time_taken
    
    async def _evaluate_candidate_points(self, observed_configs, observed_fvals, candidate_configs, 
                                         use_context='full_context', use_feature_semantics=True, return_ei=False):
        '''Evaluate candidate points using the LLM model.'''

        if self.prompt_setting is not None:
            use_context = self.prompt_setting

        all_run_cost = 0
        all_run_time = 0

        tot_cost = 0
        time_taken = 0

        if self.use_recalibration and self.recalibrator is None:
            recalibrator, tot_cost, time_taken = await self._get_recalibrator(observed_configs, observed_fvals)
            if recalibrator is not None:
                self.recalibrator = recalibrator
            else:
                self.recalibrator = None
            print(f'[Recalibration] COMPLETED')

        all_run_cost += tot_cost
        all_run_time += time_taken

        all_prompt_templates, query_examples = gen_prompt_tempates(self.task_context, observed_configs, observed_fvals, candidate_configs, 
                                                                    n_prompts=self.n_templates, bootstrapping=self.bootstrapping,
                                                                    use_context=use_context, use_feature_semantics=use_feature_semantics, 
                                                                    shuffle_features=self.shuffle_features, apply_warping=self.apply_warping)

        print('*'*100)
        print(f'Number of all_prompt_templates: {len(all_prompt_templates)}')
        print(f'Number of query_examples: {len(query_examples)}')
        print(all_prompt_templates[0].format(Q=query_examples[0]['Q']))

        response = await self._predict(all_prompt_templates, query_examples)

        y_mean, y_std, success_rate, tot_cost, tot_tokens, time_taken = response

        if self.recalibrator is not None:
            recalibrated_res = self.recalibrator(y_mean, y_std, 0.68)   # 0.68 coverage for 1 std
            y_std = np.abs(recalibrated_res.upper - recalibrated_res.lower)/2

        all_run_cost += tot_cost
        all_run_time += time_taken

        if not return_ei:
            return y_mean, y_std, all_run_cost, all_run_time
    
        else:
            # calcualte ei
            if self.lower_is_better:
                best_fval = np.min(observed_fvals.to_numpy())
                delta = -1*(y_mean - best_fval)
            else:
                best_fval = np.max(observed_fvals.to_numpy())
                delta = y_mean - best_fval
            with np.errstate(divide='ignore'):  # handle y_std=0 without warning
                Z = delta/y_std
            ei = np.where(y_std>0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)

            return ei, y_mean, y_std, all_run_cost, all_run_time

    def select_query_point(self, observed_configs, observed_fvals, candidate_configs):
        '''Select the next query point using expected improvement.'''

        # warp
        if self.warping_transformer is not None:
            observed_configs = self.warping_transformer.warp(observed_configs)
            candidate_configs = self.warping_transformer.warp(candidate_configs)

        y_mean, y_std, cost, time_taken = asyncio.run(self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs))
        if self.lower_is_better:
            best_fval = np.min(observed_fvals.to_numpy())
            delta = -1*(y_mean - best_fval)
        else:
            best_fval = np.max(observed_fvals.to_numpy())
            delta = y_mean - best_fval

        with np.errstate(divide='ignore'):  # handle y_std=0 without warning
            Z = delta/y_std

        ei = np.where(y_std>0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)

        best_point_index = np.argmax(ei)

        # unwarp
        if self.warping_transformer is not None:
            candidate_configs = self.warping_transformer.unwarp(candidate_configs)

        best_point = candidate_configs.iloc[[best_point_index], :]  # return selected point as dataframe not series

        return best_point, cost, time_taken



