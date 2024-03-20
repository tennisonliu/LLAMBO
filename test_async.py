import os
import time
import asyncio
from openai import AsyncOpenAI

import ollama
from ollama import AsyncClient as AsyncOllamaClient


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recursively convert it to an object
                value = DictToObject(value)
            setattr(self, key, value)

async def _async_generate_oai(self, user_message):
    '''Generate a response from the LLM async.'''
    message = []
    message.append({"role": "system","content": "You are an AI assistant that helps people find information."})
    message.append({"role": "user", "content": user_message})


    async with AsyncOpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_retries=3,
        timeout=3,
    ) as client:
        resp = None
        try:
            self.rate_limiter.add_request(request_text=user_message, current_time=time.time())
            resp = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0.8,
                max_tokens=500,
                top_p=0.95,
                n=self.n_gens,
            )
            self.rate_limiter.add_request(request_token_count=resp.usage.total_tokens, current_time=time.time())
        except Exception as e:
            raise e
            # print(resp)
            # print(e)


    if resp is None:
        raise Exception('Response is None')

    tot_tokens = resp.usage.total_tokens
    tot_cost = 0.0015*(resp.usage.prompt_tokens/1000) + 0.002*(resp.usage.completion_tokens/1000)

    return resp, tot_cost, tot_tokens



async def _async_generate(user_message):
    '''Generate a response from the LLM async.'''
    message = []
    message.append({"role": "system","content": "You are an AI assistant that helps people find information."})
    message.append({"role": "user", "content": user_message})


    resp = None
    try:
        # self.rate_limiter.add_request(request_text=user_message, current_time=time.time())
        resp = await AsyncOllamaClient().chat(
            model="llama2:13b",
            messages=message,
            options={
                "temperature": 0.8,
                "top_p": 0.95,
            }
        )
        # self.rate_limiter.add_request(request_token_count=resp.usage.total_tokens, current_time=time.time())
    except Exception as e:
        raise e


    if resp is None:
        raise Exception('Response is None')

    resp = DictToObject(resp)
    def list_custom_attributes(obj):
        return [attr for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr))]

    msg_attr = list_custom_attributes(resp.message)
    print(msg_attr)
    print(resp.message.content)
    tot_tokens = resp.eval_count
    tot_cost = 0.0015*(resp.eval_count/1000) + 0.002*(resp.eval_count/1000)

    return resp, tot_cost, tot_tokens



if __name__ == "__main__":
    asyncio.run(_async_generate('How do I keep a session?'))