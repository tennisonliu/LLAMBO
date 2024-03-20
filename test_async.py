import os
import time
import asyncio
from openai import AsyncOpenAI




async def _async_generate(user_message):
    '''Generate a response from the LLM async.'''
    message = []
    message.append({"role": "system","content": "You are an AI assistant that helps people find information."})
    message.append({"role": "user", "content": user_message})


    client = AsyncOpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_retries=3,
        timeout=10,
    )

    try:
        start_time = time.time()
        resp = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0.8,
            max_tokens=500,
            top_p=0.95,
            # n=self.n_gens,
        )
        # self.rate_limiter.add_request(request_token_count=resp['usage']['total_tokens'], current_time=start_time)
    except Exception as e:
        print(resp)
        print(e)


    if resp is None:
        raise Exception('Response is None')

    print(resp)

    tot_tokens = resp.usage.total_tokens
    tot_cost = 0.0015*(resp.usage.prompt_tokens/1000) + 0.002*(resp.usage.completion_tokens/1000)

    return resp, tot_cost, tot_tokens



asyncio.run(_async_generate('How do I keep a session?'))