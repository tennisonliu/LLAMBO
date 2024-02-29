import time
import tiktoken

'''
There is a major problem with the RateLimiter, it cannot anticipate that a request will exceed limit.
'''

class RateLimiter:
    def __init__(self, max_tokens, time_frame, max_requests=700):
        # max number of tokens that can be used within time_frame
        self.max_tokens = max_tokens
        # max number of requests that can be made within time_frame
        self.max_requests = max_requests
        # time in seconds for which max_tokens is applicable
        self.time_frame = time_frame
        # keeps track of when tokens were used
        self.timestamps = []
        # keeps track of tokens used at each timestamp
        self.tokens_used = []
        # keeps track of the number of requests made
        self.request_count = 0

    def add_request(self, request_text=None, request_token_count=None, current_time=None):
        if current_time is None:
            current_time = time.time()

        # Check old requests and remove them if they're outside the time frame
        while self.timestamps and self.timestamps[0] < current_time - self.time_frame:
            self.timestamps.pop(0)
            self.tokens_used.pop(0)
            self.request_count -= 1

        # Add new request
        self.timestamps.append(current_time)

        if request_text is not None:
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            num_tokens = len(encoding.encode(request_text))
        elif request_token_count is not None:
            num_tokens = request_token_count
        else:
            raise ValueError('Either request_text or request_token_count must be specified.')

        self.tokens_used.append(num_tokens)

        self.request_count += 1

        if self.request_count >= self.max_requests:
            sleep_time = (self.timestamps[0] + self.time_frame) - current_time
            print(f'[Rate Limiter] Sleeping for {sleep_time:.2f}s to avoid hitting the request limit...')
            time.sleep(sleep_time)
            self.request_count = 0
            

        # If the sum of tokens used in the current time frame exceeds the max tokens
        if sum(self.tokens_used) > self.max_tokens:
            # Sleep for the remainder of the time frame
            sleep_time = (self.timestamps[0] + self.time_frame) - current_time
            print(f'[Rate Limiter] Sleeping for {sleep_time:.2f}s to avoid hitting the token limit...')
            time.sleep(sleep_time)
            # Clear the old requests after waking up
            self.timestamps.clear()
            self.tokens_used.clear()