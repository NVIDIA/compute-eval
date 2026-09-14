import random
import time
from abc import ABC, abstractmethod

from openai import APIStatusError, OpenAI

RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def retry_with_exponential_backoff(
    function,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
):
    """Retry retriable API failures with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return function(*args, **kwargs)
            except Exception as error:
                if not isinstance(error, APIStatusError):
                    raise Exception(
                        f"An error occurred when accessing the model API. Check your endpoint and payload: {error}"
                    ) from None

                status_code = error.status_code
                if status_code in RETRIABLE_STATUS_CODES:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded") from None
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Model API returned {status_code}; retrying after {delay:.2f} seconds")
                    time.sleep(delay)
                    continue
                if status_code == 400:
                    raise Exception("Invalid request was made. Check the headers and payload") from None
                if status_code == 401:
                    raise Exception("Unauthorized HTTP request. Check your headers and API key") from None
                if status_code == 403:
                    raise Exception("You are forbidden from accessing this resource") from None
                raise Exception(f"Model API returned status {status_code}: {error}") from None

    return wrapper


class ModelInterface(ABC):
    @property
    @abstractmethod
    def api_key(self) -> str:
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @retry_with_exponential_backoff
    def call_chat_completions_endpoint(self, **kwargs):
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=900.0)
        return client.chat.completions.create(**kwargs)

    @retry_with_exponential_backoff
    def call_completions_endpoint(self, **kwargs):
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=900.0)
        return client.completions.create(**kwargs)

    def generate_response(self, system_prompt, prompt, request_options):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request = {"model": self.model_name, "messages": messages, "stream": False}
        if request_options.get("thinking", False):
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "thinking": True}}
        if (reasoning := request_options.get("reasoning")) is not None:
            request["reasoning_effort"] = reasoning

        for parameter in ("temperature", "top_p", "max_tokens", "stop"):
            if (value := request_options.get(parameter)) is not None:
                request[parameter] = value

        response = self.call_chat_completions_endpoint(**request)
        try:
            return response.choices[0].message.content
        except Exception:
            raise Exception("There was an error when accessing the completion") from None

    def generate_text_completion(self, prompt, request_options):
        request = {"model": self.model_name, "prompt": prompt, "stream": False}
        for parameter in ("temperature", "top_p", "max_tokens", "stop"):
            if (value := request_options.get(parameter)) is not None:
                request[parameter] = value

        response = self.call_completions_endpoint(**request)
        try:
            return response.choices[0].text
        except Exception:
            raise Exception("There was an error when accessing the text completion") from None
