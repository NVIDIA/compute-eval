import random
import time
from abc import ABC, abstractmethod

from openai import OpenAI

RETRIABLE_STATUS_CODES = [
    # These are server side errors where we can get correct response if we try again later
    429,  # Too many requests, happens you are exceeding the rate limit
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout Error
]


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except Exception as e:
                status_code = getattr(e, "status_code", None)

                # Check if the status code is retriable status code
                if status_code in RETRIABLE_STATUS_CODES:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.") from None

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Print error message
                    print(f"Error occurred {str(e)}, retrying after {delay:.2f} seconds.")

                    # Sleep for the delay
                    time.sleep(delay)
                elif status_code == 400:
                    raise Exception("Invalid request was made. Check the headers and payload") from None
                elif status_code == 401:
                    raise Exception("Unauthorized HTTP request. Check your headers and API key") from None
                elif status_code == 403:
                    raise Exception("You are forbidden from accessing this resource") from None
                else:
                    raise Exception(
                        f"An error occurred when accessing the model API. Check your headers and payload. Error: {str(e)}"
                    ) from None

    return wrapper


class ModelInterface(ABC):
    """
    Base class for generating code completions.
    """

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
        """
        Call the chat completions endpoint of the model API.
        """
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return client.chat.completions.create(**kwargs)

    def generate_response(self, system_prompt, prompt, params):
        """
        Generate code completions by communicating with the OpenAI API.

        Args:
            system_prompt (str, optional): The system prompt to use for generating completions.
            prompt (str): The user prompt to use for generating completions.
            params (dict, optional): Additional parameters for the API call.

        Returns:
            str: The generation result from the model.
        """

        messages = []

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        params_dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        if (reasoning := getattr(self, "reasoning", None)) is not None:
            params_dict["reasoning_effort"] = reasoning

        # Add optional parameters only if they're not None
        # Some models (e.g. o1-mini) don't support passing some args
        # We need to exclude them
        temperature = get_parameter_value("temperature", params, None)
        if temperature is not None:
            params_dict["temperature"] = temperature

        top_p = get_parameter_value("top_p", params, None)
        if top_p is not None:
            params_dict["top_p"] = top_p

        max_tokens = get_parameter_value("max_tokens", params, 2048)
        if max_tokens is not None:
            params_dict["max_tokens"] = max_tokens

        response = self.call_chat_completions_endpoint(**params_dict)

        try:
            completion = response.choices[0].message.content
        except KeyError as e:
            print(f"WARNING: The completion object is invalid. Could not find the key {str(e)}")
            completion = ""
        except Exception:
            raise Exception("There was an error when accessing the completion") from None

        return completion


def get_parameter_value(parameter, parameters, default_value):
    if parameters is not None and parameter in parameters:
        return parameters[parameter]
    else:
        return default_value
