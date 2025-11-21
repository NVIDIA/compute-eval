import dotenv

from compute_eval.models.model_interface import ModelInterface
from compute_eval.token_provider import get_token_for_url

# Check API keys in order of preference
_api_key_names = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEMO_API_KEY"]


class OpenAIModel(ModelInterface):
    """
    Generate code completions using OpenAI models.

    Args:
        base_url (str): Base URL for the OpenAI API model.
        model_name (str): Name of the model to use for generating completions.
    """

    _api_key_printed = False

    def __init__(
        self,
        model_name: str,
        base_url: str | None,
        reasoning: str | None = None,
    ):
        dotenv.load_dotenv()

        self._model_name = model_name
        self._base_url = base_url or "https://api.openai.com/v1"
        self.reasoning = reasoning

        self._api_key_name = None

        for key_name in _api_key_names:
            if get_token_for_url(self._base_url, key_name) is not None:
                self._api_key_name = key_name
                break

        if self._api_key_name is None:
            raise Exception(
                f"Could not find any of: {', '.join(_api_key_names)}. Please set one of these environment variables."
            )

        if not OpenAIModel._api_key_printed:
            print(f"Using {self._api_key_name} for authentication")
            OpenAIModel._api_key_printed = True

    @property
    def api_key(self) -> str:
        url = get_token_for_url(self.base_url, self._api_key_name)
        if url is None:
            raise Exception(f"Could not get {self._api_key_name}.")
        return url

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model_name(self) -> str:
        return self._model_name
