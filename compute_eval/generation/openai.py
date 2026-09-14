import dotenv

from .auth import get_api_token
from .model import ModelInterface

_API_KEY_NAMES = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEMO_API_KEY"]


class OpenAIModel(ModelInterface):
    _api_key_printed = False

    def __init__(self, model_name: str, base_url: str | None):
        dotenv.load_dotenv()
        self._model_name = model_name
        self._base_url = base_url or "https://api.openai.com/v1"
        self._api_key_name = next(
            (name for name in _API_KEY_NAMES if get_api_token(self._base_url, name) is not None),
            None,
        )
        if self._api_key_name is None:
            raise Exception(f"Could not find any of: {', '.join(_API_KEY_NAMES)}")
        if not OpenAIModel._api_key_printed:
            print(f"Using {self._api_key_name} for authentication")
            OpenAIModel._api_key_printed = True

    @property
    def api_key(self) -> str:
        token = get_api_token(self.base_url, self._api_key_name)
        if token is None:
            raise Exception(f"Could not get {self._api_key_name}")
        return token

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model_name(self) -> str:
        return self._model_name
