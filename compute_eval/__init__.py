import importlib
from collections.abc import Callable

# Execution result type.
#
# Tuple of (exit_code, timed_out, output)
EvaluatorRuntimeResult = tuple[int, bool, str]

# Evaluator runtime function type.
#
# Callable that takes (command: str, timeout_seconds: float, task_tag: str | None)
# Returns ExecutionResult
EvaluatorRuntime = Callable[[str, float, str | None], EvaluatorRuntimeResult]

__all__ = ["EvaluatorRuntime", "EvaluatorRuntimeResult"]

# Load Nvidia internal extensions if available
try:
    from compute_eval.internal import MODEL_CLASSES

    def _lazy_load_class(class_path: str):
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def get_model_class(model: str):
        class_path = MODEL_CLASSES.get(model, "compute_eval.models.openAI_model.OpenAIModel")
        return _lazy_load_class(class_path)

except ImportError:
    from compute_eval.models.openAI_model import OpenAIModel

    def get_model_class(model: str):
        return OpenAIModel
