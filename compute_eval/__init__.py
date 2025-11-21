import importlib

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
