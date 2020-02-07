from ECS.core.model_tools import load_class


class AbstractModel:

    def __init__(self, classpath: str = ""):
        """
            :raise ImportError
        """
        self.instance = None
        self.class_type = None
        if classpath:
            self.class_type = load_class(classpath)

    def load(self, path: str):
        pass

    def save(self, path: str, metadata: dict) -> None:
        pass

    def predict(self, data):
        pass

    def predict_proba(self, data):
        pass

    def run_grid_search(self, **kwargs):
        pass

    def refit(self, **kwargs):
        pass
