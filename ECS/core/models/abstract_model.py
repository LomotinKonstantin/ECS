class AbstractModel:

    def load(self, path: str):
        pass

    def save(self, path: str):
        pass

    def predict(self, data):
        pass

    def predict_proba(self, data):
        pass

    '''
    Включает в себя refit
    '''
    def run_grid_search(self):
        pass

    def __load_class(self, path: str):
        pass
