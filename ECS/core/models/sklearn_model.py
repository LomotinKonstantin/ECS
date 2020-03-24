from joblib import dump, load

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from ECS.core.models.abstract_model import AbstractModel
from ECS.interface.logging_tools import get_logger, info_ps


class SklearnModel(AbstractModel):

    def __init__(self, classpath: str):
        super().__init__(classpath)
        self.instance = self.class_type()
        self.logger = get_logger(f"SklearnModel {classpath}")

    def save(self, path: str, metadata: dict):
        """
        Сохранить модель в файл.
        Функция используется для абстрагирования от используемого метода.
        Логичнее было бы использовать hdf5 или pickle,
        но в АТС почему-то используется joblib.
        Скорее всего, внутри все равно сидит pickle.
        :param metadata: словарь с дополнительной информацией
        :param path: путь к файлу для сохранения
        :return: None
        """
        with open(path, "wb") as file:
            dump(self.instance, file)
            dump(metadata, file)

    def load(self, path: str) -> tuple:
        """
        Загрузить модель из файла
        :param path: путь к модели
        :return: кортеж (объект_модели, словарь_метадаты)
                 Если модель старой версии и не содержит метадаты,
                 на втором месте будет None
        """
        metadata = None
        with open(path, "rb") as file:
            model = load(file)
            try:
                metadata = load(file)
            except EOFError:
                pass
        return model, metadata

    def refit(self, best_params: dict, x_train: list,
              y_train: list, **kwargs) -> None:
        binary = kwargs["binary"]
        self.instance = self.class_type()
        if binary:
            self.instance = OneVsRestClassifier(self.instance)
        self.instance.set_params(**best_params)
        self.instance.fit(x_train, y_train)

    def run_grid_search(self,
                        hyperparameters: dict,
                        x_train: list,
                        y_train: list,
                        # Для совместимости интерфейса с KerasModel
                        **kwargs) -> dict:
        """
        Запустить поиск по сетке параметров для модели для поиска лучшей комбинации,
        чтобы затем обучить ее на полной выборке без CV
        :param hyperparameters: словарь гиперпараметров,
                                где ключи - названия из документации,
                                а значения - списки значений гиперпараметров
        :param x_train: array-like список обучающих векторов
        :param y_train: array-like список меток
        :param binary: использовать ли OneVsAll
        :param n_folds: количество фолдов кросс-валидации
        :param n_jobs: количество параллельных потоков
        :return: словарь лучших параметров в совместимом формате
        """
        binary = kwargs["binary"]
        n_folds = kwargs["n_folds"]
        n_jobs = kwargs["n_jobs"]
        scoring = 'f1_weighted'
        skf = StratifiedKFold(shuffle=True, n_splits=n_folds)
        hypers_copy = hyperparameters.copy()
        if binary:
            self.instance = OneVsRestClassifier(self.instance)
        for key in hyperparameters:
            hypers_copy["estimator__" + key] = hypers_copy.pop(key)
        grid_searcher = GridSearchCV(estimator=self.instance,
                                     param_grid=hypers_copy,
                                     n_jobs=n_jobs,
                                     scoring=scoring,
                                     cv=skf,
                                     verbose=0)
        info_ps(self.logger, "Fitting grid-searcher...")
        grid_searcher.fit(x_train, y_train)
        best_params = grid_searcher.best_params_
        # # Убираем "estimator__" из ключей
        # best_params = {}
        # for key, value in grid_searcher.best_params_.items():
        #     best_params[key.split("estimator__")[-1]] = value
        return best_params

    def predict_proba(self, data):
        return self.instance.predict_proba(data)

    def predict(self, data):
        return self.instance.predict(data)
