import warnings
warnings.filterwarnings("ignore")

from importlib import import_module

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump, load


def load_class(classpath: str) -> type:
    """
    Импортировать класс по его пути
    :param classpath: полное имя класса (например, nltk.stem.snowball.RussianStemmer)
    :return: тип загружаемого класса
    """
    components = classpath.split(".")
    class_name = components[-1]
    module_name = ".".join(components[:-1])
    module = import_module(module_name)
    class_type = getattr(module, class_name)
    return class_type


def run_grid_search(model_instance,
                    hyperparameters: dict,
                    x_train: list,
                    y_train: list,
                    binary: bool,
                    n_folds: int,
                    n_jobs: int) -> dict:
    """
    Запустить поиск по сетке параметров для модели для поиска лучшей комбинации,
    чтобы затем обучить ее на полной выборке без CV
    :param model_instance: инстанс модели, например MLPClassifier()
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
    # TODO: Возможно, стоит добавить таймер
    scoring = 'f1_weighted'
    skf = StratifiedKFold(shuffle=True, n_splits=n_folds)
    hypers_copy = hyperparameters.copy()
    if binary:
        model_instance = OneVsRestClassifier(model_instance)
        for key in hyperparameters:
            hypers_copy["estimator__" + key] = hypers_copy.pop(key)
    grid_searcher = GridSearchCV(estimator=model_instance,
                                 param_grid=hypers_copy,
                                 n_jobs=n_jobs,
                                 scoring=scoring,
                                 cv=skf,
                                 verbose=0)
    grid_searcher.fit(x_train, y_train)
    best_params = grid_searcher.best_params_
    # # Убираем "estimator__" из ключей
    # best_params = {}
    # for key, value in grid_searcher.best_params_.items():
    #     best_params[key.split("estimator__")[-1]] = value
    return best_params


def refit_model(model_instance, best_params: dict,
                x_train: list, y_train: list, binary: bool):
    if binary:
        model_instance = OneVsRestClassifier(model_instance)
    model_instance.set_params(**best_params)
    model_instance.fit(x_train, y_train)
    return model_instance


def save_model(model, path: str) -> None:
    """
    Сохранить модель в файл.
    Функция используется для абстрагирования от используемого метода.
    Логичнее было бы использовать hdf5 или pickle,
    но в АТС почему-то используется joblib.
    :param model: объект модели
    :param path: путь к файлу для сохранения
    :return: None
    """
    dump(model, path)


def load_model(path: str):
    """
    Загрузить модель из файла
    :param path: путь к модели
    :return: объект модели
    """
    return load(path)