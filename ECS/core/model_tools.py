import warnings
warnings.filterwarnings("ignore")
from importlib import import_module
from datetime import datetime
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import \
    confusion_matrix, \
    accuracy_score, \
    precision_score, \
    recall_score, \
    f1_score
from joblib import dump, load
import pandas as pd
import numpy as np


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


def save_model(model, path: str, metadata: dict) -> None:
    """
    Сохранить модель в файл.
    Функция используется для абстрагирования от используемого метода.
    Логичнее было бы использовать hdf5 или pickle,
    но в АТС почему-то используется joblib.
    Скорее всего, внутри все равно сидит pickle.
    :param metadata: словарь с дополнительной информацией
    :param model: объект модели
    :param path: путь к файлу для сохранения
    :return: None
    """
    with open(path, "wb") as file:
        dump(model, file)
        dump(metadata, file)


def load_model(path: str) -> tuple:
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


def create_report(model, x_test: list, y_test: list):
    # Для обратной совместимости используется старый код
    pred = []
    for p in model.predict_proba(x_test):
        all_prob = pd.Series(p, index=model.classes_)
        pred.append(list(all_prob.sort_values(ascending=False).index))
    return count_stats(predicts=pred, y_test=y_test,
                       amounts=[1, 2, 3, 4, 5, -1])


def save_excel_report(report: dict, path: str, rubricator: str) -> None:
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    names = list(map(str, report.keys()))
    names.sort()
    for i in names:
        report[i].to_excel(writer, sheet_name=f"{rubricator}_{i}")
    writer.save()


def save_txt_report(report: str, path: str) -> None:
    with open(path, "w") as file:
        file.write(report)


def seconds_to_duration(seconds: int) -> str:
    hours = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}:{mins}:{secs}"


def create_description(model_name: str, hyper_grid: dict,
                       best_params: dict, train_file: str,
                       test_file: str, train_size: int,
                       test_size: int, training_secs: int,
                       stats: dict) -> str:
    """
    Сформировать отчет в текстовом виде
    :param model_name: название модели
    :param hyper_grid: сетка гиперпараметров
    :param best_params: словарь найденных лучших гиперпараметров
    :param train_file: путь к тренировочному файлу
    :param test_file: путь к тестовому файлу
    :param train_size: размер обучающей выборки
    :param test_size: размер тестовой выборки
    :param training_secs: время обучения в секундах
    :param stats: табличный отчет, сгенерированный через count_stats
    :return: Строку с отчетом
    """
    now = datetime.today()
    descr = f"Date of creation: {now.day}.{now.month}.{now.year}\n"
    descr += f"Type of classifier:\t {model_name}\n"
    descr += "Tested parameters:"
    for param, value in hyper_grid.items():
        descr += f"\n\t{param}: {value}"
    descr += "\nBest parameters:"
    for param, value in best_params.items():
        descr += f"\n\t{param}: {value}"
    descr += f"\nTrain and test data sizes and files:\n"
    descr += f"\t{train_size}\t{train_file}\n"
    descr += f"\t{test_size}\t{test_file}\n"
    descr += f"Total training time:\t {seconds_to_duration(training_secs)}\n"
    descr += "\nResults (accuracy, precision, recall, f1-score):\n"
    keys = list(stats.keys())
    keys.sort()
    for i in keys:
        mac = stats[i].loc['macro']
        mic = stats[i].loc['micro']
        macro_str = f"{mac['accuracy'].round(5)}\t{mac['precision'].round(5)}\t" \
                    f"{mac['recall'].round(5)}\t{mac['f1-score'].round(5)}"
        micro_str = f"{mic['accuracy'].round(5)}\t{mic['precision'].round(5)}\t" \
                    f"{mic['recall'].round(5)}\t{mic['f1-score'].round(5)}"
        descr += f"For {i} answers: \n\tMacro {macro_str} \n\tMicro {micro_str}\n"
    return descr


def dump_metadata_to_str(model_name: str, language: str,
                         rubricator: str, pooling: str, vector_dim: int) -> str:
    now = datetime.now()
    str_form = f"{language}_{rubricator}_{model_name}_{pooling}{vector_dim}"
    str_form += f"_{now.day}_{now.month}_{now.year}"
    return str_form


def create_report_fname(model_name: str, language: str,
                        rubricator: str, pooling: str, vector_dim: int) -> str:
    """
    Упаковать метаинформацию в строку.
    Если потом добавить расширение, получится имя файла.
    Пример результата: info_en_rgnti_perceptron_sum50_4_9_19
    :param model_name: название классификатора
    :param language: язык
    :param rubricator: 'subj', 'ipv' или 'rgnti'
    :param pooling: 'mean', 'max' или 'sum'
    :param vector_dim: размерность вектора
    :return: Сгенерированная строка
    """
    meta_str = dump_metadata_to_str(model_name=model_name, language=language,
                                    rubricator=rubricator, pooling=pooling, vector_dim=vector_dim)
    fname = f"info_{meta_str}"
    return fname


def create_model_fname(model_name: str, language: str,
                       rubricator: str, pooling: str, vector_dim: int) -> str:
    """
    Создать имя файла с расширением .plk и метаинформацией.
    Пример результата: clf_model_en_rgnti_perceptron_sum50_4_9_19.plk
    :param model_name: название классификатора
    :param language: язык
    :param rubricator: 'subj', 'ipv' или 'rgnti'
    :param pooling: 'mean', 'max' или 'sum'
    :param vector_dim: размерность вектора
    :return: Сгенерированная строка
    """
    meta_str = dump_metadata_to_str(model_name=model_name, language=language,
                                    rubricator=rubricator, pooling=pooling, vector_dim=vector_dim)
    fname = f"clf_model_{meta_str}.plk"
    return fname


def short_report(stats: dict, time_elapsed: int) -> str:
    mini_report = f"Work time is {seconds_to_duration(time_elapsed)}"
    mini_report += "\n\t\taccuracy\tprecision\trecall\tf1-score\n"
    keys = list(stats.keys())
    keys.sort()
    for i in keys:
        mac = stats[i].loc['macro']
        mic = stats[i].loc['micro']
        macro_str = f"{mac['accuracy'].round(5)}\t{mac['precision'].round(5)}\t" \
                    f"{mac['recall'].round(5)}\t{mac['f1-score'].round(5)}"
        micro_str = f"{mic['accuracy'].round(5)}\t{mic['precision'].round(5)}\t" \
                    f"{mic['recall'].round(5)}\t{mic['f1-score'].round(5)}"
        mini_report += f"For {i} answers:\n\tMacro {macro_str} \n\tMicro {micro_str}\n"
    return mini_report


##################################################################
##                                                              ##
#          ACHTUNG! Ниже находится Территория Легаси.            #
#    Тут был проведен лишь легкий косметический рефакторинг,     #
#    чтобы не ругался линтер. На этой заповедной территории      #
#    сохранены даже старые комментарии. Возможно, однажды это    #
#    будет  переписано. Кто знает, кто знает...                  #
##                                                              ##
##################################################################
def count_stats(predicts: list, y_test: list,
                amounts: list):
    """
    Counts statistics for predictions of a classifier

    Args:
    predicts    -- classifiers answers for X_test.
    y_test      -- real rubrics of X_test.
    legend      -- list with ordered unique rubrics. If equals to None,
                   legend will be created in alphabet order. !!!УДАЛЕНО!!!
    amounts     -- list with amounts of answers we want to test (-1 means all answers). 1 by default.
    version     -- int or str with version of file. 1 by default. !!!УДАЛЕНО!!!

    Возвращает словарь датафреймов. Ключи - '1'...'5' и 'all'
    """
    legend = [item for sublist in predicts for item in sublist]
    legend = pd.Series(map(str, legend))
    legend = legend.unique()
    legend.sort()
    legend = list(legend)
    if -1 not in amounts:
        amounts = list(map(int, amounts))
        amounts.sort()
        amounts = amounts[::-1]
    else:
        amounts = list(set(amounts) - {-1})
        amounts.sort()
        amounts = [-1] + amounts[::-1]
    keys, values = [], []
    for a in amounts:
        k = []
        if a != -1:
            for j in predicts:
                k += [j[:a]]
        else:
            k = predicts
        cur_pred = k
        stats = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1-score', 'TP', 'FP', 'FN', 'TN'])
        for i in legend:
            cur_predicts = []
            cur_y_test = []
            for j in zip(cur_pred, y_test):
                if (type(j[0]) == list and i in j[0]) or i == j[0]:
                    cur_predicts += [1]
                else:
                    cur_predicts += [0]
                if (type(j[1]) == list and i in j[1]) or i == j[1]:
                    cur_y_test += [1]
                else:
                    cur_y_test += [0]
            temp = []
            for l in binary_metrics(cur_predicts, cur_y_test):
                temp += [l]
            mat = confusion_matrix(cur_predicts, cur_y_test)
            if mat.shape == (1, 1):
                conf_matrix = [0, 0, 0] + list(np.array(mat).ravel())
            else:
                conf_matrix = list(np.array(mat).ravel())[::-1]
            stats = stats.append(pd.DataFrame([temp + conf_matrix],
                                              columns=['accuracy', 'precision', 'recall',
                                                       'f1-score', 'TP', 'FP', 'FN', 'TN'], index=[i]))
        stats = stats.sort_index()
        stats_mean = stats.mean().values
        tp, fp, fn, tn = stats_mean[4:]
        acc_temp = (tp + tn) / (tp + fp + fn + tn)
        pr_temp = tp / (tp + fp)
        rec_temp = tp / (tp + fn)
        f1_temp = 2 * pr_temp * rec_temp / (pr_temp + rec_temp)
        stats = stats.append(pd.DataFrame([list(stats_mean[0:4]) + ['-'] * 4],
                                          columns=['accuracy', 'precision', 'recall', 'f1-score',
                                                   'TP', 'FP', 'FN', 'TN'],
                                          index=['macro']))
        stats = stats.append(pd.DataFrame([[acc_temp, pr_temp, rec_temp, f1_temp] + list(stats_mean[4:])],
                                          columns=['accuracy', 'precision', 'recall', 'f1-score',
                                                   'TP', 'FP', 'FN', 'TN'],
                                          index=['micro']))
        if a != -1:
            keys += [str(a)]
        else:
            keys += ['all']
        values += [stats]
    full_stats = dict(zip(keys, values))
    return full_stats


def binary_metrics(predicts, y_test):
    """
    Counts binary accuracy, precision, recall and f1.

    Args:
    predicts    -- classifiers answers for X_test (0 and 1 for a particular rubric).
    y_test      -- real rubrics of X_test (0 and 1 for a particular rubric).
    """
    ac = accuracy_score(y_test, predicts)
    pr = precision_score(y_test, predicts)
    rec = recall_score(y_test, predicts)
    f1 = f1_score(y_test, predicts)
    return [ac, pr, rec, f1]


if __name__ == '__main__':
    trained_model = load_model(r"D:\Desktop\VINITI\ECS\test\unit_tests\test_data\test_model.plk")
