import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("..")
from argparse import ArgumentParser
import os
from time import time
from collections import Counter
# KERAS_TODO
# Все хорошенько переписать
from ECS.interface.valid_config import ValidConfig
from ECS.interface.logging_tools import create_logger, error_ps
from ECS.core.dataset import Dataset
from ECS.core.models.sklearn_model import SklearnModel
from ECS.core.models.keras_model import KerasModel
from ECS.core.data_tools import \
    generate_w2v_fname, \
    timestamp
from ECS.core.model_tools import \
    create_report, \
    save_excel_report, \
    create_description, \
    create_report_fname, \
    save_txt_report, \
    save_model, \
    create_model_fname, \
    short_report, \
    seconds_to_duration


def add_args(parser: ArgumentParser):
    parser.add_argument("exp_path",
                        type=str,
                        help="Full path to the folder with settings.ini file", )


def inplace_drop_rubrics(x: list, y: list, rubrics) -> None:
    """
    Убрать из выборки записи с метками rubrics
    !!! МЕНЯЕТ АРГУМЕНТЫ x И y !!!

    :param x: массив данных
    :param y: массив соответствующих меток классов
    :param rubrics: Коллекция рубрик, которые надо удалить
    :return: None
    """
    for drop_rubr in rubrics:
        # Получаем список индексов рубрики
        indices = [ind for ind, val in enumerate(y) if val == drop_rubr]
        # Удаляем записи
        for ind in reversed(indices):
            del y[ind]
            del x[ind]


def inplace_rubric_filter(x: list, y: list, threshold: int) -> dict:
    """
    Убрать из выборки записи из рубрик, которые встречаются
    меньше, чем threshold раз
    !!! МЕНЯЕТ АРГУМЕНТЫ x И y !!!

    :param threshold: минимальное количество текстов в рубрике
    :param x: массив данных
    :param y: массив соответствующих меток классов
    :return: словарь удаленных рубрик с количеством текстов в каждой
    """
    # Считаем сколько текстов в рубриках
    c = Counter(y)
    # Создаем фильтр для рубрик, размер которых ниже порога
    to_drop = list(filter(lambda c_key: c[c_key] < threshold, c))
    res = {k: v for k, v in c.items() if k in to_drop}
    inplace_drop_rubrics(x, y, to_drop)
    return res


def inplace_keep_rubrics(x: list, y: list, rubrics) -> None:
    indices = [ind for ind, val in enumerate(y) if val not in rubrics]
    # Удаляем записи
    for ind in reversed(indices):
        del y[ind]
        del x[ind]


def main():
    # Получаем аргументы командной строки
    argparser = ArgumentParser()
    add_args(argparser)
    args = argparser.parse_args()
    exp_path = args.exp_path
    settings_path = os.path.join(exp_path, "settings.ini")
    # Создаем логгер

    if not os.path.exists(settings_path):
        print(f"No settings.ini file found in {exp_path}")
        exit(0)

    logger = create_logger(os.path.join(exp_path, f"{timestamp()}.log"), "ecs.main")

    # Загружаем и проверяем конфиг
    config = ValidConfig()
    config.read(settings_path, encoding="cp1251")
    logger.info("Validating experiment settings...")
    config.validate_all()
    logger.info("\n" + "=" * 20 + "Validation OK" + "=" * 20 + "\n")

    total_timer = time()

    # Находим датасет
    training_file = config.get("TrainingData", "dataset")
    test_file = config.get("TrainingData", "test_file")
    # Получаем параметры
    pooling = config.get("WordEmbedding", "pooling")
    binary = config.get_primitive("Experiment", "binary")
    rubricators = config.get_as_list("Experiment", "rubricator")
    n_jobs = config.getint("Experiment", "threads")
    n_folds = config.getint("Experiment", "n_folds")

    # Готовим датасет
    dataset = Dataset(config)

    # На этом этапе уже должны быть созданы папки кэша,
    # так как мы гарантированно прогнали все генераторы
    # Копируем файл настроек и W2V
    clear_cache_folder = os.path.dirname(dataset.get_train_clear_cache_path())
    vector_cache_folder = os.path.dirname(dataset.get_train_matrix_cache_path())
    with open(os.path.join(clear_cache_folder, "settings.ini"), "w") as clear_copy_file:
        config.write(clear_copy_file)
    with open(os.path.join(vector_cache_folder, "settings.ini"), "w") as vector_copy_file:
        config.write(vector_copy_file)
    w2v_model = dataset.get_w2v_model()
    w2v_fname = generate_w2v_fname(vector_dim=w2v_model.vector_size, language=dataset.get_language())
    w2v_cache_path = os.path.join(vector_cache_folder, w2v_fname)
    w2v_save_path = os.path.join(exp_path, w2v_fname)
    # Кэшируем w2v
    w2v_model.save(w2v_cache_path)
    # Сохраняем модель как результат
    logger.info(f"Saving W2V model to {w2v_save_path}")
    w2v_model.save(w2v_save_path)

    # Обучаем и тестируем модели
    model_names = config.get_as_list("Classification", "models")
    mapping_config = ValidConfig()
    mapping_config.read(os.path.join(os.path.dirname(__file__), "interface", "map.ini"))
    model_import_mapping = mapping_config.get_as_dict("SupportedModels")
    # Для каждого рубрикатора создается свой датасет, т.к. каждый текст
    # обладает разным количеством разных кодов
    # и если размножить векторы сразу для всех рубрикаторов,
    # память может быстро закончиться
    for rubricator in rubricators:
        x_train, x_test, y_train, y_test = dataset.sklearn_dataset_split(rubricator)
        min_training_rubr = config.get_primitive(rubricator, "min_training_rubric", fallback="0") or 1
        min_test_rubr = config.get_primitive(rubricator, "min_validation_rubric", fallback="0") or 1
        train_filter_res = {}

        if min_training_rubr > 1:
            train_filter_res = inplace_rubric_filter(x_train, y_train, threshold=min_training_rubr)
            log_str = f"Dropped rubrics from training dataset for {rubricator}:\n" + \
                      "\n".join([f"{k}\t({v} texts)" for k, v in train_filter_res.items()])
            logger.info(log_str)
        if min_test_rubr > 1:
            y_test_cntr = Counter(y_test)
            test_filter_res = inplace_rubric_filter(x_test, y_test, threshold=min_test_rubr)
            # В датасете тексты с множественными метками дублируются,
            # поэтому можно просто дропнуть записи с удаленными из train рубриками,
            # чтобы не учитывать их при тестировании
            inplace_drop_rubrics(x_test, y_test, rubrics=train_filter_res.keys())
            total_test_drop = test_filter_res.copy()
            for key in train_filter_res:
                total_test_drop[key] = y_test_cntr[key]
            log_str = "Dropped rubrics from test dataset:\n" + \
                      "\n".join([f"{k}\t({v} texts)" for k, v in total_test_drop.items()])
            logger.info(log_str)
        # Оставляем пересечение рубрик
        logger.info("Building training/test rubric intersection")
        intersect_rubrics = {k: v for k, v in Counter(y_test).items() if k in Counter(y_train)}
        inplace_keep_rubrics(x_train, y_train, intersect_rubrics)
        inplace_keep_rubrics(x_test, y_test, intersect_rubrics)
        # Пост-валидация
        for desc, ds in {"Training dataset": y_train, "Test dataset": y_test}.items():
            # Проверка на слишком строгие пороги
            if len(ds) == 0:
                logger.error(desc + " is empty! All the texts were removed due to the threshold")
                exit(0)
            # Проверка на неприятный баг. Все по непонятной причине ломается,
            # когда остаются только тексты любой одной рубрики
            if len(set(ds)) == 1:
                logger.error(f"{desc} contains only 1 rubric '{ds[1]}'. "
                             f"This will cause invalid behavior and ECS crash. Finishing execution")
                exit(0)

        for model_name in model_names:
            hypers = config.get_hyperparameters(model_name)
            if model_name == "svm":
                hypers["probability"] = [True]
            try:
                if model_name == "keras":
                    model_interface = KerasModel()
                else:
                    model_interface = SklearnModel(model_import_mapping[model_name])
            except ImportError as ie:
                logger.warning(f"\n>>> Unable to create model {model_name}, it will be skipped.")
                logger.warning(f">>> ({ie})\n")
                continue
            logger.info(f"Fitting parameters for model {model_name} by {rubricator}")
            timer = time()
            try:
                n_outputs = len(dataset.label_binarizers[rubricator].classes_)
                n_inputs = x_train[0].size
                grid_search_params = {
                    "binary": binary,
                    "n_folds": n_folds,
                    "n_jobs": n_jobs,
                    "dataset": dataset,
                    "n_inputs": n_inputs,
                    "n_outputs": n_outputs,
                    "rubricator": rubricator,
                    "conv_type": pooling,
                    "x_test": x_test,
                    "y_test": y_test,
                }
                best_params = model_interface.run_grid_search(hyperparameters=hypers,
                                                              x_train=x_train,
                                                              y_train=y_train,
                                                              **grid_search_params)
            except ValueError as ve:
                logger.warning(f"\n>>> Detected incorrect hyperparameters ({ve}) for model '{model_name}'."
                               f"It will be skipped.")
                continue
            except OSError as ose:
                state_str = f"(model: {model_name}, rubricator: {rubricator})"
                error_ps(logger, f"OS has interrupted the grid search process: {ose} {state_str}")
                exit(1)
            except AttributeError as ae:
                logger.error(f"Unsupported data type: {ae}")
                exit(1)
            else:
                try:
                    refit_params = {
                        "binary": binary,
                        "conv_type": pooling,
                        "dataset": dataset,
                        "rubricator": rubricator,
                    }
                    logger.info(f"Fitting best parameters to model {model_name}")
                    model_interface.refit(best_params=best_params,
                                          x_train=x_train, y_train=y_train,
                                          **refit_params)
                except OSError as ose:
                    state_str = f"(model: {model_name}, rubricator: {rubricator})"
                    error_ps(logger, f"OS has interrupted the refitting process: {ose} {state_str}")
                    exit(1)
                else:
                    time_elapsed = int(time() - timer)

                    # Сохраняем модель
                    model_fname = create_model_fname(model_name=model_name,
                                                     language=dataset.get_language(),
                                                     rubricator=rubricator, pooling=pooling,
                                                     vector_dim=w2v_model.vector_size)
                    model_path = os.path.join(exp_path, model_fname)
                    # Пока эта информация не используется, но в будущем может пригодиться
                    model_metadata = {
                        **dataset.get_matrix_md_filter(),
                        **best_params
                    }
                    logger.info(f"Saving model to {model_path}")
                    save_model(model=model_interface.instance, path=model_path, metadata=model_metadata)

                    # Создаем и сохраняем отчеты
                    logger.info("Testing model and creating report")
                    excel_report = create_report(model=model_interface.instance,
                                                 x_test=x_test,
                                                 y_test=y_test)
                    text_report = create_description(model_name=model_name,
                                                     hyper_grid=hypers,
                                                     best_params=best_params,
                                                     train_file=training_file,
                                                     test_file=test_file,
                                                     train_size=len(x_train),
                                                     test_size=len(x_test),
                                                     stats=excel_report,
                                                     training_secs=time_elapsed)
                    report_fname = create_report_fname(model_name=model_name,
                                                       language=dataset.get_language(),
                                                       rubricator=rubricator,
                                                       pooling=pooling,
                                                       vector_dim=w2v_model.vector_size)
                    excel_path = os.path.join(exp_path, f"{report_fname}.xlsx")
                    txt_path = os.path.join(exp_path, f"{report_fname}.txt")
                    logger.info(f"Saving reports to {excel_path} and {txt_path}")
                    save_excel_report(path=excel_path, report=excel_report, rubricator=rubricator)
                    save_txt_report(path=txt_path, report=text_report)
                    # Печатаем мини-отчет
                    mini_report = short_report(excel_report, time_elapsed)
                    logger.info(f"\n{mini_report}")
    logger.info("Done")
    logger.info(f"Total time elapsed: {seconds_to_duration(int(time() - total_timer))}")


if __name__ == '__main__':
    main()
