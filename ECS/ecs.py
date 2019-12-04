import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("..")
from argparse import ArgumentParser
import os
from time import time
from collections import Counter

from ECS.interface.valid_config import ValidConfig
from ECS.interface.logging_tools import create_logger, error_ps, get_logger
from ECS.preprocessor.Preprocessor2 import Preprocessor
from ECS.core.data_tools import \
    find_cached_clear, \
    find_cached_vectors, \
    vector_from_pkl_generator, \
    vector_from_csv_generator, \
    caching_vector_generator, \
    create_w2v, \
    generate_w2v_fname, \
    pp_from_csv_generator, \
    caching_pp_generator, \
    timestamp, \
    aggregate_full_dataset, \
    df_to_labeled_dataset, \
    create_labeled_tt_split, load_w2v
from ECS.core.model_tools import \
    load_class, \
    run_grid_search, \
    refit_model, \
    create_report, \
    save_excel_report, \
    create_description, \
    create_report_fname, \
    save_txt_report, \
    save_model, \
    create_model_fname, \
    short_report, \
    seconds_to_duration


def append_to_fname(fname: str, append: str, extension=None) -> str:
    components = fname.split(".")
    components[-2] += append
    if extension is not None:
        components[-1] = extension
    return ".".join(components)


def generate_clear_cache_path(raw_path: str, exp_name: str) -> str:
    """
    Создает название файла чистого кэша для сырого файла.
    Если надо, создает папку.
    :param raw_path: путь к сырому файлу
    :param exp_name: название эксперимента
    :return: сгенерированный путь
    """
    cache_folder = os.path.join(os.path.dirname(raw_path), f"{exp_name}_clear")
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    cache_fname = append_to_fname(os.path.basename(raw_path), "_clear", extension="csv")
    cache_fpath = os.path.join(cache_folder, cache_fname)
    return cache_fpath


def generate_vector_cache_path(raw_path: str, exp_name: str) -> str:
    """
    Создает название файла чистого кэша для сырого файла.
    Если надо, создает папку.
    :param raw_path: путь к сырому файлу
    :param exp_name: название эксперимента
    :return: сгенерированный путь
    """
    cache_folder = os.path.join(os.path.dirname(raw_path), f"{exp_name}_vectors")
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    cache_fname = append_to_fname(os.path.basename(raw_path), "_vectors", extension="pkl")
    cache_fpath = os.path.join(cache_folder, cache_fname)
    return cache_fpath


def extract_pp_settings(v_config: ValidConfig) -> dict:
    pp_section = v_config.get_as_dict("Preprocessing")
    pp_columns = {k: pp_section[k] for k in
                  ["id", "title", "text", "keywords", "subj", "ipv", "rgnti", "correct"]}
    pp_params = {k: pp_section[k] for k in
                 ["remove_stopwords", "normalization", "kw_delim", "language", "batch_size"]}
    pp_params["remove_formulas"] = True
    pp_params["default_lang"] = None
    pp_params["columns"] = pp_columns
    return pp_params


def create_caching_pp_gen(raw_path: str, exp_name: str, chunk_size: int, pp_params: dict):
    cache_file = generate_clear_cache_path(raw_path=raw_path, exp_name=exp_name)
    return caching_pp_generator(raw_file=raw_path,
                                chunk_size=chunk_size,
                                cache_path=cache_file,
                                **pp_params)


def create_clear_generators(base_dir: str,
                            clear_filter: dict,
                            chunk_size: int,
                            training_fpath: str,
                            test_fpath: str,
                            experiment_title: str,
                            pp_params: dict) -> dict:
    """
    Создает подходящий генератор чистых текстов.
    Если найден кэш, то создает читающий генератор,
    если нет - предобрабатывает сырые данные
    :param base_dir: базовая директория для поиска кэша
    :param clear_filter: словарь с настройками препроцессора
                         для поиска подходящего кэша
    :param chunk_size: размер чанка
    :param training_fpath: путь к сырому обучающему файлу
    :param test_fpath: путь к сырому тест-файлу (может быть равен '')
    :param experiment_title: название эксперимента
    :param pp_params: словарь с полными настройками препроцессора
    :return: словарь вида {'train': генератор, ['test': генератор]}
    """
    try:
        pp_sources = {}
        cached_clear = find_cached_clear(base_dir=base_dir, metadata_filter=clear_filter)
        clear_file_list = []
        # Если мы нашли кэш, список станет непустым
        # А если нет, то мы не зайдем в условие
        if len(cached_clear) > 0:
            # Если нашли больше, чем одну папку с чистыми текстами,
            # берем случайную
            _, clear_file_list = cached_clear.popitem()
        # Если список был пустым, то find_cache_for source
        # вернет пустую строку и будет создан кэширующий генератор
        train_pp_cache = find_cache_for_source(clear_file_list, training_fpath)
        if train_pp_cache:
            pp_sources[train_pp_cache] = pp_from_csv_generator(train_pp_cache, chunk_size)
        else:
            pp_sources[training_fpath] = create_caching_pp_gen(raw_path=training_fpath,
                                                               exp_name=experiment_title,
                                                               chunk_size=chunk_size,
                                                               pp_params=pp_params)
        # Тут возможно 2 ситуации:
        # 1. Тестового файла не предусмотрено (ничего не делать)
        # 2. Тестовый кэш не найден (создать кэширующий генератор)
        if test_fpath:
            test_pp_cache = find_cache_for_source(clear_file_list, test_fpath)
            if test_pp_cache:
                pp_sources[test_pp_cache] = pp_from_csv_generator(test_pp_cache, chunk_size)
            else:
                pp_sources[test_fpath] = create_caching_pp_gen(raw_path=test_fpath,
                                                               exp_name=experiment_title,
                                                               chunk_size=chunk_size,
                                                               pp_params=pp_params)
        return train_test_match(pp_sources,
                                train_raw_path=training_fpath,
                                test_raw_path=test_fpath)
    except Exception as e:
        logger = get_logger("ecs.create_clear_generators")
        error_ps(logger, f"Error occurred during creation of clear generators: {e}")
        exit(1)


def recognize_language(filename: str, encoding="utf8", n_lines=10) -> str:
    chunk = []
    with open(filename, encoding=encoding) as txt_file:
        for n, line in enumerate(txt_file):
            chunk.append(line)
            if line == n_lines - 1:
                break
    text_sample = " ".join(chunk)
    p = Preprocessor()
    return p.recognize_language(text_sample)


def add_args(parser: ArgumentParser):
    parser.add_argument("exp_path",
                        type=str,
                        help="Full path to the folder with settings.ini file", )


def find_cached_w2v(cache_folder: str) -> str:
    """
    Найти модель w2v в папке с кэшированными векторами
    :param cache_folder: папка с кэшированными векторами
    :return: путь к w2v или пустая строка
    """
    for entry in os.listdir(cache_folder):
        if entry.endswith(".model"):
            return os.path.join(cache_folder, entry)
    return ""


def path_to_search_pattern(path: str) -> str:
    basename = os.path.basename(path)
    if "." in basename:
        search_pattern = os.path.basename(path).split(".")[0]
    else:
        search_pattern = basename
    return search_pattern


def match_raw_and_gen(raw_path: str, cache_dict: dict):
    search_pattern = path_to_search_pattern(raw_path)
    for source_path, gen in cache_dict.items():
        if search_pattern in os.path.basename(source_path):
            return gen
    return None


def train_test_match(cache_dict: dict, train_raw_path: str, test_raw_path: str) -> dict:
    """
    Найти тестовые и обучающие генераторы (или пути) в кэше
    :param cache_dict: словарь вида {source_path: generator}
    :param train_raw_path: путь к исходному обучающему файлу
    :param test_raw_path: путь к исходному тестовому файлу
    :return: словарь вида {'train': generator/path, ['test': generator/path]}
             поля 'test' может не быть, еслт test_raw_path - пустая строка
             В теории результат может быть пустым словарем
    """
    res = {}
    train_gen = match_raw_and_gen(train_raw_path, cache_dict)
    if train_gen is not None:
        res["train"] = train_gen
    if test_raw_path:
        test_gen = match_raw_and_gen(test_raw_path, cache_dict)
        if test_gen is not None:
            res["test"] = test_gen
    return res


def find_cache_for_source(cache_list: list, raw_path: str) -> str:
    if raw_path == "":
        return ""
    search_pattern = path_to_search_pattern(raw_path)
    for cache_path in cache_list:
        if search_pattern in os.path.basename(cache_path):
            return cache_path
    return ""


def create_reading_vector_gen(path: str, chunk_size: int):
    if path.endswith(".csv"):
        # Поддержка старого текстового формата
        vec_gen = vector_from_csv_generator(path, chunk_size=chunk_size)
    else:
        # Новый бинарный формат (pickle)
        vec_gen = vector_from_pkl_generator(path, chunk_size=chunk_size)
    return vec_gen


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
        for ind in indices:
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


def main():
    # Получаем аргументы командной строки
    argparser = ArgumentParser()
    add_args(argparser)
    args = argparser.parse_args()
    exp_path = args.exp_path
    settings_path = os.path.join(exp_path, "settings.ini")
    # Создаем логгер
    logger = create_logger(os.path.join(exp_path, f"{timestamp()}.log"), "ecs.main")

    if not os.path.exists(settings_path):
        logger.info(f"No settings.ini file found in {exp_path}")
        exit(0)

    # Загружаем и проверяем конфиг
    config = ValidConfig()
    config.read(settings_path, encoding="cp1251")
    logger.info("Validating experiment settings...")
    config.validate_all()
    logger.info("=" * 20 + "Validation OK" + "=" * 20 + "\n")

    # Находим датасет
    training_file = config.get("TrainingData", "dataset")
    test_file = config.get("TrainingData", "test_file")
    dataset_folder = os.path.dirname(training_file)

    # Определяем язык
    language = config.get("Preprocessing", "language")
    if language == "auto":
        # Для распознавания языка грузим 10 первых строк
        language = recognize_language(training_file, encoding="cp1251", n_lines=10)
        # Потом мы будем копировать файл настроек в кэш
        # Поэтому поддерживаем его актуальным
        config.set("Preprocessing", "language", language)

    # Получаем параметры
    exp_title = config.get_primitive("Experiment", "experiment_title")
    if exp_title == "":
        exp_title = timestamp()
    remove_stopwords = config.get_primitive("Preprocessing", "remove_stopwords")
    normalization = config.get_primitive("Preprocessing", "normalization")
    chunk_size = config.get_primitive("Preprocessing", "batch_size")
    use_model = config.get("WordEmbedding", "use_model")
    w2v_exists = os.path.exists(use_model)
    vector_dim = config.getint("WordEmbedding", "vector_dim")
    pooling = config.get("WordEmbedding", "pooling")
    window = config.getint("WordEmbedding", "window")
    binary = config.get_primitive("Experiment", "binary")
    rubricators = config.get_as_list("Experiment", "rubricator")
    n_jobs = config.getint("Experiment", "threads")
    n_folds = config.getint("Experiment", "n_folds")
    # Заплатка
    # TODO: починить
    test_percent = 0
    if not test_file:
        test_percent = config.getint("TrainingData", "test_percent")
        test_percent = test_percent / 100

    # Готовим фильтры настроек для поиска кэша
    clear_metadata_filter = {
        "language": language,
        "remove_stopwords": remove_stopwords,
        "normalization": normalization
    }
    vector_metadata_filter = {
        **clear_metadata_filter,
    }
    for key in ["vector_dim", "window", "pooling"]:
        vector_metadata_filter[key] = config.get_primitive("WordEmbedding", key)
    # Создаем источники векторов согласно схеме
    total_timer = time()
    cached_vectors = find_cached_vectors(base_dir=dataset_folder,
                                         metadata_filter=vector_metadata_filter)
    train_test_vec_exist = False
    cached_w2v_path = ""
    vector_cache_path = ""
    train_vec_cache = ""
    test_vec_cache = ""
    vector_gens = {}
    # Проводим разведку
    if len(cached_vectors) > 0:
        vector_cache_folder, vector_cache_files = cached_vectors.popitem()
        train_vec_cache = find_cache_for_source(vector_cache_files, training_file)
        test_vec_cache = None
        if test_file != "":
            test_vec_cache = find_cache_for_source(vector_cache_files, test_file)
        train_test_vec_exist = train_vec_cache and test_vec_cache
        cached_w2v_path = find_cached_w2v(vector_cache_folder)
    if train_test_vec_exist and cached_w2v_path and use_model != "":
        logger.info("Cached vectors found")
        w2v_model, language = load_w2v(cached_w2v_path)
        config.set("Preprocessing", "language", language)
        vector_gens["train"] = create_reading_vector_gen(train_vec_cache, chunk_size)
        vector_gens["test"] = create_reading_vector_gen(test_vec_cache, chunk_size)
    else:
        # Либо нет готовых обучающих векторов,
        # либо в кэше нет модели W2V,
        # либо нужно создать их заново при помощи указанной модели
        # Получаем источники чистых текстов
        # Словарь вида {'train': генератор, ['test': генератор]}
        pp_gens = create_clear_generators(
            base_dir=dataset_folder,
            clear_filter=clear_metadata_filter,
            chunk_size=chunk_size,
            training_fpath=training_file,
            test_fpath=test_file,
            experiment_title=exp_title,
            pp_params=extract_pp_settings(config)
        )
        if w2v_exists:
            # Используем ее
            # И обновляем язык
            logger.info(f"Using Word2Vec model: {os.path.basename(use_model)}")
            w2v_model, language = load_w2v(use_model)
            config.set("Preprocessing", "language", language)
        else:
            # Создаем новую
            # Для совместимости преобразуем в список
            logger.info("Creating new Word2Vec model")
            w2v_model = create_w2v(pp_sources=list(pp_gens.values()),
                                   vector_dim=vector_dim,
                                   window_size=window)
            # Увы, генераторы - это одноразовые итераторы
            # Придется создать их заново
            # Должны гарантированно получиться
            # читающие,а не кэширующие
            pp_gens = create_clear_generators(
                base_dir=dataset_folder,
                clear_filter=clear_metadata_filter,
                chunk_size=chunk_size,
                training_fpath=training_file,
                test_fpath=test_file,
                experiment_title=exp_title,
                pp_params=extract_pp_settings(config)
            )
        vector_cache_path = generate_vector_cache_path(raw_path=training_file, exp_name=exp_title)
        try:
            train_vec_gen = caching_vector_generator(pp_source=pp_gens["train"],
                                                     w2v_file=w2v_model,
                                                     cache_path=vector_cache_path,
                                                     conv_type=pooling,
                                                     pp_metadata=clear_metadata_filter)
        except Exception as e:
            error_ps(logger, f"Error occurred during creation caching vector generator (training): {e}")
        else:
            vector_gens["train"] = train_vec_gen
        if test_file:
            vector_cache_path = generate_vector_cache_path(raw_path=test_file, exp_name=exp_title)
            test_vec_gen = caching_vector_generator(pp_source=pp_gens["test"],
                                                    w2v_file=w2v_model,
                                                    cache_path=vector_cache_path,
                                                    conv_type=pooling,
                                                    pp_metadata=clear_metadata_filter)
            vector_gens["test"] = test_vec_gen
    # Время хорошенько загрузить память
    # Собираем обучающие и тестировочные выборки
    training_gen = vector_gens["train"]
    logger.info("Collecting the training dataset in memory")
    training_df = aggregate_full_dataset(training_gen)
    test_df = None
    if test_file != "":
        test_gen = vector_gens["test"]
        logger.info("Collecting the test dataset in memory")
        test_df = aggregate_full_dataset(test_gen)

    # На этом этапе уже должны быть созданы папки кэша,
    # так как мы гарантированно прогнали все генераторы
    # Копируем файл настроек и W2V
    # По недоразумению мы не храним путь к чистому кэшу,
    # поэтому создаем его заново. TODO: Исправить
    clear_cache_folder = os.path.dirname(generate_clear_cache_path(training_file, exp_title))
    vector_cache_folder = os.path.dirname(vector_cache_path)
    with open(os.path.join(clear_cache_folder, "settings.ini"), "w") as clear_copy_file:
        config.write(clear_copy_file)
    with open(os.path.join(vector_cache_folder, "settings.ini"), "w") as vector_copy_file:
        config.write(vector_copy_file)
    w2v_fname = generate_w2v_fname(vector_dim=w2v_model.vector_size, language=language)
    w2v_cache_path = os.path.join(vector_cache_folder, w2v_fname)
    w2v_save_path = os.path.join(exp_path, w2v_fname)
    w2v_model.save(w2v_cache_path)
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
        if test_df is None:
            x_train, x_test, y_train, y_test = create_labeled_tt_split(full_df=training_df,
                                                                       test_percent=test_percent,
                                                                       rubricator=rubricator)
        else:
            x_train, y_train = df_to_labeled_dataset(full_df=training_df, rubricator=rubricator)
            x_test, y_test = df_to_labeled_dataset(full_df=test_df, rubricator=rubricator)
        min_training_rubr = config.getint(rubricator, "min_training_rubric", fallback=0)
        min_test_rubr = config.getint(rubricator, "min_validation_rubric", fallback=0)
        train_filter_res = {}
        test_filter_res = {}
        if min_training_rubr > 0:
            train_filter_res = inplace_rubric_filter(x_train, y_train, threshold=min_training_rubr)
            log_str = "Dropped rubrics from training dataset:\n" + \
                      "\n".join([f"{k} ({v} texts)" for k, v in train_filter_res.items()])
            logger.info(log_str)
        if min_test_rubr > 0:
            test_filter_res = inplace_rubric_filter(x_test, y_test, threshold=min_test_rubr)
            log_str = "Dropped rubrics from test dataset:\n" + \
                      "\n".join([f"{k} ({v} texts)" for k, v in test_filter_res.items()])
            logger.info(log_str)
        # Проверка на слишком строгие пороги
        for desc, ds in {"Training dataset": y_train, "Test dataset": y_test}.items():
            if len(ds) == 0:
                logger.error(desc + " is empty! All the texts were removed due to the threshold")
                exit(0)
        for model_name in model_names:
            hypers = config.get_hyperparameters(model_name)
            if model_name == "svm":
                hypers["probability"] = [True]
            try:
                model_type = load_class(model_import_mapping[model_name])
            except ImportError as ie:
                logger.warning(f"\n>>> Unable to import model {model_name}, it will be skipped.")
                logger.warning(f">>> ({ie})\n")
                continue
            logger.info(f"Fitting parameters for model {model_name} by {rubricator}")
            model_instance = model_type()
            timer = time()
            try:
                best_params = run_grid_search(model_instance=model_instance,
                                              hyperparameters=hypers,
                                              x_train=x_train,
                                              y_train=y_train,
                                              binary=binary,
                                              n_folds=n_folds,
                                              n_jobs=n_jobs)
            except ValueError as ve:
                logger.warning(f"\n>>> Detected incorrect hyperparameters ({ve}) for model '{model_name}'."
                               f"It will be skipped.")
                continue
            except OSError as ose:
                state_str = f"(model: {model_name}, rubricator: {rubricator})"
                error_ps(logger, f"OS has interrupted the grid search process: {ose} {state_str}")
                exit(1)
            else:
                try:
                    best_model = refit_model(model_instance=model_type(),
                                             best_params=best_params,
                                             x_train=x_train, y_train=y_train, binary=binary)
                except OSError as ose:
                    state_str = f"(model: {model_name}, rubricator: {rubricator})"
                    error_ps(logger, f"OS has interrupted the refitting process: {ose} {state_str}")
                    exit(1)
                else:
                    time_elapsed = int(time() - timer)

                    # Сохраняем модель
                    model_fname = create_model_fname(model_name=model_name, language=language,
                                                     rubricator=rubricator, pooling=pooling,
                                                     vector_dim=w2v_model.vector_size)
                    model_path = os.path.join(exp_path, model_fname)
                    # Пока эта информация не используется, но в будущем может пригодиться
                    model_metadata = {
                        **vector_metadata_filter,
                        **best_params
                    }
                    logger.info(f"Saving model to {model_path}")
                    save_model(model=best_model, path=model_path, metadata=model_metadata)

                    # Создаем и сохраняем отчеты
                    logger.info("Testing model and creating report")
                    excel_report = create_report(model=best_model, x_test=x_test, y_test=y_test)
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
                                                       language=language,
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
