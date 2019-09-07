import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("..")
from argparse import ArgumentParser
import os
from time import time
from ECS.interface.valid_config import ValidConfig
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


def create_clear_generators(base_dir: str,
                            clear_filter: dict,
                            chunksize: int,
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
    :param chunksize: размер чанка
    :param training_fpath: путь к сырому обучающему файлу
    :param test_fpath: путь к сырому тест-файлу (может быть равен '')
    :param experiment_title: название эксперимента
    :param pp_params: словарь с полными настройками препроцессора
    :return: словарь вида {путь_к_источнику: генератор}
    """
    pp_sources = {}
    cached_clear = find_cached_clear(base_dir=base_dir, metadata_filter=clear_filter)
    if len(cached_clear) > 0:
        # Если нашли больше, чем одну папку с чистыми текстами,
        # берем случайную
        _, clear_file_list = cached_clear.popitem()
        for file_path in clear_file_list:
            gen = pp_from_csv_generator(file_path, chunksize)
            pp_sources[file_path] = gen
    else:
        # Чистых текстов нет
        training_cache_file = generate_clear_cache_path(training_fpath, exp_name=experiment_title)
        pp_sources[training_fpath] = caching_pp_generator(raw_file=training_fpath,
                                                          chunk_size=chunk_size,
                                                          cache_path=training_cache_file,
                                                          **pp_params)
        if test_fpath != "":
            test_cache_file = generate_clear_cache_path(test_fpath, exp_name=experiment_title)
            pp_sources[test_fpath] = caching_pp_generator(raw_file=test_fpath,
                                                          chunk_size=chunk_size,
                                                          cache_path=test_cache_file,
                                                          **pp_params)
    return pp_sources


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


def gen_source_lookup(vector_generators: dict, raw_path: str):
    """
    Найти генератор, который соответствует указанному сырому файлу.
    Может завершить исполнение с сообщением об ошибке
    :param vector_generators: словарь {source_path: generator}
    :param raw_path: путь к сырому файлу
    :return: объект генератора
    """
    res_gen = None
    search_pattern = os.path.basename(raw_path).split(".")[0]
    for path in vector_generators:
        if search_pattern in os.path.basename(path):
            res_gen = vector_generators[path]
            break
    if res_gen is None:
        linesep = "\n"
        print(f"Generator lookup error for {raw_path}\n"
              f"Indexed sources: {linesep.join(vector_generators.keys())}")
        exit(0)
    return res_gen


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


if __name__ == '__main__':
    # Получаем аргументы командной строки
    argparser = ArgumentParser()
    add_args(argparser)
    args = argparser.parse_args()
    exp_path = args.exp_path
    settings_path = os.path.join(exp_path, "settings.ini")
    if not os.path.exists(settings_path):
        print("No settings.ini file found in ", exp_path)
        exit(0)
    # Загружаем и проверяем конфиг
    config = ValidConfig()
    config.read(settings_path, encoding="cp1251")
    print("Validating experiment settings...")
    config.validate_all()
    print("=" * 20, "Validation OK", "=" * 20)
    # Находим датасет
    training_file = config.get("TrainingData", "dataset")
    test_file = config.get("TrainingData", "test_file")
    dataset_folder = os.path.dirname(training_file)
    # Определяем язык
    language = config.get("Preprocessing", "language")
    if language == "auto":
        # Для распознавания языка грузим 10 первых строк
        language = recognize_language(training_file, encoding="cp1251", n_lines=10)
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
    test_percent = config.getint("TrainingData", "test_percent") / 100
    # Объявляем путь к кэшу и модель W2V в общей области видимости,
    # чтобы использовать потом
    vector_cache_path = ""
    w2v_model = None
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
    total_timer = time()
    # Создаем источники векторов согласно схеме
    cached_vectors = find_cached_vectors(base_dir=dataset_folder, metadata_filter=vector_metadata_filter)
    # Словарь вида {путь_к_исходному_файлу: генератор}
    vector_gens = {}
    vector_folder = ""
    file_list = []
    cached_w2v_exists = False
    if len(cached_vectors) > 0:
        vector_folder, file_list = cached_vectors.popitem()
        cached_w2v_path = find_cached_w2v(vector_folder)
        cached_w2v_exists = os.path.exists(cached_w2v_path)
        w2v_model, language = load_w2v(cached_w2v_path)
        config.set("Preprocessing", "language", language)
    if cached_w2v_exists and not w2v_exists:
        print("Cached vectors found")
        vector_cache_path = file_list[0]
        for vec_file_path in file_list:
            if vec_file_path.endswith(".csv"):
                # Поддержка старого текстового формата
                vec_gen = vector_from_csv_generator(vec_file_path, chunk_size=chunk_size)
            else:
                # Новый бинарный формат (pickle)
                vec_gen = vector_from_pkl_generator(vec_file_path, chunk_size=chunk_size)
            vector_gens[vec_file_path] = vec_gen
    else:
        # Либо нет готовых векторов,
        # либо нужно создать их заново при помощи указанной модели
        # Получаем источники чистых текстов
        # Словарь вида {путь_к_исходному_файлу: генератор}
        pp_gens = create_clear_generators(
            base_dir=dataset_folder,
            clear_filter=clear_metadata_filter,
            chunksize=chunk_size,
            training_fpath=training_file,
            test_fpath=test_file,
            experiment_title=exp_title,
            pp_params=extract_pp_settings(config)
        )
        if w2v_exists:
            # Используем ее
            # И обновляем язык
            print(f"Using Word2Vec model: {os.path.basename(use_model)}")
            w2v_model, language = load_w2v(use_model)
            config.set("Preprocessing", "language", language)
        else:
            # Создаем новую
            # Для совместимости преобразуем в список
            print("Creating new Word2Vec model")
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
                chunksize=chunk_size,
                training_fpath=training_file,
                test_fpath=test_file,
                experiment_title=exp_title,
                pp_params=extract_pp_settings(config)
            )
        # Создаем кэширующие генераторы векторов
        # Функция принимает как путь к файлу модели W2V,
        # так и саму модель.
        for source_path, pp_gen in pp_gens.items():
            fake_source_path = os.path.join(dataset_folder, os.path.basename(source_path))
            vector_cache_path = generate_vector_cache_path(raw_path=fake_source_path, exp_name=exp_title)
            vec_gen = caching_vector_generator(pp_source=pp_gen,
                                               w2v_file=w2v_model,
                                               cache_path=vector_cache_path,
                                               conv_type=pooling,
                                               pp_metadata=clear_metadata_filter)
            vector_gens[source_path] = vec_gen
    # Время хорошенько загрузить память
    # Собираем обучающие и тестировочные выборки
    # У нас нет механизма сопоставления генераторов
    # и сырых файлов, поэтому делаем допущения и проводим поиск
    # TODO: Добавить механизм, который позволяет отслеживать источник данных
    print("Creating dataset")
    training_gen = gen_source_lookup(vector_gens, training_file)
    training_df = aggregate_full_dataset(training_gen)
    test_df = None
    if test_file != "":
        test_gen = gen_source_lookup(vector_gens, test_file)
        test_df = aggregate_full_dataset(test_gen)

    # На этом этапе уже должны быть созданы папки кэша,
    # так как мы гарантированно прогнали все генераторы
    # Копируем файл настроек и W2V
    # По недоразумению мы не храним путь к чистому кэшу,
    # поэтому создаем заново. TODO: Исправить
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
        for model_name in model_names:
            hypers = config.get_hyperparameters(model_name)
            if model_name == "svm":
                hypers["probability"] = [True]
            try:
                model_type = load_class(model_import_mapping[model_name])
            except ImportError as ie:
                print(f"Не удалось импортировать модель {model_name}, модель будет пропущена "
                      f"(Ошибка '{ie}')")
                continue
            print(f"Fitting parameters for model {model_name} by {rubricator}")
            model_instance = model_type()
            timer = time()
            best_params = run_grid_search(model_instance=model_instance,
                                          hyperparameters=hypers,
                                          x_train=x_train,
                                          y_train=y_train,
                                          binary=binary,
                                          n_folds=n_folds,
                                          n_jobs=n_jobs)
            best_model = refit_model(model_instance=model_type(),
                                     best_params=best_params,
                                     x_train=x_train, y_train=y_train, binary=binary)
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
            save_model(model=best_model, path=model_path, metadata=model_metadata)

            # Создаем и сохраняем отчеты
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
            save_excel_report(path=excel_path, report=excel_report, rubricator=rubricator)
            save_txt_report(path=txt_path, report=text_report)
            # Печатаем мини-отчет
            mini_report = short_report(excel_report, time_elapsed)
            print(mini_report)
    print("Done")
    print(f"Total time elapsed: {seconds_to_duration(int(time() - total_timer))}")
