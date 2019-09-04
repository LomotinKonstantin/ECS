import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
import os

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
    create_labeled_tt_split
from ECS.core.model_tools import \
    load_class, \
    run_grid_search, \
    refit_model, \
    create_report, \
    save_report


def append_to_fname(fname: str, append: str) -> str:
    components = fname.split(".")
    components[-2] += append
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
    cache_fname = append_to_fname(os.path.basename(raw_path), "_clear")
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
    cache_fname = append_to_fname(os.path.basename(raw_path), "_vectors")
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
        print(f"Произошла ошибка поиска обучающих данных для пути {raw_path}."
              f"Индексированные источники: {vector_generators.keys()}")
        exit(0)
    return res_gen


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
    rubricators = config.get_as_list("Experiment" ,"rubricator")
    n_jobs = config.get("Experiment", "threads")
    n_folds = config.get("Experiment", "n_folds")
    test_percent = config.getint("TrainingData", "test_percent") / 100
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

    # Начинаем создавать источники векторов согласно схеме
    cached_vectors = find_cached_vectors(base_dir=dataset_folder, metadata_filter=vector_metadata_filter)
    # Словарь вида {путь_к_исходному_файлу: генератор}
    vector_gens = {}
    if (len(cached_vectors) > 0) and not w2v_exists:
        # Найдены подходящие векторы и их можно использовать
        vector_folder, file_list = cached_vectors.popitem()
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
            w2v_model = use_model
        else:
            # Создаем новую
            # Для совместимости преобразуем в список
            w2v_model = create_w2v(pp_sources=list(pp_gens.values()),
                                   vector_dim=vector_dim,
                                   window_size=window)
            # Увы, генераторы - это одноразовые итераторы
            # Придется создать их заново
            # Должны быть гарантированно читающие,
            # а не кэширующие
            pp_gens = create_clear_generators(
                base_dir=dataset_folder,
                clear_filter=clear_metadata_filter,
                chunksize=chunk_size,
                training_fpath=training_file,
                test_fpath=test_file,
                experiment_title=exp_title,
                pp_params=extract_pp_settings(config)
            )
        # TODO: скопировать файл настроек в чистый кэш
        # TODO: скопировать файл настроек и модель W2V в кэш векторов
        # Создаем кэширующие генераторы векторов
        # Функция принимает как путь к файлу модели W2V,
        # так и саму модель
        for source_path, pp_gen in pp_gens:
            fake_source_path = os.path.join(dataset_folder, os.path.basename(source_path))
            cache_path = generate_vector_cache_path(raw_path=fake_source_path, exp_name=exp_title)
            vec_gen = caching_vector_generator(pp_source=pp_gen,
                                               w2v_file=w2v_model,
                                               cache_path=cache_path,
                                               conv_type=pooling,
                                               pp_metadata=clear_metadata_filter)
            vector_gens[source_path] = vec_gen

        # Время хорошенько загрузить память
        # Собираем обучающие и тестировочные выборки
        # У нас нет механизма сопоставления генераторов
        # и сырых файлов, поэтому делаем допущения и проводим поиск
        # TODO: Добавить механизм, который позволяет отслеживать источник данных
        training_gen = gen_source_lookup(vector_gens, training_file)
        training_df = aggregate_full_dataset(training_gen)
        test_df = None
        if test_file != "":
            test_gen = gen_source_lookup(vector_gens, test_file)
            test_df = aggregate_full_dataset(test_gen)

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
                hypers = config.get_as_dict(model_name)
                try:
                    model_type = load_class(model_import_mapping[model_name])
                except ImportError as ie:
                    print(f"Не удалось импортировать модель {model_name}, модель будет пропущена "
                          f"(Ошибка '{ie}')")
                    continue
                model_instance = model_type()
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
                report = create_report(model=best_model, x_test=x_test, y_test=y_test)
                save_report()

    # Создаем рабочего
    worker = Worker()
    worker.set_math(True)
    # Загружаем индекс и датасет
    dataset_folder = config.get("TrainingData", "dataset")
    index = Index(dataset_folder)
    # index.load()
    index.build()
    vec_list = index.vectors_list()
    clr_list = index.cleared_texts_list()
    actual_data_path = ""
    data_type = "raw"
    we_settings = config.get_as_dict("WordEmbedding")
    vector_dim = we_settings["vector_dim"]
    pooling = we_settings["pooling"]
    use_model = config.get("WordEmbedding", "use_model", fallback="")
    w2v_exists = exists(use_model)
    # Сразу определяем язык
    language = config.get("Preprocessing", "language")
    if language == "auto":
        # Для распознавания языка грузим 10 первых строк
        file = next(walk(dataset_folder))[2][0]
        language = recognize_language(join(dataset_folder, file),
                                      encoding="cp1251", n_lines=10)
    config.validate_normalization(language)
    print("=" * 20, "Validation OK", "=" * 20)
    print("Language:", language)
    if len(vec_list) != 0 and not w2v_exists:
        we_settings = config.get_as_dict("WordEmbedding")
        pp_settings = config.get_as_dict("Preprocessing")
        vec_found = None
        clear_found = None
        common_settings = {**we_settings, **pp_settings}
        for entry in vec_list:
            if dicts_equal(common_settings, entry, ignore_keys=["path", "ds_title"]):
                actual_data_path = entry["path"]
                data_type = "vectors"
                break
        # print(f"vec_found = {vec_found}")
        # print(f"clear_found = {clear_found}")

    if actual_data_path == "":
        if len(clr_list) != 0:
            pp_settings = config.get_as_dict("Preprocessing")
            for entry in clr_list:
                if dicts_equal(pp_settings, entry, ignore_keys=["path", "ds_title"]):
                    actual_data_path = entry["path"]
                    data_type = "clear"
                    break
    if actual_data_path == "":
        actual_data_path = dataset_folder
    # Список файлов в нужном каталоге датасета
    dataset_files = next(walk(actual_data_path))[2]
    # Удаляем файл конфига из списка
    if "settings.ini" in dataset_files:
        dataset_files.remove("settings.ini")
    test_file = config.get("TrainingData", "test_file", fallback="")
    #
    if test_file != "":
        test_lst = []
        if data_type == "vectors":
            test_lst = list(filter(lambda a: "test" in a
                                             and "single_theme" not in a
                                             and a[-4:] == ".csv", dataset_files))
        else:
            test_lst = list(filter(lambda a: a.startswith(test_file.split(".")[-2])
                                             and a[-4:] == ".csv", dataset_files))
        test_file = test_lst[0]
        dataset_files.remove(test_file)
        test_file = join(actual_data_path, test_file)

    if data_type == "vectors":
        train_lst = list(filter(lambda a: "test" not in a
                                          and "single_theme" not in a
                                          and a[-4:] == ".csv", dataset_files))
        train_file = join(actual_data_path, train_lst[0])
    else:
        train_file = join(actual_data_path, dataset_files[0])
    # Далее проходим весь пайплайн создания векторов
    # (при необходимости)
    title = config.get("Experiment", "experiment_title", fallback="")
    if title.strip() == "":
        title = datetime.today().strftime("%d-%b-%Y___%X")
        title = title.replace(":", "-")
    # if experiment_exists(title):
    #     print("Эксперимент с таким названием уже существует")
    #     exit(0)
    if data_type == "raw":
        # Предобработка
        preprocessor = Preprocessor()
        clear_folder = join(actual_data_path, title + "_clear")
        if not exists(clear_folder):
            mkdir(clear_folder)
        clear_train_file = join(clear_folder,
                                split(append_to_fname(train_file, "_clear"))[-1])
        config_params = config.get_as_dict("Preprocessing")
        columns = {}
        for i in ["id", "title", "text", "keywords", "subj", "ipv", "rgnti", "correct"]:
            columns[i] = config_params.pop(i)
        pp_args = {"columns": columns,
                   **config_params}
        preprocessor.preprocess_file(train_file,
                                     clear_train_file,
                                     remove_formulas=False,
                                     **pp_args)
        clear_test_file = ""
        if test_file != "":
            clear_test_file = join(clear_folder,
                                   split(append_to_fname(test_file, "_clear"))[-1])
            preprocessor.preprocess_file(test_file,
                                         clear_test_file,
                                         remove_formulas=False,
                                         **pp_args)
        copyfile(settings_path,
                 join(clear_folder, "settings.ini"))
        # Если надо, создаем новую модель в2в
        we_settings = config.get_as_dict("WordEmbedding")
        vector_dim = we_settings["vector_dim"]
        pooling = we_settings["pooling"]
        data_loaded = False
        if use_model == "" or not w2v_exists:
            if not w2v_exists:
                print("Model {} not found".format(use_model))
            print("Creating Word2Vec model")
            worker.set_lang(language)
            worker.set_res_folder(exp_path)
            if clear_test_file != "":
                worker.load_data(clear_train_file, test_path=clear_test_file)
            else:
                train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
                worker.load_data(clear_train_file, split_ratio=train_percent)
            data_loaded = True
            worker.create_w2v_model(vector_dim)
        else:
            # Ищем такую модель
            # Для совместимости после упрощения
            actual_w2v_path = use_model
            # for file in next(walk(join(exp_path, use_model)))[2]:
            #     if file.split(".")[-1] == "model":
            #         actual_w2v_path = join(exp_path, "reports", use_model, file)
            # if actual_data_path == "":
            #     print("Что-то пошло не так. Эта надпись не должна была появиться")
            #     exit(1)
            worker.load_w2v(actual_w2v_path)
        # Векторизируем
        vector_folder = join(actual_data_path, title + "_vectors")
        if not data_loaded:
            if clear_test_file != "":
                worker.load_data(clear_train_file, test_path=clear_test_file)
            else:
                train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
                worker.load_data(clear_train_file, split_ratio=train_percent)
        worker.set_res_folder(vector_folder)
        worker.data_cleaning()
        worker.set_conv_type(pooling)
        print("Creating vectors")
        worker.create_w2v_vectors()
        copyfile(settings_path,
                 join(vector_folder, "settings.ini"))
        if use_model == "" or not w2v_exists:
            model_file = list(filter(lambda a: a.split(".")[-1] == "model",
                                     next(walk(exp_path))[2]))[0]
            copyfile(join(exp_path, model_file),
                     join(vector_folder, split(model_file)[-1]))
        else:
            copyfile(use_model, join(vector_folder, split(use_model)[-1]))
    elif data_type == "clear":
        print("Cached preprocessed dataset found")
        # use_model = config.get("WordEmbedding", "use_model", fallback="")
        data_loaded = False
        if use_model == "" or not w2v_exists:
            print("Creating Word2Vec model")
            worker.set_lang(language)
            worker.set_res_folder(exp_path)
            if test_file != "":
                worker.load_data(train_file, test_path=test_file)
            else:
                train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
                worker.load_data(train_file, split_ratio=train_percent)
            data_loaded = True
            worker.create_w2v_model(vector_dim)
        else:
            # Ищем такую модель
            actual_w2v_path = use_model
            # for file in next(walk(join(exp_path, "reports", use_model)))[2]:
            #     if file.split(".")[-1] == "model":
            #         actual_w2v_path = join(exp_path, "reports", use_model, file)
            # if actual_data_path == "":
            #     print("Что-то пошло не так. Эта надпись не должна была появиться")
            #     exit(1)
            worker.load_w2v(actual_w2v_path)
        # Векторизируем
        vector_folder = join(dataset_folder, title + "_vectors")
        if not data_loaded:
            if test_file != "":
                worker.load_data(train_file, test_path=test_file)
            else:
                train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
                worker.load_data(train_file, split_ratio=train_percent)
        worker.set_res_folder(vector_folder)
        worker.data_cleaning()
        worker.set_conv_type(pooling)
        print("Создание векторов")
        worker.create_w2v_vectors()
        copyfile(settings_path,
                 join(vector_folder, "settings.ini"))
        if use_model == "" or not w2v_exists:
            model_file = list(filter(lambda a: a.split(".")[-1] == "model",
                                     next(walk(exp_path))[2]))[0]
            copyfile(join(exp_path, model_file),
                     join(vector_folder, split(model_file)[-1]))
        else:
            copyfile(use_model, join(vector_folder, split(use_model)[-1]))
    else:
        print("Cached vectors found")
        if test_file == "":
            train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
            worker.load_data(train_file, split_ratio=train_percent)
        else:
            worker.load_data(train_file, test_path=test_file)
        w2v_copy_src = ""
        if use_model == "" or not w2v_exists:
            if not w2v_exists:
                print("Model {} not found".format(use_model))
            for file in listdir(dirname(train_file)):
                if file.endswith(".model"):
                    w2v_copy_src = join(dirname(train_file), file)
        else:
            w2v_copy_src = use_model
        if exists(w2v_copy_src):
            copyfile(w2v_copy_src, join(exp_path, split(w2v_copy_src)[-1]))
        else:
            print("Error: specified model {} does not exist".format(w2v_copy_src))
    # Учим классификатор
    if not worker.set_res_folder(exp_path):
        exit(0)
    # if (use_model == "" or not w2v_exists) and data_type == "vectors":
    #     model_file = list(filter(lambda a: a.split(".")[-1] == "model",
    #                              next(walk(actual_data_path))[2]))[0]
    #     copyfile(join(actual_data_path, model_file),
    #              join(exp_path, split(model_file)[-1]))
    worker.set_lang(language)
    worker.set_conv_type(pooling)
    binary = config.get_as_dict("Experiment")["binary"]
    n_folds = config.getint("Experiment", "n_folds")
    threads = config.getint("Experiment", "threads")
    for rubr in config.get_as_list("Experiment", "rubricator"):
        worker.set_rubr_id(rubr)
        for model in config.get_as_list("Classification", "models"):
            print("Fitting parameters for model {} by {}".format(model, rubr))
            ModelType = config.get_model_type(model)
            instance = ModelType()
            hypers = config.get_hyperparameters(model)
            #
            if model == "svm":
                hypers["probability"] = [True]
            #
            worker.data_train = worker.data_train.rename(columns=str)
            try:
                worker.data_test = worker.data_test.rename(columns=str)
            except Exception:
                pass
            worker.search_for_clf(model=instance,
                                  parameters=hypers,
                                  jobs=threads,
                                  oneVsAll=binary,
                                  skf_folds=n_folds,
                                  description=model)
