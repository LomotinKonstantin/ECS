import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
import os

from ECS.interface.valid_config import ValidConfig
from ECS.interface.index import Index
from ECS.preprocessor.Preprocessor2 import Preprocessor


def append_to_fname(fname: str, append: str) -> str:
    components = fname.split(".")
    components[-2] += append
    return ".".join(components)


# def experiment_exists(title: str) -> bool:
#     return title in next(walk(join(dirname(__file__), "reports")))[1]


def recognize_language(filename: str, encoding="utf8", n_lines=10) -> str:
    chunk = []
    with open(filename, encoding=encoding) as file:
        for n, line in enumerate(file):
            chunk.append(line)
            if line == n_lines - 1:
                break
    text_sample = " ".join(chunk)
    p = Preprocessor()
    return p.recognize_language(text_sample)


def add_args(argparser: ArgumentParser):
    argparser.add_argument("exp_path",
                           type=str,
                           help="Full path to the folder with settings.ini file",)


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
