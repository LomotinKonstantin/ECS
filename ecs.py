import warnings

warnings.filterwarnings("ignore")

from interface.valid_config import ValidConfig
from interface.index import Index
from preprocessor.Preprocessor2 import Preprocessor
from os.path import dirname, join, exists, split
from os import walk, mkdir
from core.Worker import Worker
from datetime import datetime
from shutil import copyfile

'''
Воркфлоу следующий:
1. Обработать конфиг:
    * Загрузить
    * Провалидировать
2. Загрузить индекс
    Индекс имеет следующую структуру:
    Каждая запись дополнительно включает поле path - путь к папке с текстами/векторами
    {
        <dataset_name1>: {
            clear: [
                {<preprocessing_section_of_copied_settings1>},
                {<preprocessing_section_of_copied_settings2>},
                {<preprocessing_section_of_copied_settings3>}
            ],
            vectors: [
                {<word_embedding_section_of_copied_settings1>},
                {<word_embedding_section_of_copied_settings3>},
            ]
        },
        
        <dataset_name2>: {
            clear: [
                {<preprocessing_section_of_copied_settings4>}
            ],
            vectors: []
        },
        
        <dataset_name3>: {
            clear: [],
            vectors: []
        }
    }
3. Согласно общей схеме получить актуальный датасет
4. Запустить Worker
'''


def dicts_equal(d1: dict, d2: dict, ignore_keys=()) -> bool:
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())
    for i in ignore_keys:
        keys1.discard(i)
        keys2.discard(i)
    if keys1 != keys2:
        return False
    for key in keys1:
        if d1[key] != d2[key]:
            return False
    return True


def append_to_fname(fname: str, append: str) -> str:
    components = fname.split(".")
    components[-2] += append
    return ".".join(components)


def experiment_exists(title: str) -> bool:
    return title in next(walk(join(dirname(__file__), "reports")))[1]


def recognize_language(file: str, encoding="utf8", n_lines=10) -> str:
    with open(file, encoding=encoding) as file:
        text_sample = " ".join([next(file) for _ in range(n_lines)])
    p = Preprocessor()
    return p.recognize_language(text_sample)


if __name__ == '__main__':
    # Загружаем и проверяем конфиг
    config = ValidConfig()
    config.read(join(dirname(__file__), "settings.ini"), encoding="cp1251")
    print("Validating experiment settings...")
    config.validate_all()
    # Создаем рабочего
    worker = Worker()
    # Загружаем индекс и датасет
    index = Index()
    index.load()
    index.rebuild()
    data_folder = join(dirname(__file__), "datasets")
    dataset_title = config.get("TrainingData", "dataset")
    vec_list = index.vectors_list(dataset_title)
    actual_data_path = ""
    data_type = "raw"
    we_settings = config.get_as_dict("WordEmbedding")
    vector_dim = we_settings["vector_dim"]
    pooling = we_settings["pooling"]
    # Сразу определяем язык
    language = config.get("Preprocessing", "language")
    if language == "auto":
        # Для распознавания языка грузим 10 первых строк
        file = next(walk(join(data_folder, dataset_title)))[2][0]
        language = recognize_language(join(data_folder, dataset_title, file), encoding="cp1251")
    print("Language:", language)
    if len(vec_list) != 0:
        we_settings = config.get_as_dict("WordEmbedding")
        for entry in vec_list:
            if dicts_equal(we_settings, entry, ignore_keys=["path", "ds_title"]):
                actual_data_path = entry["path"]
                data_type = "vectors"
                break
    if actual_data_path == "":
        clr_list = index.cleared_texts_list(dataset_title)
        if len(clr_list) != 0:
            pp_settings = config.get_as_dict("Preprocessing")
            for entry in clr_list:
                if dicts_equal(pp_settings, entry, ignore_keys=["path", "ds_title"]):
                    actual_data_path = entry["path"]
                    data_type = "clear"
                    break
    if actual_data_path == "":
        actual_data_path = join(data_folder, dataset_title)
    # список файлов в нужном каталоге датасета
    dataset_files = next(walk(actual_data_path))[2]
    # удаляем файл конфига из списка
    if "settings.ini" in dataset_files:
        dataset_files.remove("settings.ini")
    test_file = config.get("TrainingData", "test_file", fallback="")
    #
    if test_file != "":
        test_lst = []
        if data_type == "vectors":
            test_lst = list(filter(lambda a: "test" in a and "single_theme" not in a, dataset_files))
        else:
            test_lst = list(filter(lambda a: a.startswith(test_file.split(".")[-2]), dataset_files))
        test_file = test_lst[0]
        dataset_files.remove(test_file)
        test_file = join(actual_data_path, test_file)

    if data_type == "vectors":
        train_lst = list(filter(lambda a: "test" not in a and "single_theme" not in a, dataset_files))
        train_file = join(actual_data_path, train_lst[0])
    else:
        train_file = join(actual_data_path, dataset_files[0])
    # Далее проходим весь пайплайн создания векторов
    # (при необходимости)
    title = config.get("Experiment", "experiment_title", fallback="")
    if title.strip() == "":
        title = datetime.today().strftime("%d-%b-%Y___%X")
        title = title.replace(":", "-")
    if experiment_exists(title):
        print("Эксперимент с таким названием уже существует")
        exit(0)
    result_path = join(dirname(__file__), "reports", title)
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
        copyfile(join(dirname(__file__), "settings.ini"),
                 join(clear_folder, "settings.ini"))
        # Если надо, создаем новую модель в2в
        use_model = config.get("WordEmbedding", "use_model", fallback="")
        we_settings = config.get_as_dict("WordEmbedding")
        vector_dim = we_settings["vector_dim"]
        pooling = we_settings["pooling"]
        w2v_model = None
        data_loaded = False
        if use_model == "":
            print("Creating Word2Vec model")
            worker.set_lang(language)
            worker.set_res_folder(result_path)
            if clear_test_file != "":
                worker.load_data(clear_train_file, test_path=clear_test_file)
            else:
                train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
                worker.load_data(clear_train_file, split_ratio=train_percent)
            data_loaded = True
            worker.create_w2v_model(vector_dim)
        else:
            # Ищем такую модель
            actual_w2v_path = ""
            for file in next(walk(join(dirname(__file__), "reports", use_model)))[2]:
                if file.split(".")[-1] == "model":
                    actual_w2v_path = join(dirname(__file__), "reports", use_model, file)
            if actual_data_path == "":
                print("Что-то пошло не так. Эта надпись не должна была появиться")
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
        copyfile(join(dirname(__file__), "settings.ini"),
                 join(vector_folder, "settings.ini"))
    elif data_type == "clear":
        print("Cached preprocessed dataset found")
        use_model = config.get("WordEmbedding", "use_model", fallback="")
        w2v_model = None
        data_loaded = False
        if use_model == "":
            print("Creating Word2Vec model")
            worker.set_lang(language)
            worker.set_res_folder(result_path)
            if test_file != "":
                worker.load_data(train_file, test_path=test_file)
            else:
                train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
                worker.load_data(train_file, split_ratio=train_percent)
            data_loaded = True
            worker.create_w2v_model(vector_dim)
        else:
            # Ищем такую модель
            actual_w2v_path = ""
            for file in next(walk(join(dirname(__file__), "reports", use_model)))[2]:
                if file.split(".")[-1] == "model":
                    actual_w2v_path = join(dirname(__file__), "reports", use_model, file)
            if actual_data_path == "":
                print("Что-то пошло не так. Эта надпись не должна была появиться")
            worker.load_w2v(actual_w2v_path)
        # Векторизируем
        vector_folder = join(data_folder, dataset_title, title + "_vectors")
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
        copyfile(join(dirname(__file__), "settings.ini"),
                 join(vector_folder, "settings.ini"))
    else:
        print("Cached vectors found")
        if test_file == "":
            train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
            worker.load_data(train_file, split_ratio=train_percent)
        else:
            worker.load_data(train_file, test_path=test_file)

    # Учим классификатор
    if not worker.set_res_folder(result_path):
        exit(0)
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
            worker.search_for_clf(model=instance,
                                  parameters=hypers,
                                  jobs=threads,
                                  OneVsAll=binary,
                                  skf_folds=n_folds)
    copyfile(join(dirname(__file__), "settings.ini"), join(result_path, "settings.ini"))
