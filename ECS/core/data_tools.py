import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import datetime

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from ECS.preprocessor.Preprocessor2 import Preprocessor
from ECS.interface.valid_config import ValidConfig
from ECS.interface.logging_tools import get_logger, info_ps, error_ps


N_LOG_MSGS = 10
N_CHUNKS_INTERVAL = 50


def is_dict_subset(dict_to_check: dict, big_dict: dict) -> bool:
    for key, value in dict_to_check:
        if value != big_dict[key]:
            return False
    return True


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


def pp_from_raw_generator(raw_file: str, chunk_size: int, **pp_settings) -> pd.DataFrame:
    """
    Генератор предобработанных текстов
    :param raw_file: путь к файлу исходных текстов
    :param chunk_size: размер чанка для предобработки
    :param pp_settings: настройки препроцессора
    """
    pp = Preprocessor()
    for n, chunk in enumerate(pd.read_csv(raw_file, encoding="cp1251", quoting=3, sep="\t", chunksize=chunk_size)):
        try:
            pp_chunk = pp.preprocess_dataframe(df=chunk, **pp_settings)
        except Exception as e:
            logger = get_logger("ecs.data_tools.pp_from_raw_generator")
            error_ps(logger, f"Error occurred during preprocessing chunk {n} of file {raw_file}: {e}")
            exit(1)
        else:
            yield pp_chunk


def pp_from_csv_generator(clear_csv_path: str, chunk_size: int) -> pd.DataFrame:
    for chunk in pd.read_csv(clear_csv_path, sep="\t", encoding="cp1251", chunksize=chunk_size):
        yield chunk


def caching_pp_generator(raw_file: str,
                         cache_path: str,
                         chunk_size: int,
                         **pp_settings) -> pd.DataFrame:
    """
    Кэширующий генератор предобработанных данных.
    Обеспечивает транзакционность кэширования.
    :param raw_file: путь к файлу исходных текстов
    :param cache_path: путь к файлу кэша
    :param chunk_size: размер чанка
    :param pp_settings: настройки препроцессора
    :return: чанк
    """
    cache_invalid_path = f"{cache_path}.invalid"
    with open(cache_invalid_path, "w") as cache_file:
        cache_header = True
        for pp_chunk in pp_from_raw_generator(raw_file=raw_file, chunk_size=chunk_size, **pp_settings):
            pp_chunk.to_csv(cache_file, encoding="utf8", sep="\t", index=False, header=cache_header)
            cache_header = False
            yield pp_chunk
    if os.path.exists(cache_path):
        os.remove(cache_path)
    os.rename(cache_invalid_path, cache_path)


def create_w2v(pp_sources: list,
               vector_dim: int,
               window_size: int) -> Word2Vec:
    """
    Создать модель Word2Vec для предобработанных текстов.
    :param window_size: размер окна контекста
    :param vector_dim: размерность векторной модели
    :param pp_sources: список инстансов генератора
                       предобработанных данных (pp_generator или caching_pp_generator)
    :returns обученная модель Word2Vec
    """
    logger = get_logger("ecs.data_tools.create_w2v")
    w2v = Word2Vec(size=vector_dim, min_count=3, workers=3, window=window_size)
    init = False
    for n_source, pp_source in enumerate(pp_sources, start=1):
        info_ps(logger, f"Training W2V on source {n_source}/{len(pp_sources)}")
        for n_chunk, pp_chunk in enumerate(pp_source, start=1):
            try:
                if n_chunk % N_CHUNKS_INTERVAL == 0:
                    info_ps(logger, f"\nChunk {n_chunk}")
                sentence = [text.split() for text in pp_chunk["text"].values]
                w2v.build_vocab(sentence, update=init)
                # TODO: вынести количество эпох в параметры
                w2v.train(sentence, epochs=20, total_examples=len(sentence))
                init = True
            except Exception as e:
                error_ps(logger, f"An error occurred during training W2V: {e}. Source {n_source}, chunk {n_chunk} "
                                 f"(~{(n_chunk - 1) * len(pp_chunk) + 1}{n_chunk * len(pp_chunk)} "
                                 f"lines in clear file)")
                exit(1)
    return w2v


def timestamp() -> str:
    now = datetime.datetime.today()
    date = f"{now.hour}-{now.minute}-{now.second}_{now.day}_{now.month}_{str(now.year)[2:]}"
    return date


def generate_w2v_fname(vector_dim: int,
                       language: str,
                       version=1,
                       additional_info=None) -> str:
    """
    Сгенерировать имя файла модели, совместимое с ATC.
    Для правильной работы ATC получает информацию о модели из имени файла.
    :param vector_dim: размерность векторной модели
    :param language: язык словаря
    :param version: [опционально] версия
    :param additional_info: [опционально] допольнительная короткая строка с информацией
    :return: сгенерированное имя файла
    """
    now = datetime.datetime.today()
    date = f"{now.day}_{now.month}_{str(now.year)[2:]}"
    name = f"w2v_model_{vector_dim}_{language}"
    if additional_info is None:
        name = f"{name}_v{version}_{date}.model"
    else:
        name = f"{name}_{additional_info}_v{version}_{date}.model"
    return name


def load_w2v(w2v_path: str) -> tuple:
    """
    Зарузить модель Word2Vec и получить язык из имени файла
    :type w2v_path: str
    :param w2v_path путь к файлу модели
    """
    w2v_model = Word2Vec.load(w2v_path)
    lang = os.path.basename(w2v_path).split("_")[3]
    return w2v_model, lang


def vector_from_csv_generator(matrix_path: str, chunk_size: int):
    for chunk in pd.read_csv(matrix_path, index_col=0, sep="\t", chunksize=chunk_size, encoding="cp1251"):
        yield chunk


def apply_pooling(matrix: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "none":
        return matrix
    elif pooling == "sum":
        return matrix.sum(axis=0)
    elif pooling == "mean":
        return matrix.mean(axis=0)
    elif pooling == "max":
        return matrix.max(axis=0)


def vectorize_text(text: str, w2v_model: Word2Vec, conv_type: str) -> np.ndarray:
    """
    Превратить текст в тензор признаков
    :param text: строка со словами, разделенными пробелом
    :param w2v_model: модель Word2Vec
    :param conv_type: если 'sum', 'mean' или 'max' - применяет выбранный пулинг к матрице по столбцам
                      если 'none' - возвращает матрицу
    :return: матрица или вектор признаков
    """
    tokens = text.split()
    text_matrix = []
    for word in tokens:
        try:
            word_vector = w2v_model[word]
        except KeyError:
            word_vector = np.zeros(w2v_model.vector_size)
        text_matrix.append(word_vector)
    text_matrix = np.vstack(text_matrix)
    if conv_type in ["none", "sum", "max", "mean"]:
        return apply_pooling(text_matrix, conv_type)
    else:
        get_logger("ecs.data_tools.vectorize_text").error(f"Conv type '{conv_type}' is not supported")
        exit(1)


def vectorize_pp_chunk(pp_chunk: pd.DataFrame, w2v_model: Word2Vec, conv_type: str) -> pd.DataFrame:
    """
    Добавляет столбец features
    :param pp_chunk: датафрейм с колонкой text, содержащей предобработанные тексты
    :param w2v_model: модель Word2Vec
    :param conv_type: тип свертки: sum, mean или max (или none)
    :return: исходный датафрейм с добавленной колонкой features
    """
    matrices = []
    for text in pp_chunk["text"]:
        matrices.append(vectorize_text(text, w2v_model, conv_type))
    pp_chunk["features"] = matrices
    return pp_chunk


def caching_matrix_generator(pp_source,
                             w2v_file,
                             cache_path: str,
                             pp_metadata: dict) -> pd.DataFrame:
    """
    Кэширующий генератор векторов
    :param pp_metadata: словарь с настройками препроцессора.
            Есть смысл сохранять только поля remove_stopwords и normalization,
    :param pp_source: генератор предобработанных чанков
    :param w2v_file: путь к файлу модели Word2Vec или сам объект модели
    :param cache_path: путь к файлу кэша
    :return: чанк в формате pd.Dataframe
    """
    if isinstance(w2v_file, str):
        model, _ = load_w2v(w2v_file)    # type: Word2Vec
    else:
        model = w2v_file
    metadata = {
        "vector_dim": model.vector_size,
        # Удалено:
        # language - уже есть в pp_metadata
        # pooling - пулинг теперь не применяется при кэшировании
        "window": model.window,
        **pp_metadata
    }
    cache_invalid_path = f"{cache_path}.invalid"
    with open(cache_invalid_path, "bw") as cache_file:
        pickle.dump(metadata, cache_file)
        for pp_chunk in pp_source:
            # Кэшируем только матрицы
            matrix_chunk = vectorize_pp_chunk(pp_chunk, w2v_model=model, conv_type="none")
            for row in matrix_chunk.index:
                entry = (
                    matrix_chunk.loc[row, "features"],
                    matrix_chunk.loc[row, "subj"],
                    matrix_chunk.loc[row, "ipv"],
                    matrix_chunk.loc[row, "rgnti"]
                )
                pickle.dump(entry, cache_file)
            # Пусть вызывающий сам применяет пулинг
            yield matrix_chunk
    if os.path.exists(cache_path):
        os.remove(cache_path)
    os.rename(cache_invalid_path, cache_path)


def read_pkl_metadata(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as pkl_file:
        metadata = pickle.load(pkl_file)
    return metadata


def matrix_from_pkl_generator(pkl_path: str, chunk_size: int) -> pd.DataFrame:
    pkl_file = open(pkl_path, "rb")
    # Первая строка - метадата
    # Пропускаем ее
    pickle.load(pkl_file)
    chunk = {
        "features": [], "subj": [],
        "ipv": [], "rgnti": []
    }
    cntr = 0
    while True:
        try:
            entry = pickle.load(pkl_file)
            chunk["features"].append(entry[0])
            chunk["subj"].append(entry[1])
            chunk["ipv"].append(entry[2])
            chunk["rgnti"].append(entry[3])
            cntr += 1
            if cntr == chunk_size:
                cntr = 0
                yield pd.DataFrame(chunk)
                chunk = {
                    "features": [], "subj": [],
                    "ipv": [], "rgnti": []
                }
        except EOFError:
            break
    pkl_file.close()
    if len(chunk["features"]) > 0:
        yield pd.DataFrame(chunk)


def find_cached_matrices(base_dir: str, metadata_filter: dict) -> dict:
    """
    Найти закэшированные векторы, отвечающие условиям
    :param base_dir: директория датасета для начала поиска
    :param metadata_filter: словарь с настройками препроцессора, размерностью окна w2v и вектора и языком
    :return: словарь путей к найденным файлам вида
             {"абс_путь_к_папке": ["абс_путь_к_файлу1", "абс_путь_к_файлу2"]}
    """
    vector_files = {}
    for entry in os.scandir(base_dir):  # type: os.DirEntry
        if not (entry.is_dir() and entry.name.endswith("_matrices")):
            continue
        for file in os.scandir(entry.path):  # type: os.DirEntry
            if file.name.endswith(".pkl"):
                file_metadata = read_pkl_metadata(file.path)
                if dicts_equal(file_metadata, metadata_filter):
                    vector_files.setdefault(entry.path, [])
                    vector_files[entry.path].append(file.path)
    return vector_files


def find_cached_clear(base_dir: str, metadata_filter: dict) -> dict:
    clear_files = {}
    for entry in os.scandir(base_dir):  # type: os.DirEntry
        if not (entry.is_dir() and entry.name.endswith("_clear")):
            continue
        for file in os.scandir(entry.path):  # type: os.DirEntry
            if file.name.endswith(".csv"):
                settings_path = os.path.join(entry.path, "settings.ini")
                if not os.path.exists(settings_path):
                    continue
                config = ValidConfig()
                config.read(settings_path, encoding="cp1251")
                file_metadata = {}
                for key in ["normalization", "language", "remove_stopwords"]:
                    file_metadata[key] = config.get_primitive("Preprocessing", key)
                if dicts_equal(file_metadata, metadata_filter, ignore_keys=["vector_dim", "pooling", "window"]):
                    clear_files.setdefault(entry.path, [])
                    clear_files[entry.path].append(file.path)
    return clear_files


def labeled_data_generator(vector_gen, rubricator: str) -> tuple:
    """
    Генератор обучающих пар (X, y)
    Так как большая часть моделей не умеет работать с несколькими метками у одного текста,
    векторы размножаются таким образом, чтобы каждому соответствовала только одна метка
    :param rubricator: "subj", "ipv" или "rgnti"
    :param vector_gen: генератор векторов
    :return: пара (X, y), где X - numpy 2-D array,
                              y - list-like коллекция строковых меток
    """
    for vec_chunk in vector_gen:
        yield df_to_labeled_dataset(vec_chunk, rubricator)


def aggregate_full_dataset_with_pooling(vector_gen, pooling: str) -> pd.DataFrame:
    logger = get_logger("ecs.data_tools.aggregate_full_dataset")
    rubricators = ["subj", "ipv", "rgnti"]
    full_df = pd.DataFrame(columns=["features", *rubricators])
    try:
        for chunk in vector_gen:    # type: pd.DataFrame
            chunk[rubricators] = chunk[rubricators].astype(str)
            chunk["features"].apply(apply_pooling, args=(pooling,))
            full_df = pd.concat([full_df, chunk], ignore_index=True)
    except Exception as e:
        error_ps(logger, f"Error occurred during loading the dataset in memory: {e}")
        exit(1)
    return full_df


def aggregate_labeled_dataset(labeled_data_source) -> tuple:
    """
    Заглушка для неприятной особенности - sklearn не умеет
    учиться инкрементально. Придется собирать датасет в память.
    :param labeled_data_source: источник обучающих пар чанков
    :return: пару x, y - полный набор векторов и меток
    """
    # KERAS_TODO
    # Перенести в класс модели склерн
    x = []
    y = []
    for x_chunk, y_chunk in labeled_data_source:
        x.extend(x_chunk)
        y.extend(y_chunk)
    return x, y


def create_labeled_tt_split(full_df: pd.DataFrame,
                            rubricator: str,
                            test_percent: float) -> tuple:
    x, y = df_to_labeled_dataset(full_df, rubricator)
    return train_test_split(x, y, test_size=test_percent, shuffle=True)


def df_to_labeled_dataset(full_df: pd.DataFrame, rubricator: str) -> tuple:
    x = []
    y = []
    for row in full_df.index:
        codes = full_df.loc[row, rubricator].split("\\")
        for code in codes:
            vec = full_df.loc[row, "features"]
            x.append(vec)
            if rubricator == "rgnti":
                code = code[:5]
            y.append(code)
    return x, y


def path_to_search_pattern(path: str) -> str:
    basename = os.path.basename(path)
    if "." in basename:
        search_pattern = os.path.basename(path).split(".")[0]
    else:
        search_pattern = basename
    return search_pattern


def find_cache_for_source(cache_list: list, raw_path: str) -> str:
    if raw_path == "":
        return ""
    search_pattern = path_to_search_pattern(raw_path)
    for cache_path in cache_list:
        if search_pattern in os.path.basename(cache_path):
            return cache_path
    return ""


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


def create_reading_matrix_gen(path: str, chunk_size: int):
    # Новый бинарный формат (pickle)
    matr_gen = matrix_from_pkl_generator(path, chunk_size=chunk_size)
    return matr_gen


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


def create_caching_pp_gen(raw_path: str, exp_name: str, chunk_size: int, pp_params: dict):
    cache_file = generate_clear_cache_path(raw_path=raw_path, exp_name=exp_name)
    return caching_pp_generator(raw_file=raw_path,
                                chunk_size=chunk_size,
                                cache_path=cache_file,
                                **pp_params)


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


def generate_matrix_cache_path(raw_path: str, exp_name: str) -> str:
    """
    Создает название файла чистого кэша для сырого файла.
    Если надо, создает папку.
    :param raw_path: путь к сырому файлу
    :param exp_name: название эксперимента
    :return: сгенерированный путь
    """
    cache_folder = os.path.join(os.path.dirname(raw_path), f"{exp_name}_matrices")
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    cache_fname = append_to_fname(os.path.basename(raw_path), "_matrices", extension="pkl")
    cache_fpath = os.path.join(cache_folder, cache_fname)
    return cache_fpath


def count_generator_items(gen) -> int:
    return sum(1 for _ in gen)


if __name__ == '__main__':
    print(generate_w2v_fname(vector_dim=100,
                             language="ch",
                             version=42, additional_info="wow"))
