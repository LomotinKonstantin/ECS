import os
import pickle

import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from ECS.preprocessor.Preprocessor2 import Preprocessor


def pp_from_raw_generator(raw_file: str, chunk_size: int, **pp_settings) -> pd.DataFrame:
    """
    Генератор предобработанных текстов
    :param raw_file: путь к файлу исходных текстов
    :param chunk_size: размер чанка для предобработки
    :param pp_settings: настройки препроцессора
    """
    pp = Preprocessor()
    for chunk in pd.read_csv(raw_file, encoding="cp1251", quoting=3, sep="\t", chunksize=chunk_size):
        pp_chunk = pp.preprocess_dataframe(df=chunk, **pp_settings)
        yield pp_chunk


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
    cache_invalid_name = f"{os.path.basename(raw_file).split('.')[0]}.invalid"
    cache_invalid_path = os.path.join(os.path.dirname(cache_path), cache_invalid_name)
    with open(cache_invalid_path, "w") as cache_file:
        cache_header = True
        for pp_chunk in pp_from_raw_generator(raw_file=raw_file, chunk_size=chunk_size, **pp_settings):
            pp_chunk.to_csv(cache_file, encoding="utf8", sep="\t", index=False, header=cache_header)
            cache_header = False
            yield pp_chunk
    if os.path.exists(cache_path):
        os.remove(cache_path)
    os.rename(cache_invalid_path, cache_path)


def create_w2v(pp_sources: list, vector_dim: int) -> Word2Vec:
    """
    Создать модель Word2Vec для предобработанных текстов.
    :param vector_dim: размерность векторной модели
    :param pp_sources: список инстансов генератора предобработанных данных (pp_generator или caching_pp_generator)
    :returns обученная модель Word2Vec
    """
    w2v = Word2Vec(size=vector_dim, window=4, min_count=3, workers=3)
    init = False
    for pp_source in pp_sources:
        for pp_chunk in pp_source:
            sentence = [text.split() for text in pp_chunk["text"].values]
            w2v.build_vocab(sentence, update=init)
            # TODO: вынести количество эпох и размер окна в параметры
            w2v.train(sentence, epochs=20, total_examples=len(sentence))
            init = True
    return w2v


def generate_w2v_fname(vector_dim: int,
                       language: str,
                       dataset_size: int,
                       version=1,
                       additional_info=None) -> str:
    """
    Сгенерировать имя файла модели, совместимое с ATC.
    Для правильной работы ATC получает информацию о модели из имени файла.
    :param vector_dim: размерность векторной модели
    :param language: язык словаря
    :param dataset_size: размер обучающего датасета
    :param version: [опционально] версия
    :param additional_info: [опционально] допольнительная короткая строка с информацией
    :return: сгенерированное имя файла
    """
    import datetime
    now = datetime.datetime.today()
    date = f"{now.day}_{now.month}_{str(now.year)[2:]}"
    name = f"w2v_model_{vector_dim}_{language}_{dataset_size // 1000}k"
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


def vector_from_csv_generator(vector_path: str, chunk_size: int):
    for chunk in pd.read_csv(vector_path, index_col=0, sep="\t", chunksize=chunk_size):
        yield chunk


def vectorize_text(text: str, w2v_model: Word2Vec, conv_type: str) -> np.ndarray:
    tokens = text.split()
    text_matrix = []
    for word in tokens:
        try:
            word_vector = w2v_model[word]
        except KeyError:
            word_vector = np.zeros(w2v_model.vector_size)
        text_matrix.append(word_vector)
    text_matrix = np.vstack(text_matrix)
    if conv_type == "sum":
        return text_matrix.sum(axis=0)
    elif conv_type == "mean":
        return text_matrix.mean(axis=0)
    elif conv_type == "max":
        return text_matrix.max(axis=0)
    else:
        raise ValueError(f"Conv type '{conv_type}' is not supported")


def vectorize_pp_chunk(pp_chunk: pd.DataFrame, w2v_model: Word2Vec, conv_type: str) -> pd.DataFrame:
    """
    Добавляет столбец vectors
    :param pp_chunk: датафрейм с колонкой text, содержащей предобработанные тексты
    :param w2v_model: модель Word2Vec
    :param conv_type: тип свертки: sum, mean или max
    :return: исходный датафрейм с добавленной колонкой vectors
    """
    vectors = []
    for text in pp_chunk["text"]:
        vectors.append(vectorize_text(text, w2v_model, conv_type))
    pp_chunk["vectors"] = vectors
    return pp_chunk


def caching_vector_generator(pp_source,
                             w2v_file: str,
                             cache_path: str,
                             conv_type: str) -> pd.DataFrame:
    """
    Кэширующий генератор векторов
    :param pp_source: генератор предобработанных чанков
    :param w2v_file: путь к файлу модели Word2Vec
    :param cache_path: путь к файлу кэша
    :param conv_type: тип свертки матрицы текста: sum, mean или max
    :return: чанк в формате pd.Dataframe
    """
    model, lang = load_w2v(w2v_file)
    metadata = {
        "language": lang,
        "conv_type": conv_type
    }
    cache_invalid_path = f"{os.path.basename(cache_path)}.invalid"
    with open(cache_invalid_path, "wb") as cache_file:
        pickle.dump(metadata, cache_file)
        for pp_chunk in pp_source:
            vector_chunk = vectorize_pp_chunk(pp_chunk, w2v_model=model, conv_type=conv_type)
            for row in vector_chunk.index:
                entry = (
                    vector_chunk.loc[row, "vectors"],
                    vector_chunk.loc[row, "subj"],
                    vector_chunk.loc[row, "ipv"],
                    vector_chunk.loc[row, "rgnti"],
                )
                pickle.dump(entry, cache_file)
            yield vector_chunk
    if os.path.exists(cache_path):
        os.remove(cache_path)
    os.rename(cache_invalid_path, cache_path)


def read_pkl_metadata(pkl_path: str) -> dict:
    pkl_file = open(pkl_path, "rb")
    metadata = pickle.load(pkl_file)
    return metadata


def vector_from_pkl_generator(pkl_path: str, chunk_size: int) -> pd.DataFrame:
    pkl_file = open(pkl_path, "rb")
    # Первая строка - метадата
    pickle.load(pkl_file)
    chunk = {
        "vectors": [], "subj": [],
        "ipv": [], "rgnti": []
    }
    cntr = 0
    while 1:
        try:
            entry = pickle.load(pkl_file)
            chunk["vectors"].append(entry[0])
            chunk["subj"].append(entry[1])
            chunk["ipv"].append(entry[2])
            chunk["rgnti"].append(entry[3])
            cntr += 1
            if cntr == chunk_size:
                cntr = 0
                yield pd.DataFrame(chunk)
                chunk = {
                    "vectors": [], "subj": [],
                    "ipv": [], "rgnti": []
                }
        except EOFError:
            break
    if len(chunk["vectors"]) > 0:
        yield pd.DataFrame(chunk)


def create_vector_generator():
    # TODO
    pass


if __name__ == '__main__':
    print(generate_w2v_fname(vector_dim=100,
                             language="ch",
                             dataset_size=88005553535,
                             version=42, additional_info="wow"))
