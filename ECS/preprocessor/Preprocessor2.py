import os
import re
from math import ceil
from random import choice
from time import time
import pandas as pd
from os.path import join, dirname
from configparser import ConfigParser
from importlib import import_module


def expand_language(lang: str):
    if lang == "ru":
        return "russian"
    if lang == "en":
        return "english"
    return lang


class Normalizer:

    def __init__(self, preproc: str, language: str):
        self.language = language
        self.preproc = preproc
        self.map_config = ConfigParser()
        self.map_config.read_file(open(join(dirname(__file__), "map.ini"), 'r'))
        section = f"Supported{self.language.capitalize()}Models"
        if section not in self.map_config.keys():
            raise ValueError('Preprocessing is not supported for this language')
        if self.preproc not in list(self.map_config[section].keys()):
            raise ValueError(f'The preprocessing technique is not supported for this language')
        components = self.map_config['Supported' + self.language.capitalize() + 'Models'][self.preproc].split('.')
        module_name = ".".join(components[:-1])
        try:
            module = import_module(module_name)
            self.class_type = getattr(module, components[-1])
        except ImportError:
            print(f"Package '{module_name}' is not installed!")
            exit(0)

    def normalize(self, text: str, return_list=False):
        if self.preproc == 'textblob':
            alg = self.class_type(text)
            token_list = alg.lemmatize()
        else:
            alg = self.class_type()
            if hasattr(alg, 'stem'):
                token_list = alg.stem(text)
            elif hasattr(alg, 'lemmatize'):
                token_list = alg.lemmatize(text)
            else:
                raise ValueError(f"Algorithm {alg} has unknown API")
        # Выбор формата результата
        # print(self.preproc, self.language, token_list)
        if return_list:
            if isinstance(token_list, str):
                result = token_list.split()
            else:
                result = token_list
        else:
            if isinstance(token_list, list):
                result = " ".join(remove_empty_items(token_list))
            else:
                result = token_list
        return result


class TimeTracer:
    def __init__(self):
        self.started = False
        self.time = None

    def start(self):
        self.time = time()
        self.started = True

    def pause(self):
        self.started = False

    def check(self) -> float:
        res = time() - self.time
        self.time = time()
        return res


def remove_empty_items(lst: list):
    return list(filter(lambda x: len(x.strip()) > 0, lst))


class Preprocessor:
    """
    Класс для выполнения предобработки однотипных данных.
    Может брать данные из трех уровней источников:
        - Из набора файлов: метод preprocess_files(filenames: list, columns: dict, kw_delim: str)
        - Если данные нуждаются в дополнительной обработке до препроцессинга,
          можно обработать DataFrame: метод preprocess_dataframe(texts: DataFrame)
        - Обычный текст:    метод preprocess(text: str)

    —
    _
    """
    delim = " ☺☻ "
    sw_files = {"ru": "ru_stopwords.txt",
                "en": "en_stopwords.txt"}
    md_file = "viniti_md.txt"
    standard_columns = ["id", "text", "subj", "ipv", "rgnti"]

    DEBUG = True

    def __init__(self, groups_to_save=("буквы рус.", "пробел", "дефис", "буквы лат.")):
        """
        Препроцессор для обработки обучающих данных из ВИНИТИ.
        ---------------------------------------------------
        Аргументы:

            groups_to_save: Группы символов алфавита ВИНИТИ, которые не будут удалены из текста
                "буквы рус."       - кириллица (сохр. по умолчанию)
                "буквы греч."      - буквы греческого алфавита
                "буквы лат."       - латинские буквы (сохр. по умолчанию)
                "буквы спец."      - буквы љ, њ (хорватский), готические и ажурные буквы (использ. в математике)
                "дефис"            - в этой группе только дефис "-" (сохр. по умолчанию)
                "диакриты"         - диакритические символы (á, ŝ, ž и т.д.)
                "диакр. наезж."    - другие диакриты
                "зн. преп. обычн." - знаки препинания.
                "индексы"          - обозначения верхних и нижних индексов
                "команды"          - перевод, начало и конец строки, жирный шрифт, курсив, цвет
                "пробел            - в этой группе только пробел " "
                "служебные"        - признак текста с посторонней разметкой и нераспознанного символа,
                                     приоритеты при сортировке
                "спец. зн. доп."   - тильда, валюта, градус, планеты, обратный слэш
                "стрелки"          - разные стрелки ascii
                "точка"            - в этой группе только точка "."
                "форм. доп."       - математические символы, операторы и константы
                "форм. обычн."     - арифметические операторы
                "цифры"            - арабские цифры
        """

        self.groups_to_save = groups_to_save
        self.stopwords = self.__load_sw()
        self.viniti_md = self.__load_md()
        self.normalizer = None
        # self.email_re = re.compile("(\\w+[.-]?)+@(\\w+\\.?)+\\.\\w{2,}")
        self.email_re = re.compile(r"\S+@\S+")
        self.md_re = "+|".join(self.viniti_md)
        self.timer = TimeTracer()
        self.last_language = ""

    def recognize_language(self, text: str, default="none"):
        """
        Распознавание языка текста на основе предварительно загруженных стоп-слов.
        Если стоп-слова не загружены, нужно загрузить их:
            self.stopwords = self.__load_sw()
        -------------------------------------------------
        Аргументы:

            text: Текст, язык которого надо определить.

            default: Действия в случае несоответствия текста ни одному из языков.
                     Варианты:
                         - "random": Попробовать угадать язык случайным образом
                         - "error" : Вызвать ошибку
                         - "none"  : Вернуть None (default)
        """
        text_set = set(text.lower().split())
        res_lang = None
        max_entries = 0
        for lang, sw in self.stopwords.items():
            sw_set = set(sw)
            entries = len(text_set.intersection(sw_set))
            if entries > max_entries:
                max_entries = entries
                res_lang = lang
        if res_lang is None:
            if default == "random":
                res_lang = choice(tuple(self.stopwords.keys()))
            elif default == "error":
                raise ValueError("Не удалось определить язык")
        return res_lang

    @staticmethod
    def __dense(text: str) -> str:
        return re.sub("\\s{2,}", " ", text)

    def __remove_stopwords(self, text: str, lang: str, sub=" ") -> str:
        return sub.join(filter(lambda x: x not in self.stopwords[lang], text.split()))

    @staticmethod
    def __remove_single_formulas(text: str, sub=" ") -> str:
        # return re.sub("(?<=[^$])\\$(?=[^$]).+?(?<=[^$])\\$(?=[^$])", sub, text)
        return re.sub("_[ёЁ](var)?", sub, text)

    def __remove_email(self, text: str, sub=" ") -> str:
        return self.email_re.sub(sub, text)

    def __remove_md(self, text: str) -> str:
        res = text
        for md_element in self.viniti_md:
            res = res.replace(md_element, " ")
        return res

    def __beautify(self, text: str) -> str:
        # # Fix -blablabla-
        # res = re.sub("-{2,}", "-", text)
        # res = re.sub("((?<=\\s)-(?=\\S))|((?<=\\S)-(?=\\s))", " ", res)
        # # Fix blabla - a f - g blablabla
        # res = re.sub("(\\b-\\B)|(\\B-\\b)", " ", res)
        # Fuck all this stuff
        # We don't need '-' bullshit
        res = re.sub("[-_]", " ", text)
        res = re.sub(r"(?<=\s)\S(?=\s)", " ", res)
        return self.__dense(res).strip()

    def preprocess(self,
                   text: str,
                   remove_stopwords: bool,
                   remove_formulas: bool,
                   normalization: str,
                   language="auto",
                   default_lang="error") -> str:
        """
        Предобработка одного текста.
        ----------------------------
        Аргументы:

            text: Строка с текстом для предобработки.

            remove_stopwords: Удалять стоп-слова.
                              Возможные варианты: [True, False].

            remove_formulas: Удалять TeX-формулы, заключенные в $...$
                             Формулы вида $$...$$ удаляются всегда.
                             Возможные варианты: [True, False].

            normalizaion: Метод нормализации слова. Возможные варианты:
                          ["no", "lemmatization", "stemming"].

            language: Язык текста. По умолчанию язык определяется автоматически.
                      Возможные варианты: ["auto", "ru", "en"].
                      Автоопределение занимает время (особенно на больших текстах),
                      поэтому лучше задать определенный язык.

            default_lang: Действия в случае несоответствия текста ни одному из языков.
                          Аргумент используется только при language="auto".
                          Варианты:
                             - "random": Попробовать угадать язык случайным образом
                             - "error" : Вызвать ошибку
                             - "none"  : Вернуть None (default)
        """
        lang = language
        if language == "auto":
            lang = self.recognize_language(text.lower(), default_lang)
        if lang is None:
            raise ValueError("Unable to recognize language!")
        # res = text
        self.last_language = lang
        res = text
        print("Removing emails...")
        res = self.__remove_email(res)
        # print(self.delim in res, type(res))
        # print("Removing $$...$$")
        # res = self.__remove_double_formulas(res)
        if remove_formulas:
            print("Removing $...$")
            res = self.__remove_single_formulas(res)
        print("Removing markdown")
        res = self.__remove_md(res)
        # print(self.delim in res, type(res))
        if remove_stopwords:
            print("Dense...")
            res = self.__dense(res)
            # print(self.delim in res, type(res))
            print("Removing stopwords")
            res = self.__remove_stopwords(res.lower(), lang)
            # print(self.delim in res, type(res))
        print("Some beauty...")
        res = self.__beautify(res)
        # with open("log.txt", "w", encoding="utf8") as f:
        #     f.write(res)
        # print(self.delim in res, type(res))
        if normalization != "no":
            print("Normalization...")
            res = Normalizer(normalization, lang).normalize(res)
            res = re.sub(" ".join(self.delim).strip(), self.delim, res)
            print(self.delim in res, type(res))
            # with open("log.txt", "w", encoding="utf-8") as f:
            #     f.write(res)
        # print("Removing widow '-'...")
        # res = re.sub("\\s-\\s", "-", res)
        print("And finally done!")
        return res

    def preprocess_dataframe(self, df: pd.DataFrame,
                             remove_stopwords: bool,
                             remove_formulas: bool,
                             normalization: str,
                             kw_delim: str,
                             language="auto",
                             default_lang="error",
                             columns=None,
                             title_weight=1,
                             body_weight=1,
                             kw_weight=1,
                             batch_size: int = 50000) -> pd.DataFrame:
        """
        Предобработка датафрейма
        -------------------------
        Аргументы:
            df: Датафрейм с колонками обязательными колонками
                ["id_publ", "title", "ref_txt", "kw_list", "subj", "ipv", "rgnti"].
                Колонка "eor" необязательна. Если она есть, то будет удалена.
                Если колонки отличаются, см. аргумент columns.

            remove_stopwords: Удалять стоп-слова.
                              Возможные варианты: [True, False].

            remove_formulas: Удалять TeX-формулы, заключенные в $...$
                             Формулы вида $$...$$ удаляются всегда.
                             Возможные варианты: [True, False].

            normalizaion: Метод нормализации слова. Возможные варианты:
                          ["no", "lemmatization", "stemming"].

            kw_delim: Разделитель ключевых слов. В русских текстах это "\"
                      (в Python обратный слэш нужно экранировать, поэтому при указании это "\\"),
                      в английских - ";"

            language: Язык текста. По умолчанию язык определяется автоматически.
                      Возможные варианты: ["auto", "ru", "en"].
                      Автоопределение занимает время (особенно на больших текстах),
                      поэтому лучше задать определенный язык.

            default_lang: Действия в случае несоответствия текста ни одному из языков.
                          Аргумент используется только при language="auto".
                          Варианты:
                             - "random": Попробовать угадать язык случайным образом
                             - "error" : Вызвать ошибку
                             - "none"  : Вернуть None (default)

            columns: Словарь, сопоставляющий используемые в методе псевдонимы
                     колонок (ключи) и фактические названия колонок (значения).
                     Если названия колонок в датафрейме отличаются,
                     достаточно задать, что есть что, и передать через этот аргумент.

            title_weigh--→: Целочисленные коэффициенты, на которые
            body_weight--→: умножаются соответствующие
            kw_weight----→: части документа.

            batch_size: Количество текстов, предобрабатывающихся одновременно.
                        Если количество текстов в датафрейме больше этого числа,
                        они будут обрабатываться батчами. Если меньше, то этот
                        аргумент ни на что не влияет. Чем больше размер батча, тем больше
                        памяти будет израсходовано. Чем он меньше, тем дольше будет обработка.

        ---------------------------
        Возвращает: Датафрейм с колонками по стандарту ATC_dev:
                        id - идентификатор текста (не индекс!)
                        text - основной текст документа
                        subj - коды отделов
                        ipv - коды РЖ
                        rgnti - коды ГРНТИ
        """
        if columns is None:
            columns = {"id": "id_publ",
                       "title": "title",
                       "text": "ref_txt",
                       "keywords": "kw_list",
                       "subj": "SUBJ",
                       "ipv": "IPV",
                       "rgnti": "RGNTI",
                       "correct": "eor"}
        self.__trace_time("Init")
        for key, value in columns.items():
            if value is None:
                if key != "correct":
                    raise ValueError("Колонки не могут быть None! См. докстринг.")
            if value not in df.columns:
                if value != "correct":
                    raise ValueError(f"Колонка {value} не найдена в датафрейме. "
                                     f"См. аргумент columns")
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive integer! Found {batch_size}")
        self.__trace_time("Columns checking")
        # Подготовка датафрейма, объединение частей документа
        un_df = df.drop([columns["title"], columns["keywords"]], axis=1)
        if columns["correct"] is not None:
            un_df = un_df.drop(columns["correct"], axis=1)
        un_df.columns = self.standard_columns
        self.__trace_time("Copying DF")
        #
        batches = ceil(len(un_df.index) / batch_size)
        print("Starting merging document part columns")
        current_batch = 0
        for n, i in enumerate(df.index):
            new_batch = ceil(n / batch_size)
            if new_batch > current_batch:
                print("Merging part {}/{}".format(new_batch, batches))
                current_batch = new_batch
            title = str(df.loc[i, columns["title"]])
            body = str(df.loc[i, columns["text"]])
            kw = str(df.loc[i, columns["keywords"]]).replace(kw_delim, " ")
            if kw == "nan":
                kw = ""
            un_df.loc[i, "text"] = "{} {} {}".format(title * title_weight, body * body_weight, kw * kw_weight)
        self.__trace_time("Merging document part columns")
        # Предобработка
        print("Starting preprocessing")
        result = []
        if len(un_df.index) > batch_size:
            for i in range(batches - 1):
                print("Part", i + 1, "/", batches)
                raw_part = self.delim.join(un_df.text.values[batch_size * i: batch_size * (i + 1)])
                new_part = self.preprocess(text=raw_part,
                                           remove_stopwords=remove_stopwords,
                                           remove_formulas=remove_formulas,
                                           normalization=normalization,
                                           language=language,
                                           default_lang=default_lang)
                if new_part is not None:
                    result += new_part.split(self.delim)
            print("Part", batches, "/", batches)
            raw_part = self.delim.join(un_df.text.values[batch_size * (batches - 1): len(un_df.index)])
            new_part = self.preprocess(text=raw_part,
                                       remove_stopwords=remove_stopwords,
                                       remove_formulas=remove_formulas,
                                       normalization=normalization,
                                       language=language,
                                       default_lang=default_lang)
            if new_part is not None:
                result += new_part.split(self.delim)
            self.__trace_time("Batch preprocessing")
        else:
            print("DF size < batch_size")
            raw_part = self.delim.join(un_df.text.values)
            new_part = self.preprocess(text=raw_part,
                                       remove_stopwords=remove_stopwords,
                                       remove_formulas=remove_formulas,
                                       normalization=normalization,
                                       language=language,
                                       default_lang=default_lang)
            if new_part is not None:
                result = new_part.split(self.delim)
            self.__trace_time("Full-DF preprocessing")
        print("Source index len: {}\nResult len: {}".format(len(un_df.index), len(result)))
        if len(un_df.index) != len(result):
            # print(list(filter(lambda x: self.delim in x, result)))
            raise IndexError("Regexp has devoured sth again :(")
        un_df.text = pd.Series(list(map(lambda x: x.strip(), result)))
        self.__trace_time("Stripping")
        print("Successfully processed", len(result), "texts of ", len(df.index))
        return un_df

    def preprocess_file(self, fn_in: str, fn_out: str,
                        remove_stopwords: bool,
                        remove_formulas: bool,
                        normalization: str,
                        kw_delim: str,
                        language="auto",
                        default_lang="error",
                        columns=None,
                        title_weight=1,
                        body_weight=1,
                        kw_weight=1,
                        batch_size=50000):
        if columns is None:
            columns = {"id": "id_publ",
                       "title": "title",
                       "text": "ref_txt",
                       "keywords": "kw_list",
                       "subj": "SUBJ",
                       "ipv": "IPV",
                       "rgnti": "RGNTI",
                       "correct": "eor"}
        df = pd.read_csv(fn_in, encoding="cp1251", quoting=3, sep="\t")
        res = self.preprocess_dataframe(df=df,
                                        remove_stopwords=remove_stopwords,
                                        remove_formulas=remove_formulas,
                                        normalization=normalization,
                                        kw_delim=kw_delim,
                                        language=language,
                                        columns=columns,
                                        default_lang=default_lang,
                                        batch_size=batch_size,
                                        title_weight=title_weight,
                                        body_weight=body_weight,
                                        kw_weight=kw_weight)
        res.to_csv(fn_out, encoding="utf8", sep="\t", index=False)

    def __load_md(self) -> tuple:
        md_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "viniti_md.txt"),
                            encoding="cp1251",
                            sep="\t",
                            names=["element", "meaning", "code", "group"],
                            quoting=3)
        md_df.drop(md_df.index[md_df.group.isin(self.groups_to_save)], inplace=True)
        return tuple(sorted(md_df.element, key=lambda x: len(x), reverse=True))

    def __load_sw(self) -> dict:
        res = {}
        for lang, fn in self.sw_files.items():
            sw_file = open(os.path.join(os.path.dirname(__file__), fn), encoding="utf8")
            sw = sw_file.read().split()
            res[lang] = tuple(remove_empty_items(sw))
        return res

    def __trace_time(self, description: str):
        if self.DEBUG:
            if description.lower() == "init":
                self.timer.start()
                print("Time recording is started")
            else:
                print(description, "has taken {} sec".format(self.timer.check()))


if __name__ == "__main__":
    # in_folder = r"D:\Desktop\VINITI\research\data\Eng_Samples2"
    # files = ["SampleEng2learn.txt",
    #          "SampleEng2test.txt"]
    in_folder = r"D:\Desktop\VINITI\research\data\english_test_train_sep17"
    files = ["SampleLearn2016Eng_2017-09-21.txt",
             "SampleTest2017Eng_2017-09-21.txt"]
    out_folder = os.path.join(in_folder, "clear")
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    t = time()
    a = Preprocessor()
    columns_dict = {"id": "id_bo",
                    "title": "title",
                    "text": "ref_txt",
                    "keywords": "kw_list",
                    "subj": "SUBJ",
                    "ipv": "IPV",
                    "rgnti": "RGNTI",
                    "correct": "eor"}
    for file in files:
        print(file)
        test_df = pd.read_csv(os.path.join(in_folder, file), encoding="cp1251", quoting=3, sep="\t")
        pp_df = a.preprocess_dataframe(df=test_df,
                                       remove_stopwords=True,
                                       remove_formulas=True,
                                       normalization="no",
                                       kw_delim=";",
                                       language="auto",
                                       columns=columns_dict,
                                       default_lang="none",
                                       batch_size=40000)
        pp_df.to_csv(os.path.join(out_folder, (file[:-4] + "_clear.txt")), encoding="utf8", sep="\t", index=False)
    print(time() - t, "sec")
