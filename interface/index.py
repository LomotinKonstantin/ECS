# import json
import os
from re import fullmatch
from interface.valid_config import ValidConfig


class Index:
    def __init__(self, root_dir: str):
        self.path = os.path.join(root_dir, "index.json")
        self.data = None

    # def load(self) -> None:
    #     """
    #     Загружает индекс. Если файла нет, вызывается rebuild.
    #
    #     :return: None
    #     """
    #     if self.exists():
    #         self.data = json.load(open(self.path))
    #     else:
    #         print("Repairing index")
    #         self.rebuild()

    def __form_entry(self, path_to_config: str, section: str) -> dict:
        parser = ValidConfig()
        parser.read(path_to_config)
        new_entry = parser.get_as_dict(section)
        new_entry["ds_title"] = parser.get("Experiment", "experiment_title")
        return new_entry

    def __get_ct_entry(self, path_to_config: str) -> dict:
        return self.__form_entry(path_to_config, "Preprocessing")

    def __get_vectors_entry(self, path_to_config: str) -> dict:
        return self.__form_entry(path_to_config, "WordEmbedding")

    def build(self) -> None:
        """
        Строит индекс.
        :return: None
        """
        index = {}
        dirs = lambda a: next(os.walk(a))[1]
        dataset_folder = os.path.dirname(self.path)
        clear = []
        vectors = []
        ds_path = dataset_folder
        try:
            for inner_folder in dirs(ds_path):
                # Путь к копии файла настроек
                # с которыми формировались векторы или чистые тексты
                path_to_config = os.path.join(ds_path, inner_folder, "settings.ini")
                if fullmatch(".+_clear", inner_folder) is not None:
                    entry = self.__get_ct_entry(path_to_config)
                    entry["path"] = os.path.join(ds_path, inner_folder)
                    clear.append(entry)
                elif fullmatch(".+_vectors", inner_folder) is not None:
                    entry = {**self.__get_vectors_entry(path_to_config),
                             **self.__get_ct_entry(path_to_config)}
                    entry["path"] = os.path.join(ds_path, inner_folder)
                    vectors.append(entry)
        except StopIteration:
            pass
        index[dataset_folder] = {
            "clear": clear,
            "vectors": vectors
        }
        self.data = index
        # json.dump(self.data, open(self.path, "w"), indent=4)

    def exists(self) -> bool:
        """
        Проверяет файл индекса на существование

        :return: Признак существования индекса, bool
        """
        return os.path.exists(self.path)

    def dataset_in_index(self, dataset: str) -> bool:
        """
        Проверяет наличие датасета в индексе
        :param dataset: название датасета
        :return: Признак наличия датасета в индексе, bool
        """
        return dataset in self.data.keys()

    def __get_entries(self, dataset: str, entry_type: str) -> list:
        """
        Получить список записей для датасета dataset и
        типа предобработки entry_type.

        :param dataset: название датасета
        :param entry_type: 'clear' или 'vectors'

        :return: список записей
        """
        return self.data[dataset][entry_type]

    def __check_title_existance(self, dataset: str, entry_type: str, title: str) -> bool:
        """
        Проверить существование записи с названием title среди записей
        о предобработанных данных типа entry_type в датасете dataset.

        :param dataset: название датасета
        :param entry_type: 'clear' или 'vectors'
        :param title: название записи

        :return: Признак наличия записи, bool
        """
        entries = self.data[dataset][entry_type]
        for entry in entries:
            if entry["ds_title"] == title:
                return True
        return False

    def __add_entry(self, dataset: str, entry_type: str, new_entry: dict, rewrite: bool) -> bool:
        """
        Добавить запись new_entry в список записей типа entry_type датасета dataset.
        Если rewrite == True, то при наличии записи с таким названием она будет
        полностью перезаписана.

        :param dataset: название датасета
        :param entry_type: 'clear' или 'vectors'
        :param new_entry: новая запись
        :param rewrite: перезаписывать записи с одинаковыми названиями эксперимента
                        По умолчанию False.

        :return: True, если удалось добавить или перезаписать запись.
                 False, если такая запись уже есть и rewrite == False
        """
        new_title = new_entry["ds_title"]
        if self.__check_title_existance(dataset, entry_type, new_title):
            if rewrite:
                for i in range(len(self.data[dataset][entry_type])):
                    entry = self.data[dataset][entry_type][i]
                    if entry["ds_title"] == new_title:
                        self.data[dataset][entry_type][i] = new_entry
                        return True
            else:
                return False
        else:
            self.data[dataset][entry_type].append(new_entry)
            return True

    def cleared_texts_list(self) -> list:
        """
        Получить список всех предобработанных вариантов датасета

        :param dataset: название датасета

        :return: Список словарей с данными о вариантах датасета
        """
        if self.data is None:
            raise ValueError("Index file is not loaded!")
        dataset = os.path.dirname(self.path)
        return self.__get_entries(dataset, "clear")

    def vectors_list(self) -> list:
        """
        Получить список всех вариантов векторов датасета

        :param dataset: полный путь к папке датасета

        :return: Список словарей с данными о векторах
        """
        if self.data is None:
            raise ValueError("Index file is not loaded!")
        dataset = os.path.dirname(self.path)
        return self.__get_entries(dataset, "vectors")

    def ct_title_exists(self, dataset: str, title: str) -> bool:
        """
        Проверяет наличие эксперимента title в записях о
        предобработанных текстах датасета dataset.

        :param dataset: название датасета
        :param title: название эксперимента в записи

        :return: Признак существования записи с таким названием, bool
        """
        return self.__check_title_existance(dataset, "clear", title)

    def vectors_title_exists(self, dataset: str, title: str) -> bool:
        """
        Проверяет наличие эксперимента title в записях о
        векторах датасета dataset.

        :param dataset: название датасета
        :param title: название эксперимента в записи

        :return: Признак существования записи с таким названием, bool
        """
        return self.__check_title_existance(dataset, "vectors", title)

    def add_clear_text(self, dataset: str, new_entry: dict, rewrite=False) -> bool:
        """
        Добавить новую запись о предобработанном варианте

        :param dataset: название датасета
        :param new_entry: запись о данных новых текстов
                        (название эксперимента и поля секции Preprocessing)
        :param rewrite: перезаписывать записи с одинаковыми названиями эксперимента
                        По умолчанию False.

        :return: True, если удалось добавить или перезаписать запись.
                 False, если такая запись уже есть и rewrite == False
        """
        if self.data is None:
            raise ValueError("Index file is not loaded!")
        return self.__add_entry(dataset, "clear", new_entry, rewrite)

    def add_vector(self, dataset: str, new_entry: dict, rewrite=False) -> bool:
        """
        Добавить новую запись о векторах

        :param dataset: название датасета
        :param new_entry: запись о данных новых текстов
                        (название эксперимента и поля секции WordEmbedding)
        :param rewrite: перезаписывать записи с одинаковыми названиями эксперимента
                        По умолчанию False.

        :return: True, если удалось добавить или перезаписать запись.
                 False, если такая запись уже есть и rewrite == False
        """
        if self.data is None:
            raise ValueError("Index file is not loaded!")
        return self.__add_entry(dataset, "vectors", new_entry, rewrite)


# if __name__ == '__main__':
#     test_json = '{"ds_title": "exp1", "some_data": {"entry1": {"foo": "bar", "finn": "jake"}, "entry2": [42, 3.14, 2.7]}}'
#     loaded_dict = json.loads(test_json)
#     assert loaded_dict["ds_title"] == "exp1", "title failed"
#     assert loaded_dict["some_data"]["entry1"]["finn"] == "jake", "finn failed"
#     assert loaded_dict["some_data"]["entry2"][1] == 3.14, "pi failed"
#     assert json.dumps(loaded_dict) == test_json, "dumps failed"
#     print("Json module works correctly")
#     index = Index()
#     index.load()
#     print(index.path)
#     print(json.dumps(index.data, sort_keys=True, indent=4))
