from configparser import ConfigParser
from os.path import join, exists, dirname
from os.path import split as os_split
from os import linesep

from ECS.interface.validation_tools import *
from ECS.core.model_tools import load_class


class ValidConfig(ConfigParser):
    """
    Конфиг, который содержит настройки ECS.
    Возможные типы данных в ini:
        1. Числа - целые и дробные
        2. Строки - без кавычек
        3. Булевские значения - регистронезависимые True и False
        3. Плоские списки со смешанными значениями
            Пример: 1, a string, 0.01, true, -0
            В таких списках не ожидается символов [].
        4. Вложенные списки целых чисел (структура нейросети)
            Пример: [3], [1, 2, 3], [15, 15]
    """
    LIST_DELIM = ","
    map_config = ConfigParser()

    def __init__(self):
        super(ValidConfig, self).__init__()
        self.map_config.read(join(dirname(__file__), "map.ini"))
        self.optionxform = str

    def get_as_list(self, section, key):
        value = self.get(section, key)
        return parse_plain_sequence(value)

    def set(self, section, option, value=None):
        if isinstance(value, list):
            str_value = (self.LIST_DELIM + " ").join(value)
        else:
            str_value = str(value)
        super().set(section, option, str_value)

    def validate_dataset(self):
        ds_section = "TrainingData"
        self.__check_existence(ds_section, "dataset")
        dataset = self.get(ds_section, "dataset")
        val_assert(dataset != "",
                   'Please specify the training dataset folder name in the "dataset directory"')
        val_assert("train" not in dataset and "test" not in dataset,
                   "Keywords 'train' and 'test' are reserved and cannot be used")
        ds_path = dataset
        if not exists(ds_path):
            val_assert(not ("/" in ds_path or "\\" in ds_path),
                       f"The dataset '{ds_path}' looks like path, but does not exist")
            print(f"Searching for '{ds_path}' in default directory")
            ds_path = join(dirname(__file__), "..", "datasets", os_split(dataset)[-1])
            val_assert(exists(ds_path),
                       "Dataset '{}' does not exist!".format(ds_path))
            self.set(ds_section, "dataset", ds_path)

        tp_option_exists = "test_percent" in self.options(ds_section)
        tf_option_exists = "test_file" in self.options(ds_section)
        val_assert(tp_option_exists or tf_option_exists,
                   'Please specify either "test_percent" or "test_file" option')
        if tf_option_exists:
            test_file = self.get(ds_section, "test_file")
        else:
            test_file = ""
        if tp_option_exists:
            test_percent = self.get(ds_section, "test_percent")
        else:
            test_percent = ""
        val_assert(test_percent != "" or test_file != "",
                   'Please specify either "test_percent" or "test_file" option')
        test_path = join(ds_path, test_file)
        if tf_option_exists:
            val_assert(exists(test_path),
                       "Specified test file does not exist!",
                       leave=test_file != "")
            val_assert("train" not in test_file and "test" not in test_file,
                       "Keywords 'train' and 'test' are reserved and cannot be used")
        if tp_option_exists:
            if test_file == "":
                try:
                    test_percent = int(test_percent)
                except ValueError:
                    val_assert(False,
                               '"test_percent" value must be an integer in [0, 99]')
                else:
                    val_assert(0 < test_percent < 100,
                               '"test_percent" value must be an integer in [1, 99]')
                    print('Test file is not specified, '
                          '{}% of the dataset will be used for testing'.format(test_percent))
            else:
                print('Test file is specified, "test_percent" option will not be used')

    def validate_experiment(self):
        # title
        title = self.get("Experiment", "experiment_title", fallback="")
        restricted = ("\\", "*", '"', "?", "/", ":", ">", "<", "|")
        if title != "":
            val_assert(not any(s in title for s in restricted),
                       "Experiment title cannot contain symbols {}".format(", ".join(restricted)))
            val_assert("train" not in title and "test" not in title,
                       "Keywords 'train' and 'test' are reserved and cannot be used")
        # binary
        self.__check_existence("Experiment", "binary")
        self.__check_value("Experiment", "binary", [True, False])
        # n_folds
        self.__check_option_entry("Experiment", "n_folds")
        n_folds = self.get("Experiment", "n_folds")
        try:
            a = int(n_folds)
            val_assert(a > 0,
                       'Value of the "n_folds" option must be positive')
        except ValueError:
            val_assert(False,
                       'Value of the "n_folds" option must be a positive integer')
        # threads
        self.__check_option_entry("Experiment", "threads")
        threads = self.get("Experiment", "threads")
        try:
            a = int(threads)
            val_assert(a > 0,
                       'Value of the "threads" option must be positive')
        except ValueError:
            val_assert(False,
                       'Value of the "threads" option must be a positive integer')
        # rubricator
        self.__check_option_entry("Experiment", "rubricator")
        self.__check_value("Experiment", "rubricator", ["subj", "ipv", "rgnti"])

    def validate_classification(self):
        self.__check_section_existence("Classification")
        for model in self.get_as_list("Classification", "models"):
            val_assert(model in self.map_config.options("SupportedModels"),
                       'Model "{}" is not supported'.format(model))
            val_assert(model in self.sections(),
                       'No options specified for model "{}"'.format(model))
            path = self.map_config.get("SupportedModels", model)
            components = path.split(".")
            class_name = components[-1]
            module_name = ".".join(components[:-1])
            try:
                m = load_class(path)
                val_assert(m is not None,
                           "Module {} cannot be loaded (noneType)".format(path))
                instance = m()
                params = instance.get_params()
                for hp in self.options(model):
                    val_assert(hp in params.keys(),
                               'Model "{}" has no hyperparameter "{}". {}'
                               'Possible hyperparameters: {}'.format(model,
                                                                     hp,
                                                                     linesep,
                                                                     ", ".join(params)))
            except ImportError as e:
                val_assert(False,
                           "Module {} cannot be loaded ({})".format(path, e))
            except AttributeError:
                val_assert(False,
                           'Module {} has no model "{}"'.format(module_name, class_name))

    def validate_preprocessing(self):
        # Tested somehow
        section = "Preprocessing"
        self.__check_section_existence(section)
        options = {"id", "title", "text",
                   "keywords", "subj", "ipv", "rgnti",
                   "correct", "remove_stopwords", "normalization",
                   "language", "batch_size", "kw_delim"}
        for key in options:
            self.__check_option_entry(section, key)
            value = self.get(section, key)
            val_assert(value != "", 'Missing value of "{}" option'.format(key))
        batch_size = self.get("Preprocessing", "batch_size", fallback="")
        #
        self.__check_value(section, "remove_stopwords", [True, False])
        #
        normalization_options = parse_plain_sequence(self.map_config.get("Supported", "normalization"))
        self.__check_value(section, "normalization", normalization_options)
        #
        lang_options = parse_plain_sequence(self.map_config.get("Supported", "languages"))
        self.__check_value(section, "language", lang_options)
        #
        val_assert(is_int(batch_size) and int(batch_size) > 0,
                   "Invalid value of 'batch_size:'\n"
                   "Only positive integers are supported")

    def validate_word_embedding(self):
        section = "WordEmbedding"
        self.__check_section_existence(section)
        use_model = self.get(section, "use_model", fallback="")
        um_option_exists = use_model != ""
        if um_option_exists:
            val_assert(exists(use_model), "Cannot find model {}".format(use_model))
            print("Using specified Word2Vec model: {}".format(use_model))
        else:
            print("W2V model is not specified, new model will be created")
        options = ("vector_dim", "pooling", "window")
        for key in options:
            self.__check_option_entry(section, key)
            value = self.get(section, key)
            val_assert(value != "", 'Missing value of "{}" option'.format(key))
        #
        vector_dim = self.get(section, "vector_dim")
        val_assert(is_int(vector_dim) and int(vector_dim) > 0,
                   "Invalid value of 'vector_dim:'\n"
                   "Only positive integers are supported")
        #
        pooling_options = parse_plain_sequence(self.map_config.get("Supported", "pooling"))
        self.__check_value(section, "pooling", pooling_options)
        #
        window = self.get(section, "window")
        val_assert(is_int(window) and int(window) > 0,
                   "Invalid value of 'vector_dim:'\n"
                   "Only positive integers are supported")

    def validate_normalization(self, valid_lang: str) -> None:
        pp_map_config = ConfigParser()
        pp_map_config.read_file(open(join(dirname(__file__), "..", "preprocessor", "map.ini"), 'r'))
        section = f"Supported{valid_lang.capitalize()}Models"
        val_assert(section in pp_map_config.keys(),
                   f"Язык {valid_lang} не поддерживается")
        preproc = self.get("Preprocessing", "normalization", fallback="")
        # Backward compatibility
        if preproc == "stemming":
            preproc = "snowball"
            self.set("Preprocessing", "normalization", preproc)
        elif preproc == "lemmatization":
            if valid_lang == "ru":
                preproc = "pymystem"
            else:
                preproc = "wordnet"
            self.set("Preprocessing", "normalization", preproc)
        options = pp_map_config[section].keys()
        val_assert(preproc in options or preproc == "no",
                   f"Method '{preproc}' is not supported for language '{valid_lang}'. "
                   f"Available options: {', '.join(options)}")

    def validate_all(self):
        self.validate_dataset()
        print("Dataset settings are valid")
        self.validate_experiment()
        print("Experiment settings are valid")
        self.validate_preprocessing()
        print("Preprocessing settings are valid")
        self.validate_word_embedding()
        print("WordEmbedding settings are valid")
        self.validate_classification()
        print("Classification settings are valid")

    def __check_existence(self, section, option):
        self.__check_section_existence(section)
        self.__check_option_entry(section, option)

    def __check_section_existence(self, section: str):
        val_assert(section in self.sections(),
                   'Missing "{}" section'.format(section))

    def __check_option_entry(self, section, option):
        val_assert(option in self.options(section),
                   'Section "{}" is missing "{}" option'.format(section, option))

    def __check_value(self, section, option, supported: list):
        value = self.get_as_list(section, option)
        for i in value:
            val_assert(i in supported,
                       'Value "{}" of the option "{}" is not supported.\n'
                       'Supported values: {}'.format(i, option, ", ".join(map(str, supported))))

    # def get_model_types(self):
    #     model_types = []
    #     for m in self.get_as_list("Experiment", "models"):
    #         path = self.map_config.get("SupportedModels", m)
    #         model_types.append(load_class(path))
    #     return model_types

    # def get_model_type(self, model):
    #     path = self.map_config.get("SupportedModels", model)
    #     return load_class(path)

    def get_hyperparameters(self, model: str):
        hypers = {}
        for option in self.options(model):
            hp_list = self.get_as_list(model, option)
            if hp_list[0] == "":
                continue
            hypers[option] = self.get_as_list(model, option)
        return hypers

    def get_as_dict(self, section: str) -> dict:
        res = {}
        for i in self.options(section):
            str_val = self.get(section, i)
            res[i] = parse_plain_sequence(str_val)
            if len(res[i]) == 1:
                res[i] = res[i][0]
        return res


if __name__ == '__main__':
    config = ValidConfig()
    config.read(join(dirname(__file__), "../../test/test_settings", "settings.ini"), encoding="cp1251")
    config.validate_dataset()
    config.validate_experiment()
    config.validate_preprocessing()
    config.validate_word_embedding()
    config.validate_classification()
