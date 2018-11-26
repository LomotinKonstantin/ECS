from configparser import ConfigParser
import re
import sys
from os.path import join, exists, dirname
from os import linesep, walk
from importlib import import_module


class ValidConfig(ConfigParser):
    LIST_DELIM = ","
    list_re = "\[.*?\]"
    map_config = ConfigParser()

    def __init__(self):
        super(ValidConfig, self).__init__()
        self.map_config.read(join(dirname(__file__), "map.ini"))
        self.optionxform = str

    def get_as_list(self, section, key):
        value = self.get(section, key)
        return self.smart_parse_list(value)

    def set(self, section, option, value=None):
        if type(value) is list:
            str_value = (self.LIST_DELIM + " ").join(value)
        else:
            str_value = str(value)
        super().set(section, option, str_value)

    def validate_dataset(self):
        ds_section = "TrainingData"
        self.__check_existance(ds_section, "dataset")
        dataset = self.get(ds_section, "dataset")
        self.__assert(dataset != "",
                      'Please specify the training dataset folder name in the "dataset directory"')
        self.__assert("train" not in dataset and "test" not in dataset,
                      "Keywords 'train' and 'test' are reserved and cannot be used")
        ds_path = join(dirname(__file__), "..", "datasets", dataset)
        self.__assert(exists(ds_path),
                      'Dataset "{}" does not exist! Check "datasets" folder'.format(dataset))
        tp_option_exists = "test_percent" in self.options(ds_section)
        tf_option_exists = "test_file" in self.options(ds_section)
        self.__assert(tp_option_exists or tf_option_exists,
                      'Please specify either "test_percent" or "test_file" option')
        if tf_option_exists:
            test_file = self.get(ds_section, "test_file")
        else:
            test_file = ""
        if tp_option_exists:
            test_percent = self.get(ds_section, "test_percent")
        else:
            test_percent = ""
        self.__assert(test_percent != "" or test_file != "",
                      'Please specify either "test_percent" or "test_file" option')
        test_path = join(ds_path, test_file)
        if tf_option_exists:
            self.__assert(exists(test_path),
                          "Specified test file does not exist!",
                          leave=test_file != "")
            self.__assert("train" not in test_file and "test" not in test_file,
                          "Keywords 'train' and 'test' are reserved and cannot be used")
        if tp_option_exists:
            if test_file == "":
                try:
                    test_percent = int(test_percent)
                except ValueError:
                    self.__assert(False,
                                  '"test_percent" value must be an integer in [0, 99]')
                else:
                    self.__assert(0 < test_percent < 100,
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
            self.__assert(not any(s in title for s in restricted),
                          "Experiment title cannot contain symbols {}".format(", ".join(restricted)))
            self.__assert("train" not in title and "test" not in title,
                          "Keywords 'train' and 'test' are reserved and cannot be used")
        # binary
        self.__check_existance("Experiment", "binary")
        self.__check_value("Experiment", "binary", [True, False])
        # n_folds
        self.__check_option_entry("Experiment", "n_folds")
        n_folds = self.get("Experiment", "n_folds")
        try:
            a = int(n_folds)
            self.__assert(a > 0,
                          'Value of the "n_folds" option must be positive')
        except ValueError:
            self.__assert(False,
                          'Value of the "n_folds" option must be a positive integer')
        # threads
        self.__check_option_entry("Experiment", "threads")
        threads = self.get("Experiment", "threads")
        try:
            a = int(threads)
            self.__assert(a > 0,
                          'Value of the "threads" option must be positive')
        except ValueError:
            self.__assert(False,
                          'Value of the "threads" option must be a positive integer')
        # rubricator
        self.__check_option_entry("Experiment", "rubricator")
        self.__check_value("Experiment", "rubricator", ["subj", "ipv", "rgnti"])

    def validate_classification(self):
        self.__check_section_existance("Classification")
        for model in self.get_as_list("Classification", "models"):
            self.__assert(model in self.map_config.options("SupportedModels"),
                          'Model "{}" is not supported'.format(model))
            self.__assert(model in self.sections(),
                          'No options specified for model "{}"'.format(model))
            path = self.map_config.get("SupportedModels", model)
            components = path.split(".")
            class_name = components[-1]
            module_name = ".".join(components[:-1])
            try:
                m = self.load_class(path)
                self.__assert(m is not None,
                              "Module {} cannot be loaded (noneType)".format(path))
                instance = m()
                params = instance.get_params()
                for hp in self.options(model):
                    self.__assert(hp in params.keys(),
                                  'Model "{}" has no hyperparameter "{}". {}'
                                  'Possible hyperparameters: {}'.format(model,
                                                                        hp,
                                                                        linesep,
                                                                        ", ".join(params)))
            except ImportError as e:
                self.__assert(False,
                              "Module {} cannot be loaded ({})".format(path, e))
            except AttributeError as ae:
                self.__assert(False,
                              'Module {} has no model "{}"'.format(module_name, class_name))

    def validate_preprocessing(self):
        # Tested somehow
        section = "Preprocessing"
        self.__check_section_existance(section)
        options = {"id", "title", "text",
                   "keywords", "subj", "ipv", "rgnti",
                   "correct", "remove_stopwords", "normalization",
                   "language", "batch_size", "kw_delim"}
        for key in options:
            self.__check_option_entry(section, key)
            value = self.get(section, key)
            self.__assert(value != "", 'Missing value of "{}" option'.format(key))
        batch_size = self.get("Preprocessing", "batch_size", fallback="")
        #
        self.__check_value(section, "remove_stopwords", [True, False])
        #
        normalization_options = self.smart_parse_list(self.map_config.get("Supported", "normalization"))
        self.__check_value(section, "normalization", normalization_options)
        #
        lang_options = self.smart_parse_list(self.map_config.get("Supported", "languages"))
        self.__check_value(section, "language", lang_options)
        #
        self.__assert(self.__is_int(batch_size) and int(batch_size) > 0,
                      "Invalid value of 'batch_size:'\n"
                      "Only positive integers are supported")

    def validate_word_embedding(self):
        section = "WordEmbedding"
        self.__check_section_existance(section)
        use_model = self.get(section, "use_model", fallback="")
        um_option_exists = use_model != ""
        if um_option_exists:
            reports = next(walk(join(dirname(__file__), "..", "reports")))[1]
            self.__assert(use_model in reports, "Cannot find model {}".format(use_model))
            for i in reports:
                files = next(walk(join(dirname(__file__), "..", "reports", i)))[2]
                for file in files:
                    if file.split(".")[-1] == "model":
                        print("W2V model '{}' found".format(use_model))
                        return
            options = {"vector_dim", "pooling"}
            for key in options:
                self.__check_option_entry(section, key)
                value = self.get(section, key)
                self.__assert(value != "", 'Missing value of "{}" option'.format(key))
            vector_dim = self.get(section, "vector_dim")
            #
            self.__assert(self.__is_int(vector_dim) and int(vector_dim) > 0,
                          "Invalid value of 'vector_dim:'\n"
                          "Only positive integers are supported")
            #
            pooling_options = self.smart_parse_list(self.map_config.get("Supported", "pooling"))
            self.__check_value(section, "pooling", pooling_options)
            print("W2V model {} not found, new model will be created".format(use_model))
        else:
            options = {"vector_dim", "pooling"}
            for key in options:
                self.__check_option_entry(section, key)
                value = self.get(section, key)
                self.__assert(value != "", 'Missing value of "{}" option'.format(key))
            vector_dim = self.get(section, "vector_dim")
            #
            self.__assert(self.__is_int(vector_dim) and int(vector_dim) > 0,
                          "Invalid value of 'vector_dim:'\n"
                          "Only positive integers are supported")
            #
            pooling_options = self.smart_parse_list(self.map_config.get("Supported", "pooling"))
            self.__check_value(section, "pooling", pooling_options)
            print("W2V model is not specified, new model will be created")

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

    def __assert(self, condition: bool, error_msg: str, leave=True):
        if not condition:
            print(error_msg)
            if leave:
                sys.exit(0)

    def __check_existance(self, section, option):
        self.__check_section_existance(section)
        self.__check_option_entry(section, option)

    def __check_section_existance(self, section: str):
        self.__assert(section in self.sections(),
                      'Missing "{}" section'.format(section))

    def __check_option_entry(self, section, option):
        self.__assert(option in self.options(section),
                      'Section "{}" is missing "{}" option'.format(section, option))

    def __check_value(self, section, option, supported: list):
        value = self.get_as_list(section, option)
        for i in value:
            self.__assert(i in supported,
                          'Value "{}" of the option "{}" is not supported.\n'
                          'Supported values: {}'.format(i, option, ", ".join(map(str, supported))))

    def load_class(self, classpath: str):
        components = classpath.split(".")
        classname = components[-1]
        module_name = ".".join(components[:-1])
        module = import_module(module_name)
        class_type = getattr(module, classname)
        return class_type

    def get_model_types(self):
        model_types = []
        for m in self.get_as_list("Experiment", "models"):
            path = self.map_config.get("SupportedModels", m)
            model_types.append(self.load_class(path))
        return model_types

    def get_model_type(self, model):
        path = self.map_config.get("SupportedModels", model)
        return self.load_class(path)

    def get_hyperparameters(self, model: str):
        hypers = {}
        for option in self.options(model):
            hp_list = self.get_as_list(model, option)
            if hp_list[0] == "":
                continue
            hypers[option] = self.get_as_list(model, option)
        return hypers

    def smart_parse_list(self, str_list: str):
        if "," not in str_list:
            if self.__is_int(str_list):
                return [int(str_list)]
            elif self.__is_float(str_list):
                return [float(str_list)]
            elif self.__is_bool(str_list):
                return [self.__str_to_bool(str_list)]
            return [str_list]
        parsed = []
        sublists = re.findall(self.list_re, str_list)
        if len(sublists) == 0:
            items = list(map(str.strip, str_list.split(",")))
            for i in items:
                if self.__is_int(i):
                    parsed.append(int(i))
                elif self.__is_float(i):
                    parsed.append(float(i))
                else:
                    parsed.append(i)
        else:
            for sl in sublists:
                parsed.append(self.smart_parse_list(sl[1:-1]))
        return parsed

    def get_as_dict(self, section: str) -> dict:
        res = {}
        for i in self.options(section):
            str_val = self.get(section, i)
            res[i] = self.smart_parse_list(str_val)
            if len(res[i]) == 1:
                res[i] = res[i][0]
        return res

    def __is_int(self, value: str):
        try:
            int(value)
            return True
        except ValueError:
            return False

    def __is_float(self, value: str):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def __is_bool(self, value: str):
        if value.lower() in ["true", "false"]:
            return True
        return False

    def __str_to_bool(self, value: str):
        if value.lower() == "false":
            return False
        return True


if __name__ == '__main__':
    config = ValidConfig()
    config.read(join(dirname(__file__), "..", "settings.ini"), encoding="cp1251")
    config.validate_dataset()
    config.validate_experiment()
    config.validate_preprocessing()
    config.validate_word_embedding()
    config.validate_classification()
