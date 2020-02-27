from configparser import ConfigParser
from os.path import join, dirname

from ECS.core.models.abstract_model import AbstractModel
from ECS.core.model_tools import load_class
from ECS.interface.validation_tools import is_int, is_float
from ECS.core.dataset import Dataset

from sklearn.model_selection import GridSearchCV, StratifiedKFold


class KerasModel(AbstractModel):
    PATH_TO_MAP_CONFIG = join(dirname(__file__), "..", "..", "interface", "map.ini")
    class_mapping = None
    opt_mapping = None

    def __init__(self):
        """
        Обертка для модели Кераса.
        :param layers - список текстовых описаний слоев:
        """
        super().__init__()
        self._init_mappings()

    def _init_mappings(self):
        if self.class_mapping is None:
            assert self.opt_mapping is None, \
                f"Inconsistent mapping state: {self.opt_mapping}"
            self.class_mapping = {}
            self.opt_mapping = {}
            map_config = ConfigParser()
            map_config.read(self.PATH_TO_MAP_CONFIG)
            layers_section = "SupportedLayers"
            for layer in map_config.options(layers_section):
                self.class_mapping[layer] = load_class(map_config.get(layers_section, layer))
            opt_section = "SupportedOptimizers"
            for opt in map_config.options(opt_section):
                self.opt_mapping[opt] = load_class(map_config.get(opt_section, opt))

    @staticmethod
    def _parse_layer(layer: str, last: bool) -> dict:
        res = {}
        parts = layer.split("_")
        res["layer_name"] = parts[0]
        if res["layer_name"] == "dropout":
            out_type = float
        else:
            out_type = int
        if len(parts) == 3:
            res["output_size"] = out_type(parts[1])
            res["activation"] = parts[2]
        elif len(parts) == 2:
            if last:
                res["output_size"] = None
                if is_int(parts[1] or is_float(parts[1])):
                    res["activation"] = None
                else:
                    res["activation"] = parts[1]
            else:
                res["output_size"] = out_type(parts[1])
                res["activation"] = None
        elif len(parts) == 1:
            res["output_size"] = None
            res["activation"] = None
        return res

    @staticmethod
    def _create_settings_dict(parsed: dict) -> dict:
        name = parsed["layer_name"]
        settings = {}
        if name == "dropout":
            settings["rate"] = parsed["output_size"]
        elif name in ["lstm", "dense", "gru"]:
            settings["units"] = parsed["output_size"]
            settings["activation"] = parsed["activation"]
        return settings

    @staticmethod
    def _is_recurrent(layer_name: str) -> bool:
        return layer_name in ["lstm", "gru"]

    def _create_layer(self, parsed: dict, last: bool,
                      first: bool, is_rnn_network: bool,
                      n_features: int, n_outputs: int):
        layer_name = parsed["layer_name"]
        layer_class = self.class_mapping[layer_name]
        settings = self._create_settings_dict(parsed)
        if layer_name != "dropout":
            if KerasModel._is_recurrent(layer_name):
                settings["return_sequences"] = True
            if first:
                if is_rnn_network:
                    # Батчи последовательностей неопределенной длины,
                    # состоящей из векторов [n_features]
                    settings["input_shape"] = (None, n_features)
                else:
                    # Батчи векторов
                    settings["input_shape"] = (n_features,)
            if last:
                if KerasModel._is_recurrent(layer_name):
                    settings["return_sequences"] = False
                settings["units"] = n_outputs
        new_layer = layer_class(**settings)
        if is_rnn_network and layer_name == "dense":
            new_layer = self.class_mapping["time_distributed"](new_layer)
        return new_layer

    def _build_from_layers(self, layers: list, n_features: int, n_outputs: int):
        Sequential = self.class_mapping["sequential"]
        instance = Sequential()
        parsed_layers = []
        rnn_present = False
        # Парсим сеть, проверяем наличие рекуррентных слоев
        for idx, layer_desc in enumerate(layers):
            parsed = KerasModel._parse_layer(layer_desc, last=(idx == len(layers) - 1))
            if KerasModel._is_recurrent(parsed["layer_name"]):
                rnn_present = True
            parsed_layers.append(parsed)
        # Настраиваем параметры, создаем слои и модель
        for idx, parsed in enumerate(parsed_layers):
            # assert parsed["output_size"], parsed
            new_layer = self._create_layer(parsed=parsed,
                                           last=(idx == len(parsed_layers) - 1),
                                           first=(idx == 0),
                                           is_rnn_network=rnn_present,
                                           n_features=n_features, n_outputs=n_outputs)
            instance.add(new_layer)
        return instance

    def create_model(self, layers: list, n_data_features: int, n_outputs: int,
                     loss: str, optimizer: str, lr: float):
        instance = self._build_from_layers(layers=layers,
                                           n_features=n_data_features,
                                           n_outputs=n_outputs)
        optimizer = self.opt_mapping[optimizer](lr=lr)
        instance.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return instance

    def run_grid_search(self, hyperparameters: dict,
                        n_inputs: int,
                        n_outputs: int,
                        dataset: Dataset,
                        n_folds: int,
                        n_jobs: int):
        scoring = 'f1_weighted'
        grid = None
        for model_descr in list(hyperparameters["models"]):
            param_grid = {
                "layers": [model_descr],
                "n_data_features": [n_inputs],
                "n_outputs": [n_outputs],
                "loss": list(hyperparameters["loss"]),
                "optimizer": list(hyperparameters["optimizer"]),
                "lr": [*[hyperparameters["learning_rate"]]],
                "epochs": list(hyperparameters["n_epochs"]),
            }
            model_wrapper = self.class_mapping["keras_classifier"](build_fn=self.create_model)
            skf = StratifiedKFold(shuffle=True, n_splits=n_folds)
            grid = GridSearchCV(estimator=model_wrapper,
                                param_grid=param_grid,
                                n_jobs=n_jobs,
                                # scoring=scoring,
                                # cv=skf,
                                verbose=0)
            grid.fit(x_train, y_train)
        return grid.best_params_


if __name__ == '__main__':
    import numpy as np
    hypers = {
        # "models": [["dense"], ["lstm_5", "dropout_0.4", "dense_tanh"]],
        "models": [["dense_20", "dropout_0.3", "dense"]],
        "optimizer": ["adam", "rmsprop"],
        "learning_rate": 0.01,
        "loss": ["mean_squared_error", "hinge"],
        "n_epochs": [10, 15],
    }
    x = np.random.rand(5, 100, 10)
    # y = np.random.randint(0, 4, (100, 1))
    y = np.zeros([5, 100, 4])
    for i in range(len(y)):
        y[i, np.random.randint(0, 4)] = 1
    m = KerasModel()
    bp = m.run_grid_search(hyperparameters=hypers, n_folds=5,
                           n_inputs=10, n_outputs=4, n_jobs=2, x_train=x, y_train=y)
    print(bp)
    # clf = m.create_model(layers=["dense_20", "dropout_0.3", "dense"],
    #                      n_data_features=10, n_outputs=4, loss="mean_squared_error",
    #                      optimizer="adam", lr=0.02)
    # print(clf.summary())
    # clf.fit(x, y)
    # print(clf.predict(np.random.rand(1, 10)))

"""
1.  Нужно разделить грид-серч для всех моделей,
    т.к. надо подбирать форму входных и выходных данных.
2.  Исправить валидацию кераса. Все гиперпараметры могут
    быть списками.
"""