from configparser import ConfigParser
from os.path import join, dirname
import logging

from ECS.core.models.abstract_model import AbstractModel
from ECS.core.model_tools import load_class
from ECS.interface.validation_tools import is_int, is_float
from ECS.core.data_tools import apply_pooling
from ECS.core.dataset import Dataset

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
import numpy as np


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
        self.logger = logging.getLogger("Keras Model")
        self.instance = None

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
            # Если слой описан полностью
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
            if self._is_recurrent(layer_name):
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
                if self._is_recurrent(layer_name):
                    settings["return_sequences"] = False
                settings["units"] = n_outputs
        new_layer = layer_class(**settings)
        if is_rnn_network and layer_name == "dense":
            new_layer = self.class_mapping["time_distributed"](new_layer)
        return new_layer

    def _build_from_layers(self, layers: list, n_features: int, n_outputs: int) -> tuple:
        Sequential = self.class_mapping["sequential"]
        instance = Sequential()
        parsed_layers = []
        rnn_present = False
        # Парсим сеть, проверяем наличие рекуррентных слоев
        for idx, layer_desc in enumerate(layers):
            parsed = self._parse_layer(layer_desc, last=(idx == len(layers) - 1))
            if self._is_recurrent(parsed["layer_name"]):
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
        return instance, rnn_present

    def create_model(self, layers: list, n_data_features: int, n_outputs: int,
                     loss: str, optimizer: str, lr: float):
        instance, is_recurrent = self._build_from_layers(layers=layers,
                                                         n_features=n_data_features,
                                                         n_outputs=n_outputs)
        optimizer = self.opt_mapping[optimizer](lr=lr)
        instance.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return instance, is_recurrent

    @staticmethod
    def data_transformer(matrices, labels,
                         is_recurrent: bool,
                         pooling_type: str,
                         dataset: Dataset,
                         rubricator: str,
                         n_epochs: int):
        for _ in range(n_epochs):
            if is_recurrent:
                matrices = map(lambda matr: apply_pooling(matr, pooling_type), matrices)
            labels = map(lambda lab: dataset.oh_encode(lab, rubricator), labels)
            for matrix, label in zip(matrices, labels):
                yield np.reshape(matrix, (1, -1)), np.reshape(label, (1, -1))

    def evaluate_model(self, model, x_test, y_test, dataset: Dataset,
                       pooling_type: str, rubricator: str, is_recurrent: bool):
        data_gen = self.data_transformer(x_test, y_test,
                                         is_recurrent=is_recurrent,
                                         pooling_type=pooling_type,
                                         dataset=dataset,
                                         rubricator=rubricator,
                                         n_epochs=1)
        y_true = []
        y_pred = []
        for tensor, label_vec in data_gen:
            y_true.append(dataset.oh_decode(label_vec[0], rubricator))
            prediction = model.predict(tensor)[0]
            y_pred.append(dataset.oh_decode(prediction, rubricator))
        return f1_score(y_true, y_pred, average="weighted")

    def run_grid_search(self, hyperparameters: dict,
                        x_train, y_train,
                        **kwargs) -> dict:
        conv_type = kwargs["conv_type"]
        # TODO: Костыль. Надо либо передавать dataset, либо выборки
        dataset = kwargs["dataset"]
        rubricator = kwargs["rubricator"]
        x_test = kwargs["x_test"]
        y_test = kwargs["y_test"]
        n_inputs = kwargs["n_inputs"]
        n_outputs = kwargs["n_outputs"]
        steps_per_epoch = len(y_train)
        best_score = -1
        best_params = {}
        for model_descr in list(hyperparameters["models"]):
            param_dict = {
                "layers": [model_descr],
                "n_data_features": [n_inputs],
                "n_outputs": [n_outputs],
                "loss": list(hyperparameters["loss"]),
                "optimizer": list(hyperparameters["optimizer"]),
                "lr": [*[hyperparameters["learning_rate"]]],
                "epochs": list(hyperparameters["n_epochs"]),
            }
            param_grid = ParameterGrid(param_dict)
            epochs = param_dict.pop("epochs")
            for param_combination in param_grid:  # type: dict
                estimator, is_recurrent = self.create_model(**param_combination)
                self.logger.info(estimator.summary())
                for n_epochs in epochs:
                    train_gen = self.data_transformer(x_train, y_train,
                                                      is_recurrent=is_recurrent,
                                                      pooling_type=conv_type,
                                                      dataset=dataset,
                                                      rubricator=rubricator,
                                                      n_epochs=n_epochs)
                    estimator.fit_generator(train_gen,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=n_epochs)
                    score = self.evaluate_model(estimator, x_test, y_test,
                                                dataset=dataset,
                                                pooling_type=conv_type,
                                                rubricator=rubricator,
                                                is_recurrent=is_recurrent)
                    if score > best_score:
                        best_params = param_combination
                        best_params["epochs"] = n_epochs
                        best_score = score
                    dict_str = "\n".join([f"{k}: {v}" for k, v in param_combination.items()])
                    log_msg = f"F1 score for \n{dict_str}\nwith {n_epochs} epochs: {round(score, 5)}"
                    self.logger.info(log_msg)
        return best_params

    def refit(self, **kwargs):
        best_params = kwargs["best_params"]
        x_train = kwargs["x_train"]
        y_train = kwargs["y_train"]
        conv_type = kwargs["conv_type"]
        dataset = kwargs["dataset"]
        rubricator = kwargs["rubricator"]
        n_epochs = best_params.pop("epochs")
        estimator, is_recurrent = self.create_model(**best_params)
        steps_per_epoch = len(y_train)
        train_gen = self.data_transformer(x_train, y_train,
                                          is_recurrent=is_recurrent,
                                          pooling_type=conv_type,
                                          dataset=dataset,
                                          rubricator=rubricator,
                                          n_epochs=n_epochs)
        estimator.fit_generator(train_gen,
                                steps_per_epoch=steps_per_epoch,
                                epochs=n_epochs)
        self.instance = estimator


if __name__ == '__main__':
    import numpy as np

    hypers = {
        "models": [["dense"], ["lstm_5", "dropout_0.4", "dense_tanh"]],
        # "models": [["dense_20", "dropout_0.3", "dense"]],
        "optimizer": ["adam", "rmsprop"],
        "learning_rate": [0.01],
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
                           n_inputs=10, n_outputs=4, n_jobs=2,
                           x_train=x, y_train=y)
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
    
    Почему dataset:
        1. Для каждой модели нужна своя форма данных - матрицы или векторы
    Почему x_train, y_train: 
        1. Юниформность
        2. Уже реализован фильтр маленьких рубрик. Для генератора это сделать сложно.
        3. GridSearch не умеет фитить генераторы
        
    Компромисс:
        1. Ручной грид серч на генераторах и костылях
        2. Генераторы на отфильтрованных x_train, y_train - применяют пулинг, делают форму
        
        * Сохраняется единый интерфейс
        * Полиморфно создается правильный источник данных
        * Данные уже отфильтрованы
        
    Как использовать Dataset в *Model?
"""
