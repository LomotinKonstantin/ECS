import os

from ECS.core.data_tools import \
    find_cached_matrices, \
    load_w2v, \
    find_cache_for_source, \
    find_cached_w2v, \
    create_reading_matrix_gen, \
    recognize_language, \
    create_clear_generators, \
    timestamp, \
    extract_pp_settings, \
    create_w2v, \
    generate_matrix_cache_path, \
    caching_matrix_generator, \
    aggregate_full_dataset_with_pooling, \
    generate_clear_cache_path, \
    create_labeled_tt_split, \
    df_to_labeled_dataset, \
    count_generator_items
from ECS.interface.logging_tools import get_logger, error_ps
from ECS.interface.valid_config import ValidConfig


class Dataset:

    DATA_COL = "features"

    def __init__(self, config: ValidConfig):

        training_file = config.get("TrainingData", "dataset")
        self.test_file = config.get("TrainingData", "test_file")
        use_model = config.get("WordEmbedding", "use_model")
        remove_stopwords = config.get_primitive("Preprocessing", "remove_stopwords")
        normalization = config.get_primitive("Preprocessing", "normalization")
        self.chunk_size = config.get_primitive("Preprocessing", "batch_size")
        exp_title = config.get_primitive("Experiment", "experiment_title")
        vector_dim = config.getint("WordEmbedding", "vector_dim")
        window = config.getint("WordEmbedding", "window")
        self.pooling = config.get("WordEmbedding", "pooling")
        if exp_title == "":
            exp_title = timestamp()
            config.set("Experiment", "experiment_title", exp_title)
        w2v_exists = os.path.exists(use_model)

        self.base_dir = os.path.dirname(training_file)
        self.logger = get_logger("Dataset Manager")
        self.w2v_model = None
        self.test_matrices_cached = False
        self.train_matrices_cached = False
        self.test_file_available = os.path.exists(self.test_file)

        self.test_percent = 0
        if not self.test_file_available:
            test_percent = config.getint("TrainingData", "test_percent")
            self.test_percent = test_percent / 100

        # Определяем язык
        self.language = config.get("Preprocessing", "language")
        if self.language == "auto":
            # Для распознавания языка грузим 10 первых строк
            self.language = recognize_language(training_file, encoding="cp1251", n_lines=10)
            # Потом мы будем копировать файл настроек в кэш
            # Поэтому поддерживаем его актуальным
            config.set("Preprocessing", "language", self.language)

        # Готовим фильтры настроек для поиска кэша
        clear_metadata_filter = {
            "language": self.language,
            "remove_stopwords": remove_stopwords,
            "normalization": normalization
        }
        self.matrix_metadata_filter = {
            **clear_metadata_filter,
        }
        for key in ["vector_dim", "window"]:
            self.matrix_metadata_filter[key] = config.get_primitive("WordEmbedding", key)

        # Создаем источники матрицы согласно схеме
        cached_matrices = find_cached_matrices(base_dir=self.base_dir,
                                               metadata_filter=self.matrix_metadata_filter)
        train_test_matr_exist = False
        cached_w2v_path = ""
        train_matr_cache = ""
        test_matr_cache = ""
        matrix_gens = {}
        # Проводим разведку
        if len(cached_matrices) > 0:
            matrix_cache_folder, matrix_cache_files = cached_matrices.popitem()
            train_matr_cache = find_cache_for_source(matrix_cache_files, training_file)
            test_matr_cache = None
            if self.test_file_available:
                test_matr_cache = find_cache_for_source(matrix_cache_files, self.test_file)
            train_test_matr_exist = train_matr_cache and test_matr_cache
            cached_w2v_path = find_cached_w2v(matrix_cache_folder)
        if train_test_matr_exist and cached_w2v_path and use_model != "":
            self.logger.info("Cached vectors found")
            w2v_model, language = load_w2v(cached_w2v_path)
            config.set("Preprocessing", "language", language)
            matrix_gens["train"] = create_reading_matrix_gen(train_matr_cache, self.chunk_size)
            matrix_gens["test"] = create_reading_matrix_gen(test_matr_cache, self.chunk_size)
        else:
            # Либо нет готовых обучающих матриц,
            # либо в кэше нет модели W2V,
            # либо нужно создать их заново при помощи указанной модели
            # Получаем источники чистых текстов
            # Словарь вида {'train': генератор, ['test': генератор]}
            pp_gens = create_clear_generators(
                base_dir=self.base_dir,
                clear_filter=clear_metadata_filter,
                chunk_size=self.chunk_size,
                training_fpath=training_file,
                test_fpath=self.test_file,
                experiment_title=exp_title,
                pp_params=extract_pp_settings(config)
            )
            if w2v_exists:
                # Используем ее
                # И обновляем язык
                self.logger.info(f"Using Word2Vec model: {os.path.basename(use_model)}")
                self.w2v_model, self.language = load_w2v(use_model)
                config.set("Preprocessing", "language", self.language)
            else:
                # Создаем новую
                # Для совместимости преобразуем в список
                self.logger.info("Creating new Word2Vec model")
                self.w2v_model = create_w2v(pp_sources=list(pp_gens.values()),
                                            vector_dim=vector_dim,
                                            window_size=window)
                # Генераторы - это одноразовые итераторы
                # Придется создать их заново
                # Должны гарантированно получиться
                # читающие,а не кэширующие
                pp_gens = create_clear_generators(
                    base_dir=self.base_dir,
                    clear_filter=clear_metadata_filter,
                    chunk_size=self.chunk_size,
                    training_fpath=training_file,
                    test_fpath=self.test_file,
                    experiment_title=exp_title,
                    pp_params=extract_pp_settings(config)
                )

            # Переходим к матрицам

            self.train_matrix_cache_path = generate_matrix_cache_path(raw_path=training_file,
                                                                      exp_name=exp_title)
            try:
                train_matr_gen = caching_matrix_generator(pp_source=pp_gens["train"],
                                                          w2v_file=self.w2v_model,
                                                          cache_path=self.train_matrix_cache_path,
                                                          pp_metadata=clear_metadata_filter)
            except Exception as e:
                error_ps(self.logger,
                         f"Error occurred during creation caching vector generator (training): {e}")
            else:
                matrix_gens["train"] = train_matr_gen
            self.test_matrix_cache_path = ""
            self.test_clear_cache_path = ""
            if self.test_file_available:
                self.test_matrix_cache_path = generate_matrix_cache_path(raw_path=self.test_file,
                                                                         exp_name=exp_title)
                self.test_clear_cache_path = generate_clear_cache_path(raw_path=self.test_file,
                                                                       exp_name=exp_title)
                test_vec_gen = caching_matrix_generator(pp_source=pp_gens["test"],
                                                        w2v_file=self.w2v_model,
                                                        cache_path=self.test_matrix_cache_path,
                                                        pp_metadata=clear_metadata_filter)
                matrix_gens["test"] = test_vec_gen
        self.matrix_gens = matrix_gens
        self.train_clear_cache_path = generate_clear_cache_path(training_file, exp_title)
        self.train_df = None
        self.test_df = None
        self.train_size = -1
        self.test_size = -1

    def train_matrix_generator(self, chunk_size=None):
        # После создания датасета в словаре лежит читающий либо кэширующий генератор
        # Поэтому первый раз всегда возвращаем его
        # А в последующие вызовы создаем читающий
        if chunk_size is None:
            chunk_size = self.chunk_size
        if self.train_matrices_cached:
            self.matrix_gens["train"] = create_reading_matrix_gen(path=self.train_matrix_cache_path,
                                                                  chunk_size=chunk_size)
        self.train_matrices_cached = True
        return self.matrix_gens["train"]

    def test_matrix_generator(self, chunk_size=None):
        if not self.test_file_available:
            self.logger.error("Unable to create test matrix generator (test file is not provided)")
            raise FileNotFoundError("No test file available")
        if chunk_size is None:
            chunk_size = self.chunk_size
        if self.test_matrices_cached:
            self.matrix_gens["test"] = create_reading_matrix_gen(path=self.test_matrix_cache_path,
                                                                 chunk_size=chunk_size)
        self.test_matrices_cached = True
        return self.matrix_gens["test"]

    def get_train_clear_cache_path(self):
        return self.train_clear_cache_path

    def get_train_matrix_cache_path(self):
        return self.train_matrix_cache_path

    def get_w2v_model(self):
        return self.w2v_model

    def get_language(self):
        return self.language

    def is_test_file_available(self):
        return self.test_file_available

    def get_matrix_md_filter(self):
        return self.matrix_metadata_filter

    def sklearn_dataset_split(self, rubricator: str) -> tuple:
        if self.train_df is None:
            self.train_df = aggregate_full_dataset_with_pooling(self.train_matrix_generator(),
                                                                self.pooling)
        if self.test_file_available:
            if self.test_df is None:
                self.test_df = aggregate_full_dataset_with_pooling(self.test_matrix_generator(),
                                                                   self.pooling)
            x_train, y_train = df_to_labeled_dataset(full_df=self.train_df, rubricator=rubricator)
            x_test, y_test = df_to_labeled_dataset(full_df=self.test_df, rubricator=rubricator)
        else:
            x_train, x_test, y_train, y_test = create_labeled_tt_split(full_df=self.train_df,
                                                                       test_percent=self.test_percent,
                                                                       rubricator=rubricator)
        self.train_size = len(x_train) + len(y_train)
        self.test_size = len(x_test) + len(y_test)
        return x_train, x_test, y_train, y_test

    def __init_sizes(self):
        if self.train_size == -1:
            # Если размер еще не известен
            samples_in_train_file = count_generator_items(self.train_matrix_generator())
            if self.test_file_available:
                self.train_size = samples_in_train_file
                self.test_size = count_generator_items(self.test_matrix_generator())
            else:
                self.test_size = int(samples_in_train_file * self.test_percent)
                self.train_size = samples_in_train_file - self.test_size

    def __infinite_row_df_generator(self, df, label_col: str, from_idx=0, to_idx=None):
        while True:
            for idx in df.index[from_idx:to_idx]:
                matr = df.loc[idx, self.DATA_COL]
                label = df.loc[idx, label_col]
                yield matr, label

    def __infinite_gen_matrix_generator(self, gen_func, label_col: str, skip_lines=0, to_idx=None):
        while True:
            cur_idx = 0
            gen = gen_func(chunk_size=1)
            for chunk in gen:
                if to_idx is not None and cur_idx > to_idx:
                    break
                for _ in range(skip_lines):
                    next(gen)
                row = chunk.index[0]
                matr = chunk.loc[row, self.DATA_COL]
                label = chunk.loc[row, label_col]
                yield matr, label
                cur_idx += 1

    def keras_infinite_train_generator(self, rubricator: str):
        """
        Генератор обучающих пар. Умеет использовать данные, уже загруженные в память
        :return: генератор возвращает пары (матрица, метка_класса) по одному сэмплу,
                 так как в матрицах разное количество строк, и они не стакаются
        """
        self.__init_sizes()
        if self.test_file_available:
            return self.__infinite_gen_matrix_generator(gen_func=self.train_matrix_generator,
                                                        label_col=rubricator)
        else:
            last_idx = self.train_size - 1
            return self.__infinite_gen_matrix_generator(gen_func=self.train_matrix_generator,
                                                        label_col=rubricator, to_idx=last_idx)

    def keras_test_generator(self, rubricator: str):
        """
        :return: генератор возвращает пары (матрица, метка_класса) по одному сэмплу,
                 так как в матрицах разное количество строк, и они не стакаются
        """
        self.__init_sizes()
        if self.test_file_available:
            return self.__infinite_gen_matrix_generator(gen_func=self.test_matrix_generator,
                                                        label_col=rubricator)
        else:
            return self.__infinite_gen_matrix_generator(gen_func=self.train_matrix_generator,
                                                        label_col=rubricator,
                                                        skip_lines=self.train_size)
