import os

from ECS.core.data_tools import \
    find_cached_vectors, \
    load_w2v, \
    find_cache_for_source, \
    find_cached_w2v, \
    create_reading_vector_gen, \
    recognize_language, \
    create_clear_generators, \
    timestamp, \
    extract_pp_settings, \
    create_w2v, \
    generate_vector_cache_path, \
    caching_vector_generator, \
    aggregate_full_dataset, \
    generate_clear_cache_path, \
    create_labeled_tt_split, \
    df_to_labeled_dataset
from ECS.interface.logging_tools import get_logger, error_ps
from ECS.interface.valid_config import ValidConfig


class Dataset:

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
        pooling = config.get("WordEmbedding", "pooling")
        if exp_title == "":
            exp_title = timestamp()
        w2v_exists = os.path.exists(use_model)

        self.base_dir = os.path.dirname(training_file)
        self.logger = get_logger("Dataset-INITIALIZATION")
        self.w2v_model = None
        self.test_vectors_cached = False
        self.train_vectors_cached = False
        self.test_file_available = bool(self.test_file)

        # Заплатка
        # TODO: починить
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
        self.vector_metadata_filter = {
            **clear_metadata_filter,
        }
        for key in ["vector_dim", "window", "pooling"]:
            self.vector_metadata_filter[key] = config.get_primitive("WordEmbedding", key)

        # Создаем источники векторов согласно схеме
        cached_vectors = find_cached_vectors(base_dir=self.base_dir,
                                             metadata_filter=self.vector_metadata_filter)
        train_test_vec_exist = False
        cached_w2v_path = ""
        train_vec_cache = ""
        test_vec_cache = ""
        vector_gens = {}
        # Проводим разведку
        if len(cached_vectors) > 0:
            vector_cache_folder, vector_cache_files = cached_vectors.popitem()
            train_vec_cache = find_cache_for_source(vector_cache_files, training_file)
            test_vec_cache = None
            if self.test_file_available:
                test_vec_cache = find_cache_for_source(vector_cache_files, self.test_file)
            train_test_vec_exist = train_vec_cache and test_vec_cache
            cached_w2v_path = find_cached_w2v(vector_cache_folder)
        if train_test_vec_exist and cached_w2v_path and use_model != "":
            self.logger.info("Cached vectors found")
            w2v_model, language = load_w2v(cached_w2v_path)
            config.set("Preprocessing", "language", language)
            vector_gens["train"] = create_reading_vector_gen(train_vec_cache, self.chunk_size)
            vector_gens["test"] = create_reading_vector_gen(test_vec_cache, self.chunk_size)
        else:
            # Либо нет готовых обучающих векторов,
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
                # Увы, генераторы - это одноразовые итераторы
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

            self.train_vector_cache_path = generate_vector_cache_path(raw_path=training_file,
                                                                      exp_name=exp_title)
            try:
                train_vec_gen = caching_vector_generator(pp_source=pp_gens["train"],
                                                         w2v_file=self.w2v_model,
                                                         cache_path=self.train_vector_cache_path,
                                                         conv_type=pooling,
                                                         pp_metadata=clear_metadata_filter)
            except Exception as e:
                error_ps(self.logger,
                         f"Error occurred during creation caching vector generator (training): {e}")
            else:
                vector_gens["train"] = train_vec_gen
            self.test_vector_cache_path = ""
            self.test_clear_cache_path = ""
            if self.test_file_available:
                self.test_vector_cache_path = generate_vector_cache_path(raw_path=self.test_file,
                                                                         exp_name=exp_title)
                self.test_clear_cache_path = generate_clear_cache_path(raw_path=self.test_file,
                                                                       exp_name=exp_title)
                test_vec_gen = caching_vector_generator(pp_source=pp_gens["test"],
                                                        w2v_file=self.w2v_model,
                                                        cache_path=self.test_vector_cache_path,
                                                        conv_type=pooling,
                                                        pp_metadata=clear_metadata_filter)
                vector_gens["test"] = test_vec_gen
        self.vector_gens = vector_gens
        self.train_clear_cache_path = generate_clear_cache_path(training_file, exp_title)
        self.train_df = None
        self.test_df = None

    def train_vector_generator(self):
        if self.train_vectors_cached:
            self.vector_gens["train"] = create_reading_vector_gen(self.train_vector_cache_path,
                                                                  self.chunk_size)
        self.train_vectors_cached = True
        return self.vector_gens["train"]

    def test_vector_generator(self):
        if not self.test_vector_cache_path:
            self.logger.error("No test cache found")
            raise FileNotFoundError("No test cache found")
        if self.test_vectors_cached:
            self.vector_gens["test"] = create_reading_vector_gen(self.test_vector_cache_path,
                                                                 self.chunk_size)
        self.test_vectors_cached = True
        return self.vector_gens["test"]

    def aggregate_test_dataset(self):
        return aggregate_full_dataset(self.test_vector_generator())

    def aggregate_train_dataset(self):
        return aggregate_full_dataset(self.train_vector_generator())

    def train_clear_cache_path(self):
        return self.train_clear_cache_path

    def train_vector_cache_path(self):
        return self.train_vector_cache_path

    def w2v_model(self):
        return self.w2v_model

    def language(self):
        return self.language

    def aggregated_data_split(self, rubricator: str) -> tuple:
        if self.train_df is None:
            self.train_df = aggregate_full_dataset(self.train_vector_generator())
        if not self.test_file_available:
            # Если нет тестового файла
            x_train, x_test, y_train, y_test = create_labeled_tt_split(full_df=self.train_df,
                                                                       test_percent=self.test_percent,
                                                                       rubricator=rubricator)
        else:
            self.test_df = aggregate_full_dataset(self.test_vector_generator())
            x_train, y_train = df_to_labeled_dataset(full_df=self.train_df, rubricator=rubricator)
            x_test, y_test = df_to_labeled_dataset(full_df=self.test_df, rubricator=rubricator)
        return x_train, x_test, y_train, y_test

    def test_file_available(self):
        return self.test_file_available

