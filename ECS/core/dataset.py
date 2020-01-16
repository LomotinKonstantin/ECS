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
    caching_vector_generator
from ECS.interface.logging_tools import get_logger, error_ps
from ECS.interface.valid_config import ValidConfig


class Dataset:

    def __init__(self, config: ValidConfig):

        training_file = config.get("TrainingData", "dataset")
        test_file = config.get("TrainingData", "test_file")
        use_model = config.get("WordEmbedding", "use_model")
        remove_stopwords = config.get_primitive("Preprocessing", "remove_stopwords")
        normalization = config.get_primitive("Preprocessing", "normalization")
        chunk_size = config.get_primitive("Preprocessing", "batch_size")
        exp_title = config.get_primitive("Experiment", "experiment_title")
        vector_dim = config.getint("WordEmbedding", "vector_dim")
        window = config.getint("WordEmbedding", "window")
        pooling = config.get("WordEmbedding", "pooling")
        if exp_title == "":
            exp_title = timestamp()
        w2v_exists = os.path.exists(use_model)

        self.base_dir = os.path.dirname(training_file)
        self.logger = get_logger("Dataset-INITIALIZATION")

        test_percent = 0
        if not test_file:
            test_percent = config.getint("TrainingData", "test_percent")
            test_percent = test_percent / 100

        # Определяем язык
        language = config.get("Preprocessing", "language")
        if language == "auto":
            # Для распознавания языка грузим 10 первых строк
            language = recognize_language(training_file, encoding="cp1251", n_lines=10)
            # Потом мы будем копировать файл настроек в кэш
            # Поэтому поддерживаем его актуальным
            config.set("Preprocessing", "language", language)

        # Готовим фильтры настроек для поиска кэша
        clear_metadata_filter = {
            "language": language,
            "remove_stopwords": remove_stopwords,
            "normalization": normalization
        }
        vector_metadata_filter = {
            **clear_metadata_filter,
        }
        for key in ["vector_dim", "window", "pooling"]:
            vector_metadata_filter[key] = config.get_primitive("WordEmbedding", key)

        # Создаем источники векторов согласно схеме
        cached_vectors = find_cached_vectors(base_dir=self.base_dir,
                                             metadata_filter=vector_metadata_filter)
        train_test_vec_exist = False
        cached_w2v_path = ""
        vector_cache_path = ""
        train_vec_cache = ""
        test_vec_cache = ""
        vector_gens = {}
        # Проводим разведку
        if len(cached_vectors) > 0:
            vector_cache_folder, vector_cache_files = cached_vectors.popitem()
            train_vec_cache = find_cache_for_source(vector_cache_files, training_file)
            test_vec_cache = None
            if test_file != "":
                test_vec_cache = find_cache_for_source(vector_cache_files, test_file)
            train_test_vec_exist = train_vec_cache and test_vec_cache
            cached_w2v_path = find_cached_w2v(vector_cache_folder)
        if train_test_vec_exist and cached_w2v_path and use_model != "":
            self.logger.info("Cached vectors found")
            w2v_model, language = load_w2v(cached_w2v_path)
            config.set("Preprocessing", "language", language)
            vector_gens["train"] = create_reading_vector_gen(train_vec_cache, chunk_size)
            vector_gens["test"] = create_reading_vector_gen(test_vec_cache, chunk_size)
        else:
            # Либо нет готовых обучающих векторов,
            # либо в кэше нет модели W2V,
            # либо нужно создать их заново при помощи указанной модели
            # Получаем источники чистых текстов
            # Словарь вида {'train': генератор, ['test': генератор]}
            pp_gens = create_clear_generators(
                base_dir=self.base_dir,
                clear_filter=clear_metadata_filter,
                chunk_size=chunk_size,
                training_fpath=training_file,
                test_fpath=test_file,
                experiment_title=exp_title,
                pp_params=extract_pp_settings(config)
            )
            if w2v_exists:
                # Используем ее
                # И обновляем язык
                self.logger.info(f"Using Word2Vec model: {os.path.basename(use_model)}")
                w2v_model, language = load_w2v(use_model)
                config.set("Preprocessing", "language", language)
            else:
                # Создаем новую
                # Для совместимости преобразуем в список
                self.logger.info("Creating new Word2Vec model")
                w2v_model = create_w2v(pp_sources=list(pp_gens.values()),
                                       vector_dim=vector_dim,
                                       window_size=window)
                # Увы, генераторы - это одноразовые итераторы
                # Придется создать их заново
                # Должны гарантированно получиться
                # читающие,а не кэширующие
                pp_gens = create_clear_generators(
                    base_dir=self.base_dir,
                    clear_filter=clear_metadata_filter,
                    chunk_size=chunk_size,
                    training_fpath=training_file,
                    test_fpath=test_file,
                    experiment_title=exp_title,
                    pp_params=extract_pp_settings(config)
                )
            vector_cache_path = generate_vector_cache_path(raw_path=training_file, exp_name=exp_title)
            try:
                train_vec_gen = caching_vector_generator(pp_source=pp_gens["train"],
                                                         w2v_file=w2v_model,
                                                         cache_path=vector_cache_path,
                                                         conv_type=pooling,
                                                         pp_metadata=clear_metadata_filter)
            except Exception as e:
                error_ps(self.logger,
                         f"Error occurred during creation caching vector generator (training): {e}")
            else:
                vector_gens["train"] = train_vec_gen
            if test_file:
                vector_cache_path = generate_vector_cache_path(raw_path=test_file, exp_name=exp_title)
                test_vec_gen = caching_vector_generator(pp_source=pp_gens["test"],
                                                        w2v_file=w2v_model,
                                                        cache_path=vector_cache_path,
                                                        conv_type=pooling,
                                                        pp_metadata=clear_metadata_filter)
                vector_gens["test"] = test_vec_gen
        self.vector_gens = vector_gens
