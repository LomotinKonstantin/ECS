import warnings

warnings.filterwarnings("ignore")

from interface.valid_config import ValidConfig
from os.path import dirname, join, exists
from core.Worker import Worker
from datetime import datetime
from shutil import copyfile


if __name__ == '__main__':
    config = ValidConfig()
    config.read(join(dirname(__file__), "settings.ini"), encoding="cp1251")
    print("Validating experiment settings...")
    config.validate_dataset()
    config.validate_experiment()
    worker = Worker()
    #########
    worker.set_lang("ru")
    #########
    # Experiment title setup
    # If the title is not defined, current date is set up as a title
    title = config.get("Experiment", "experiment_title", fallback="")
    if title.strip() == "":
        title = datetime.today().strftime("%d-%b-%Y___%X")
        title = title.replace(":", "-")
    result_path = join(dirname(__file__), "reports", title)
    if not worker.set_res_folder(result_path):
        exit(0)
    data_folder = join(dirname(__file__), "datasets")
    dataset_folder = config.get("TrainingData", "dataset")
    train_file = join(data_folder, dataset_folder,
                      config.paths.get("DatasetFiles",
                                       dataset_folder))
    test_file = config.get("TrainingData", "test_file", fallback="")
    if test_file == "":
        train_percent = 1 - config.getint("TrainingData", "test_percent") / 100
        worker.load_data(train_file, split_ratio=train_percent)
    else:
        worker.load_data(train_file, test_path=test_file)
    w2v_file = join(dirname(__file__), "core", "w2v_models", "w2v_model_50_for_rgnti_update_k_v1_25_3_18.model")
    if not exists(w2v_file):
        print("Word2Vec model file is not found")
        exit(0)
    worker.load_w2v(w2v_file)
    bin_val = config.get("Experiment", "binary")
    binary = bin_val == "1" or bin_val == "true"
    n_folds = config.getint("Experiment", "n_folds")
    threads = config.getint("Experiment", "threads")
    for rubr in config.get_as_list("Experiment", "rubricator"):
        worker.set_rubr_id(rubr)
        for model in config.get_as_list("Experiment", "models"):
            print("Fitting parameters for model {} by {}".format(model, rubr))
            ModelType = config.get_model_type(model)
            instance = ModelType()
            hypers = config.get_hyperparameters(model)
            worker.search_for_clf(model=instance,
                                  parameters=hypers,
                                  jobs=threads,
                                  OneVsAll=binary,
                                  skf_folds=n_folds)
    copyfile(join(dirname(__file__), "settings.ini"), join(result_path, "setting.ini"))
