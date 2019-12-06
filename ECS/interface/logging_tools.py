import logging
import sys

from psutil import cpu_percent, virtual_memory


def create_logger(fname: str, name: str) -> logging.Logger:
    """
    Инициализация логгера
    :param fname: путь к файлу логов
    :param name: имя логгера
    :return: логгер
    """
    logging.basicConfig(filename=fname,
                        format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
                        level=logging.ERROR, filemode="w")
    return get_logger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Получить логгер с указанным именем
    Сначала должен быть вызвана функция create_logger()
    :param name: имя логгера
    :return: логгер
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        return logger


def ps_state_str() -> str:
    cpu_str = " | ".join([f"cpu{n}: {i}%" for n, i in enumerate(cpu_percent(percpu=True))])
    cpu_str = f"[{cpu_str}]"
    mem_state = virtual_memory()
    mem_str = f"{mem_state.used >> 20}Mb ({mem_state.percent}%)"
    return f"(CPU usage: {cpu_str}, memory usage: {mem_str})"


def info_ps(logger: logging.Logger, msg: str) -> None:
    logger.info(f"{msg} {ps_state_str()}")


def warn_ps(logger: logging.Logger, msg: str) -> None:
    logger.warning(f"{msg} {ps_state_str()}")


def error_ps(logger: logging.Logger, msg: str) -> None:
    logger.error(f"{msg} {ps_state_str()}")
