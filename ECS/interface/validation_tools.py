from importlib import import_module
import sys


# Функции для парсинга типов данных настроек
# (См. докстринг ValidConfig)

def is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_bool(value: str) -> bool:
    return value.lower() in ["true", "false"]


def is_plain_sequence(value: str) -> bool:
    if value.strip() == "" or "[" in value or "]" in value:
        return False
    # Есть хотя бы 1 непустое значение
    return any(list(map(lambda x: len(x.strip()) > 0, value.split(","))))


def is_nested_int_list(value: str) -> bool:
    op_br = value.count("[")
    cl_br = value.count("]")
    if (op_br * cl_br == 0) or (op_br != cl_br):
        return False
    return all(map(is_int, value.replace("[", "").replace("]", "").split(",")))


def parse_primitive(value: str):
    stripped = value.strip()
    if is_int(stripped):
        return int(stripped)
    elif is_float(stripped):
        return float(stripped)
    elif is_bool(stripped):
        return str_to_bool(stripped)
    return stripped


def parse_plain_sequence(value: str) -> list:
    if not is_plain_sequence(value):
        raise ValueError(f"'{value}' does not represent valid plain value sequence")
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if len(item) == 0:
            continue
        parsed.append(parse_primitive(item))
    return parsed


def parse_nested_int_list(value: str) -> list:
    if not is_nested_int_list(value):
        raise ValueError(f"'{value}' does not represent valid nested lists of integers")
    # TODO: небезопасно!
    return eval(f"[{value}]")


def str_to_bool(value: str) -> bool:
    if not is_bool(value):
        raise ValueError(f"'{value}' does not represent boolean value")
    return value.lower() == "true"


def load_class(classpath: str):
    components = classpath.split(".")
    class_name = components[-1]
    module_name = ".".join(components[:-1])
    module = import_module(module_name)
    class_type = getattr(module, class_name)
    return class_type


def val_assert(condition: bool, error_msg: str, leave=True) -> None:
    if not condition:
        print(error_msg)
        if leave:
            sys.exit(0)
