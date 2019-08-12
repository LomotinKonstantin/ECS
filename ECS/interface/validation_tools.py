from importlib import import_module
import re
import sys


LIST_RE = r"\[.*?\]"


def is_int(value: str):
    try:
        int(value)
        return True
    except ValueError:
        return False


def is_float(value: str):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_bool(value: str):
    if value.lower() in ["true", "false"]:
        return True
    return False


def str_to_bool(value: str):
    if value.lower() == "false":
        return False
    return True


def load_class(classpath: str):
    components = classpath.split(".")
    class_name = components[-1]
    module_name = ".".join(components[:-1])
    module = import_module(module_name)
    class_type = getattr(module, class_name)
    return class_type


def val_assert(condition: bool, error_msg: str, leave=True):
    if not condition:
        print(error_msg)
        if leave:
            sys.exit(0)


def smart_parse_value(str_list: str):
    """
    Парсит числовые и булевские значения,
    а также списки любой вложенности с любым содержимым.
    Принимаются списки в формате ini, то есть значения через запятую, без скобок.
    Вложенные списки должны быть заключены в [].
    :param str_list: строка со значением
    :return: распарсенное значение
    """
    if "," not in str_list:
        if is_int(str_list):
            return [int(str_list)]
        elif is_float(str_list):
            return [float(str_list)]
        elif is_bool(str_list):
            return [str_to_bool(str_list)]
        return [str_list]
    parsed = []
    sublists = re.findall(LIST_RE, str_list)
    if len(sublists) == 0:
        items = list(map(str.strip, str_list.split(",")))
        for i in items:
            if is_int(i):
                parsed.append(int(i))
            elif is_float(i):
                parsed.append(float(i))
            else:
                parsed.append(i)
    else:
        for sl in sublists:
            parsed.append(smart_parse_value(sl[1:-1]))
    return parsed
