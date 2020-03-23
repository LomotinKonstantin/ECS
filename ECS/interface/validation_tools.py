import warnings
warnings.filterwarnings("ignore")
import sys

from ECS.interface.logging_tools import get_logger

val_logger = get_logger("ecs.validation")


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


def parse_nested_list(val: str, lower=False):
    if val == "":
        return [[""]]
    list_val = list(val)
    idx = 0
    obj_found = False
    while idx < len(list_val):
        sym = list_val[idx]
        if sym not in "[], ":
            if not obj_found:
                list_val.insert(idx, '"')
                idx += 1
                obj_found = True
        else:
            if obj_found:
                list_val.insert(idx, '"')
                idx += 1
                obj_found = False
        idx += 1
    if obj_found:
        list_val.append('"')
    return eval("".join(list_val))


def str_to_bool(value: str) -> bool:
    if not is_bool(value):
        raise ValueError(f"'{value}' does not represent boolean value")
    return value.lower() == "true"


def val_assert(condition: bool, error_msg: str, leave=True) -> None:
    if not condition:
        val_logger.info(error_msg)
        if leave:
            sys.exit(0)


def lang_norm_hint(lang: str, norm: str) -> str:
    intro = f"'{norm}' algorithm is not available for language '{lang}'\n"
    intro += f"Supported normalizer options for language {lang}:\n"
    if lang == "ru":
        hint = intro
        for item in ["snowball", "pymystem", "no"]:
            hint += f"\t-{item}\n"
    elif lang == "en":
        hint = intro
        for item in ["snowball", "wordnet", "porter",
                     "lancaster", "textblob", "no"]:
            hint += f"\t-{item}\n"
    else:
        hint = f"Language '{lang}' is not supported\n"
    return hint


if __name__ == '__main__':
    print(lang_norm_hint("ru", "porter"))
    print(lang_norm_hint("en", "pymystem"))
    print(lang_norm_hint("ch", "no"))
