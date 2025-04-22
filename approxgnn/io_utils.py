from argparse import ArgumentTypeError


def arg_type_check(arg: str, values: list[str]):
    if arg not in values:
        raise ArgumentTypeError(
            f"{arg} is not a valid value. It has to be one of {', '.join(values)}."
        )
    return arg


def arg_range_check(arg: str, lb: float, ub: float):
    try:
        value = float(arg)
    except ValueError as e:
        raise ArgumentTypeError(
            f"Provided value has to be a float in range [{lb}, {ub}]"
        ) from e

    if value < lb or value > ub:
        raise ArgumentTypeError(f"Provided value has to be in range [{lb}, {ub}]")

    return value
