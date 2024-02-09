from fixedpointmath import FixedPoint


def safe_cast(_type: type, _value: str, debug: bool = False):
    if debug:
        print(f"trying to cast {_value} to {_type}")
        print(f"_type is {type(_type)} and _value is {type(_value)}")
    return_value = _value
    if debug:
        print(f"_type == int: {_type == 'int'}")
    if _type == "int":
        return_value = int(float(_value))
    if debug:
        print(f"_type == float: {_type == 'float'}")
    if _type == "float":
        return_value = float(_value)
    if debug:
        print(f"_type == bool: {_type == 'bool'}")
    if _type == "bool":
        return_value = _value.lower() in {"true", "1", "yes"}
    if debug:
        print(f"_type == FixedPoint: {_type == 'FixedPoint'}")
    if _type == "FixedPoint":
        return_value = FixedPoint(_value)
    if debug:
        print(f"  result: {_value} of {type(return_value).__name__}")
    return return_value
