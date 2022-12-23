# check if linting is working

def test(x: int) -> int:
    return x

def bad_type(x: float) -> int:
    return test(x)

