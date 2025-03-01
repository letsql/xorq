def return_constant(value):
    def wrapped(*args, **kwargs):
        return value

    return wrapped
