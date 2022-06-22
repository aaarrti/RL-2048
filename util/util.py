import inspect
import functools
import pickle


def log_before(func):
    """
    Decorator to print function call details.
    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"Entered {func.__module__}.{func.__qualname__} with args ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper


def log_after(func):
    @functools.wraps(func)
    def wrapper(*func_args, **func_kwargs):
        retval = func(*func_args, **func_kwargs)
        print('Exited ' + func.__name__ + '() with value: ' + repr(retval))
        return retval

    return wrapper


def save_pickle(obj, fpath):
    with open(f'{fpath}.pickle', 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(f'{path}.pickle', "rb") as file:
        obj = pickle.load(file)
        return obj
