import time
import functools


def log(text):
    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kw):
            s1 = time.time()
            r = fun(*args, **kw)
            s2 = time.time()
            print('%s %s %s ms' % (text, fun.__name__, 1000 * (s2 - s1)))
            return r

        return wrapper

    return decorator
