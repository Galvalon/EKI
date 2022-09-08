import cProfile
import pstats
from typing import Callable


def profileit(func):
    def wrapper(*args, **kwargs):
        # Name the data file sensibly
        datafn = func.__name__ + ".profile"
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval
    return wrapper


def read_profile(file: str):
    pr = pstats.Stats(file)
    pr.sort_stats('time').print_stats(10)
