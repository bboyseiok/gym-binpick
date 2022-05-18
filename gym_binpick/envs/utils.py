from random import shuffle
from itertools import islice
from collections import deque, defaultdict, namedtuple
import time
import contextlib
import pstats
import cProfile
import random

INF = float('inf')
PI = np.pi

RRT_ITERATIONS = 20

RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)

def irange(start, stop=None, step=1):  # np.arange
    if stop is None:
        stop = start
        start = 0
    while start < stop:
        yield start
        start += step

def argmin(function, sequence):
    # TODO: use min
    values = list(sequence)
    scores = [function(x) for x in values]
    return values[scores.index(min(scores))]

def apply_alpha(color, alpha=1.):
   return tuple(color[:3]) + (alpha,)

def elapsed_time(start_time):
    return time.time() - start_time