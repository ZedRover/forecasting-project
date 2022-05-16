import multiprocessing
from multiprocessing import Pool

import itertools
import functools
from multiprocessing.pool import ThreadPool


class DodrioFakeException():

    def __init__(self, exception, misc):
        self.exception = exception
        self.misc = misc

    def __str__(self):
        return self.exception.__str__()


class TupleCaller():

    def __init__(self, func, with_try=False):
        self.func = func
        self.with_try = with_try

    def __call__(self, tup):
        if not self.with_try:
            return self.func(*tup[0], **tup[1])
        try:
            return self.func(*tup[0], **tup[1])
        except Exception as e:
            return DodrioFakeException(e, tup)
        except BaseException as e:
            return DodrioFakeException(e, tup)
        except:
            return DodrioFakeException(Exception('unknown error'), tup)


def enum_params(candidate):
    def _generator(entry):
        k, l = entry
        for v in l:
            if type(k) is str:
                yield {k: v}
            elif type(k) is tuple:
                yield dict(zip(k, v))
            else:
                raise Exception('unsupported param_candidate key type')

    _iter_params = itertools.product(*[_generator(i) for i in candidate.items()])
    return list(map(lambda x: functools.reduce(lambda a, b: {**a, **b}, x), _iter_params))


def multi_run(func, params, static_params=None, n_jobs=4, ordered=False, with_try=False, timeout=None):

    static_params = static_params or {}
    params = {**params, **{i: [j] for i, j in static_params.items()}}

    caller_func = TupleCaller(func, with_try=with_try)

    if timeout is not None and timeout > 0:
        caller_func = functools.partial(abortable_check, caller_func, timeout=timeout)

    if n_jobs == 1:
        param_list = [([], i) for i in enum_params(params)]
        yield from map(caller_func, param_list)

    with Pool(n_jobs) as pool:
        param_list = [([], i) for i in enum_params(params)]
        yield from (pool.imap_unordered if not ordered else pool.imap)(caller_func, param_list)


def abortable_check(func, *kargs, timeout=5):
    p = ThreadPool(1)
    res = p.apply_async(func, args=kargs)
    try:
        out = res.get(timeout) # Wait timeout seconds for func to complete.
    except multiprocessing.TimeoutError as e:
        return DodrioFakeException(Exception('timeout'), kargs)
    return out
