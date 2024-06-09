import os
import pickle
from functools import wraps
from Scripts.constants import data_dir
import hashlib

def cache_plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        func_defaults = func.__defaults__ or ()
        func_code = func.__code__
        func_args = func_code.co_varnames[:func_code.co_argcount]
        name_args = dict(zip(func_args, (*args, *func_defaults)))
        name_args.update(kwargs)
        s = func_name + "_" + str(name_args)
        s = s.encode('utf-8')
        hash_obj = hashlib.sha256()
        hash_obj.update(s)
        hash_val = hash_obj.hexdigest()
        save_p = os.path.join(data_dir, ".cache", f'{hash_val}.pkl')
        if os.path.isfile(save_p):
            with open(save_p, 'rb') as f:
                res = pickle.load(f)
        else:
            res = func(*args, **kwargs)
            with open(save_p, 'wb') as f:
                pickle.dump(res, f)
        return res
    return wrapper

