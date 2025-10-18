import pickle
import os
from typing import TypeVar, Callable
from pathlib import Path

T = TypeVar('T')

def load_cache_or_make_new(
    cache_path: Path,
    creator_fn: Callable[[], T],
    expected_type: type[T],
    force_regen: bool
) -> T:
    """
    load from cache, otherwise regenerate object
    """
    if cache_path.exists() and not force_regen:
        with open(cache_path, 'rb') as f:
            obj = pickle.load(f)
        assert isinstance(obj, expected_type), f"Loaded object is of type {type(obj)}, expected {expected_type}"
        return obj

    obj = creator_fn()
    
    with open(cache_path, 'wb') as f:
        pickle.dump(obj, f)
        os.system(f'chmod 666 {cache_path}')
    
    return obj

