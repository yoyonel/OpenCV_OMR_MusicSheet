from os.path import dirname
from os.path import join


def get_module_path_datasets():
    module_path = dirname(__file__)
    return join(module_path, 'data')


__all__ = ['get_module_path_datasets']
