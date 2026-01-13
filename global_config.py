try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
    
import os

def load_config(filename):
    module_dir  = os.path.dirname(__file__)
    
    with open(os.path.join(module_dir, filename), mode="rb") as fp:
        config = tomllib.load(fp)
    
    return config