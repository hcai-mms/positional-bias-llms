# ensure all modules are loaded, so Models are registered
from os.path import dirname, basename, isfile, join
import glob
for f in  glob.glob(join(dirname(__file__), "*.py")):
    if isfile(f) \
        and not f.endswith('__init__.py') \
        and not f.endswith('library.py') \
        and not f.endswith('basemodel.py'):
        __import__("models."+basename(f)[:-3])

from models.library import model_library
