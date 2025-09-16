# ensure all modules are loaded, so Scorers are registered
from os.path import dirname, basename, isfile, join
import glob
for f in  glob.glob(join(dirname(__file__), "*.py")):
    if isfile(f) \
        and not f.endswith('__init__.py') \
        and not f.endswith('library.py'):
        __import__("scorer."+basename(f)[:-3])

from scorer.library import scorer_library