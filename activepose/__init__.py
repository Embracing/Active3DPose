import os
import sys


if os.getenv('PWD'):
    MODULE_DIR = os.path.join(os.getenv('PWD'), 'activepose')
else:
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODULE_DIR)

DEBUG = getattr(sys, 'gettrace', lambda: None)() is not None
