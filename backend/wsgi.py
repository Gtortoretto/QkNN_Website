import os
import sys

path = '/home/<Your-Username>/<Your-Project-Directory>'
if path not in sys.path:
    sys.path.insert(0, path)

os.environ['SERVER_ENV'] = 'PYTHONANYWHERE'

from app import app as application