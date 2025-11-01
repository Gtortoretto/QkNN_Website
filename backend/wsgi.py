import os
import sys
from dotenv import load_dotenv

path = '/home/gtortoretto/QkNN_Website/backend'
if path not in sys.path:
    sys.path.insert(0, path)

project_folder = os.path.expanduser('~/QkNN_Website/backend')
load_dotenv(os.path.join(project_folder, '.env'))

from app import app as application
