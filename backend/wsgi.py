import os
import sys
from dotenv import load_dotenv

# Set the path to your project directory on PythonAnywhere
# Example: '/home/gtortoretto/QkNN_Website/backend'
path = '/home/gtortoretto/QkNN_Website/backend'
if path not in sys.path:
    sys.path.insert(0, path)

# Load environment variables from .env file
project_folder = os.path.expanduser('~/QkNN_Website/backend')
load_dotenv(os.path.join(project_folder, '.env'))

from app import app as application


from app import app as application