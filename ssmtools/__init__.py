import os,sys

try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    base_path = sys._MEIPASS
except Exception:
    base_path = os.path.dirname(__file__)

AUX_FILES_DIR = os.path.abspath(os.path.join(base_path, 'extras'))