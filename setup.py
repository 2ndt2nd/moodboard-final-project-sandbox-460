import sys
sys.setrecursionlimit(20000)  # Increase recursion limit to avoid RecursionError during build

from setuptools import setup

APP = ['main_mf.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'packages': ['PyQt5', 'torch', 'clip', 'tqdm'],
    'excludes': ['tkinter'],
    'plist': {
        'CFBundleName': 'MoodForager',
        'CFBundleDisplayName': 'MoodForager',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': 'Copyright © 2023 Jacques Davidson Widodo',
    },
    'optimize': 2,
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
