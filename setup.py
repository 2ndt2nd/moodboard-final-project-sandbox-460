from setuptools import setup
import py2app

APP = ['qt_app.py']
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
        'NSHumanReadableCopyright': 'Copyright © 2023 Your Name',
    },
    'iconfile': 'app_icon.icns',  # Add if you have an icon
    'optimize': 2,
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)