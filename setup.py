from setuptools import setup

APP = ["qt_app.py"]
DATA_FILES = []  # Empty! No files bundled.
OPTIONS = {
    "argv_emulation": False,  # Disable for PyQt apps
    "packages": ["PyQt5", "torch", "clip", "tqdm"],
    "includes": ["PyQt5.QtWebEngineWidgets"],
    "plist": {
        "CFBundleName": "MoodForager",
        "CFBundleDisplayName": "MoodForager",
        "CFBundleIdentifier": "com.yourname.moodforager",
        "NSHighResolutionCapable": "True"  # Retina support
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,  # Nothing extra included
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)