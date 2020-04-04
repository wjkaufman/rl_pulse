from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The RL Pulse repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open("version.py") as version_file:
    exec(version_file.read())

setup(
    name='rl_pulse',
    py_modules=['rl_pulse'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle==1.2.1',
        # 'gym[atari,box2d,classic_control]~=0.15.3',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'tensorflow>=1.8.0',
        # 'torch==1.3.1',
        'tqdm'
    ],
    description="Designing pulse sequences using RL methods.",
    author="Will Kaufman",
)
