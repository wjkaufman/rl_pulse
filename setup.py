# from os.path import join, dirname, realpath
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
    version=__version__,
    install_requires=[
        'mpi4py',
        'numpy',
        'scipy',
        'unittest',
        'psutil',
        'sphinx',
        'tensorflow==2.2.0',
        'tf-agents==0.5.0',
        'tensorflow-probability==0.10.0',
        'cloudpickle==1.4.1'
    ],
    description="Designing pulse sequences using reinforcement learning.",
    author="Will Kaufman",
)
