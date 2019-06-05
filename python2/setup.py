from setuptools import setup, find_packages
import os

__version__ = '1.0.0'

with open(os.path.join("..", "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name='IsolationForest',
    version=__version__,
    description='An implementation of the Isolation Forest anomaly detection algorithm.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/msimms/LibIsolationForest',
    packages=['isolationforest',],
    author='Mike Simms',
    author_email='mike@mikesimms.net',
    license='MIT',
    install_requires=[],
    python_requires='==2.7.*',
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
