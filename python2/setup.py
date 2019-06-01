from setuptools import setup, find_packages

__version__ = '1.0.0'

setup(
    name='IsolationForest',
    version=__version__,
    description='An implementation of the Isolation Forest anomaly detection algorithm.',
    url='https://github.com/msimms/LibIsolationForest',
    packages=['isolationforest',],
    author='Mike Simms',
    author_email='mike@mikesimms.net',
    license='MIT',
    install_requires=[],
    python_requires='==2.7.*'
)
