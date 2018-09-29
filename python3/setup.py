from setuptools import setup, find_packages

setup(
    name='libisolationforest',
    version='0.9.0',
    description='An implementation of the Isolation Forest anomaly detection algorithm.',
    url='https://github.com/msimms/LibIsolationForest',
    author='Mike Simms',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[],
    python_requires='>=3.5'
)
