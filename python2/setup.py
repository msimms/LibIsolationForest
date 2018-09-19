from setuptools import setup, find_packages

requirements = ['plotly']

setup(
    name='libisolationforest',
    version='0.1.0',
    description='',
    url='https://github.com/msimms/LibIsolationForest',
    author='Mike Simms',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=requirements,
    python_requires='==2.7.*'
)
