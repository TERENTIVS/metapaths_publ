from setuptools import setup, find_packages

setup(
    name="metapaths",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy~=1.23.5',
        'pandas~=1.5.3',
        'py2neo~=2021.2.3',
        'tqdm~=4.64.1',
    ],
)
