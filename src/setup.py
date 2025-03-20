from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    readme = fh.read()

setup(
    name='metapaths',
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        'numpy~=1.23.5',
        'pandas~=1.5.3',
        'py2neo~=2021.2.3',
        'tqdm~=4.64.1',
    ],
    author='Terence Egbelo',
    description='Metapath-based, Neo4j-powered knowledge graph completion with novel topological bias control for better accuracy on low-degree nodes.',
    url="https://github.com/TERENTIVS/metapaths_publ",
    readme=readme,
    author_email="terence.egbelo@gmail.com",
    classifiers=[
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ]
)