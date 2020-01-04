import os
from setuptools import setup, find_packages


# get __version__
ver_file = os.path.join('thumoon', 'version.py')
with open(ver_file) as f:
    exec(f.read())


with open(os.path.join('requirements.txt'),encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="thumoon",
    version=__version__,
    description='A python toolbox for hypergraph learning',
    author='Haojie Lin',
    author_email='linhj18@mails.tsinghua.edu.cn',
    url='https://github.com/iMoonLab/THU-Moon',
    install_requires=requirements,
    packages=find_packages(exclude=['test']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
