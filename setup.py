from setuptools import setup, find_packages
from os.path import abspath, join as pjoin, relpath, split

DISTNAME = 'chatbot'
DESCRIPTION = 'Chatbot for Exercise'
AUTHOR = 'lee3jjang'
AUTHOR_EMAIL = 'lee3jjang@gmail.com'
URL = 'https://github.com/lee3jjang/chatbot'
SETUP_DIR = split(abspath(__file__))[0]
with open(pjoin(SETUP_DIR, 'README.md'), encoding='utf8') as readme:
    README = readme.read()
LONG_DESCRIPTION = README

setup(
    name=DISTNAME,
    version='0.0.1',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    packages=find_packages(),
    python_requires=">=3.8",
    zip_safe=False,
)