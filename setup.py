from setuptools import setup, find_packages
from codecs import open
from os import path


__version__ = "0.0.1"

current_dir = path.abspath(path.dirname(__file__))

with open(path.join(current_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(current_dir, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs]

setup(
    name="mentflow",
    version=__version__,
    description="Maximum entropy tomography with normalizing flows",
    long_description=long_description,
    url="https://github.com/austin-hoover/ment-flow",
    license="MIT",
    keywords="",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    author=["Austin Hoover"],
    install_requires=install_requires,
    author_email="",
)
