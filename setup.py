from setuptools import setup, find_packages


setup(
    name="msbo",
    version="1.0",
    packages=find_packages(),
    install_requires=['gpytorch<1.9', 'botorch', 'pyyaml', 'matplotlib', 'openpyxl', 'pandas'],
)
