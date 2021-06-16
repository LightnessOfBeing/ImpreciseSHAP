from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requirements = f.read().splitlines()

setup(
    name="impreciseshap",
    version="0.0.1",
    description="Implementation of Imprecise SHAP",
    author="Kirill Vishniakov",
    author_email="ki.vishniakov@gmail.com",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requirements
)
