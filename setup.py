import glob
import shutil
from distutils.command.install import install
from distutils.core import setup

from setuptools import find_packages


class PostDevelopCommand(install):
    def run(self):
        install.run(self)
        shutil.rmtree(glob.glob("*.egg-info")[0])


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        #shutil.rmtree("dist")
        shutil.rmtree(glob.glob("*.egg-info")[0])
        shutil.rmtree(glob.glob("build")[0])


with open("requirements.txt") as f:
    install_requirements = f.read().splitlines()

setup(
    name="impreciseshap",
    version="0.0.1",
    description="Implementation of Imprecise SHAP",
    author="Kirill Vishniakov",
    author_email="ki.vishniakov@gmail.com",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requirements,
    cmdclass={"install": PostInstallCommand, "develop": PostDevelopCommand}
)
