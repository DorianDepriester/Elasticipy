from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='Elasticipy',
    version='1.0.3',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    url='https://github.com/DorianDepriester/Elasticipy',
    license='MIT Licence',
    author='Dorian Depriester',
    author_email='dorian.depriester@ensam.eu',
    description='Collection of tools to work on strain, stress and stiffness tensors, with plotting features',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
            'scipy',
            'numpy',
        ],
)
