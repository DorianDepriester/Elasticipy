from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='Elasticipy',
    use_scm_version=True,  # Active setuptools_scm
    setup_requires=["setuptools_scm"],  # Spécifie la dépendance à setuptools_scm
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    url='https://elasticipy.readthedocs.io/',
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
