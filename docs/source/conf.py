# Configuration file for the Sphinx documentation builder.
#
import os
import sys
src_path = os.path.abspath('../../src/')
sys.path.insert(0, src_path)
print(f"Chemin ajouté au PYTHONPATH : {sys.path}")

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Elasticipy'
copyright = '2024, Dorian Depriester'
author = 'Dorian Depriester'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx_rtd_theme',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.napoleon',
   'sphinx.ext.githubpages',
   'sphinx.ext.autosectionlabel',
   'sphinx.ext.mathjax',
   'sphinx.ext.linkcode']
templates_path = ['_templates']
exclude_patterns = []

language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['../_static']
numpydoc_class_members_toctree = False
autoclass_content = 'both'

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/DorianDepriester/Elasticipy/blob/main/src/{filename}.py"
