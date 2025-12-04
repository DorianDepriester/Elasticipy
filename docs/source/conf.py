# Configuration file for the Sphinx documentation builder.
#
import os
import sys
src_path = os.path.abspath('../../src/')
sys.path.insert(0, src_path)
print(f"Chemin ajouté au PYTHONPATH : {sys.path}")

import matplotlib
matplotlib.use("Agg")

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Elasticipy'
copyright = '%Y, Dorian Depriester'
author = 'Dorian Depriester'
release = '4.2.0'

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
   'sphinx.ext.linkcode',
   'sphinx_copybutton',
   'matplotlib.sphinxext.plot_directive']

templates_path = ['_templates']
exclude_patterns = []
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
plot_include_source = True
plot_html_show_source_link = False


language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['../_static']
html_logo = "logo/logo_text_whitebg.png"
html_favicon = 'logo/favicon.png'
numpydoc_class_members_toctree = False
autoclass_content = 'both'

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/DorianDepriester/Elasticipy/blob/main/src/{filename}.py"
