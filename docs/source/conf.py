# Configuration file for the Sphinx documentation builder.
#
import os
import sys

from docutils import nodes
from docutils.parsers.rst import Directive
import plotly.io as pio

class PlotlyDirective(Directive):
    has_content = True

    def run(self):
        code = "\n".join(self.content)
        namespace = {}
        exec(code, namespace)

        fig = namespace.get("fig")
        if fig is None:
            raise RuntimeError("Le bloc .. plotly:: doit définir une variable `fig`.")

        html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
        return [nodes.raw('', html, format='html')]

def setup(app):
    app.add_directive("plotly", PlotlyDirective)

src_path = os.path.abspath('../../src/')
sys.path.insert(0, src_path)
print(f"Chemin ajouté au PYTHONPATH : {sys.path}")

import matplotlib
matplotlib.use("Agg")

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'elasticipy'
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
   'matplotlib.sphinxext.plot_directive',
   'sphinx_gallery.gen_gallery',
]

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

sphinx_gallery_conf = {
    'examples_dirs': ['examples'],  # Dossier source des exemples
    'gallery_dirs': ['auto_examples'],  # Dossier de sortie
    'filename_pattern': r'\.py',  # Inclut tous les fichiers .py
    'ignore_pattern': r'_template\.py',  # Exclut les fichiers template
    'plot_gallery': True,  # Affiche les figures
    'reference_url': {
        'elasticipy': None,  # Désactive les liens vers la doc API si non configuré
    },
    'capture_repr': ('_repr_html_', '__repr__'),  # Capture les sorties textuelles
    'show_memory': False,  # Désactive l'affichage de la mémoire (optionnel)
    'image_scrapers': ('plotly.io._sg_scraper.plotly_sg_scraper',),
}
import plotly.io as pio
pio.renderers.default = 'sphinx_gallery_png'


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/DorianDepriester/Elasticipy/blob/main/src/{filename}.py"
