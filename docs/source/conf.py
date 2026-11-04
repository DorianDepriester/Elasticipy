# Configuration file for the Sphinx documentation builder.
#
import os
import sys

from docutils import nodes
from docutils.parsers.rst import Directive
import plotly.io as pio

class PlotlyDirective(Directive):
    has_content = True
    option_spec = {
        'show-code': bool,
    }

    def run(self):
        show_code = self.options.get('show-code', True)
        code = "\n".join(self.content)

        # Exécuter le code pour obtenir la figure
        namespace = {}
        exec(code, namespace)
        fig = namespace.get("fig")
        if fig is None:
            raise RuntimeError("Le bloc .. plotly:: doit définir une variable `fig`.")

        # Générer le HTML de la figure
        html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

        # Créer les nœuds pour le code et la figure
        nodes_list = []
        if show_code:
            code_node = nodes.literal_block(code, code)
            code_node['language'] = 'python'
            nodes_list.append(code_node)

        raw_node = nodes.raw('', html, format='html')
        nodes_list.append(raw_node)

        return nodes_list

# Pages that moved during the documentation reorganization. Old URLs (root-level
# tutorials, pre-reorganization API pages, old module names) still receive
# traffic and would otherwise return 404. A tiny redirect page is generated at
# each old location, forwarding the reader to the new path below.
doc_redirects = {
    'Tutorial_AveragingMethods': 'Tutorials/Tutorial_AveragingMethods.html',
    'Tutorial_MultiplePhases': 'Tutorials/Tutorial_MultiplePhases.html',
    'Tutorial_ReadWriteFiles': 'Tutorials/Tutorial_ReadWriteFiles.html',
    'Tutorial_StiffnessTensor': 'Tutorials/Tutorial_StiffnessTensor.html',
    'Tutorial_StressStrain': 'Tutorials/Tutorial_StressStrain.html',
    'Tutorial_ThermalExpansion': 'Tutorials/Tutorial_ThermalExpansion.html',
    'Tutorial_wave-velocities': 'Tutorials/Tutorial_wave-velocities.html',
    'example_StressStrain': 'Tutorials/Tutorial_StressStrain.html',
    'GUI': 'Tutorials/GUI.html',
    'Elasticipy': 'API/API.html',
    'modules': 'API/API.html',
    '_modules/index': 'API/API.html',
    '_modules/Elasticipy/SecondOrderTensor': 'API/API.html',
    '_modules/Elasticipy/FourthOrderTensor': 'API/API.html',
    '_modules/Elasticipy/StressStrainTensors': 'API/API.html',
    '_modules/Elasticipy/SphericalFunction': 'API/API.html',
    'Elasticipy.plasticity': 'API/Elasticipy.plasticity.html',
    'elasticipy.plasticity': 'API/Elasticipy.plasticity.html',
    'hyperelasticity': 'API/Elasticipy.hyperelasticity.html',
    'tensors.elasticity': 'API/Elasticipy.tensors.elasticity.html',
    '_static/elasticity': 'API/Elasticipy.tensors.elasticity.html',
    'Elasticipy.tensors.elasticity': 'API/Elasticipy.tensors.elasticity.html',
    'Elasticipy.tensors.finite_strain': 'API/Elasticipy.tensors.finite_strain.html',
    'Elasticipy.tensors.fourth_order': 'API/Elasticipy.tensors.fourth_order.html',
    'Elasticipy.tensors.second_order': 'API/Elasticipy.tensors.second_order.html',
    'Elasticipy.tensors.stress_strain': 'API/Elasticipy.tensors.stress_strain.html',
    'Elasticipy.tensors.thermal_expansion': 'API/Elasticipy.tensors.thermal_expansion.html',
    'Elasticipy.SecondOrderTensor': 'API/Elasticipy.tensors.second_order.html',
    'Elasticipy.FourthOrderTensor': 'API/Elasticipy.tensors.fourth_order.html',
    'Elasticipy.StressStrainTensors': 'API/Elasticipy.tensors.stress_strain.html',
    'Elasticipy.SphericalFunction': 'API/Elasticipy.spherical_function.html',
    'Elasticipy.spherical_function': 'API/Elasticipy.spherical_function.html',
    'Elasticipy.interfaces.FEPX': 'API/Elasticipy.interfaces.FEPX.html',
    'Elasticipy.interfaces.PRISMS': 'API/Elasticipy.interfaces.PRISMS.html',
    'fibre_texture': 'auto_examples/fibre_texture.html',
    'Young_3D': 'auto_examples/Young_3D.html',
    'Poisson_sections': 'auto_examples/Poisson_sections.html',
}


def _collect_redirect_pages(app):
    pages = []
    for pagename, target in doc_redirects.items():
        prefix = '../' * pagename.count('/')
        context = {
            'redirect_target': prefix + target,
            'title': 'Redirecting to ' + target,
        }
        pages.append((pagename, context, 'redirect.html'))
    return pages


def setup(app):
    app.add_directive("plotly", PlotlyDirective)
    app.connect('html-collect-pages', _collect_redirect_pages)

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
   'sphinx.ext.intersphinx',
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

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
    'orix': ('https://orix.readthedocs.io/en/stable/', None),
}

nitpick_ignore = [
    ('py:class', "'Hill'}"),
    ('py:class', "'Monte Carlo'"),
    ('py:class', "'Reuss'"),
    ('py:class', "'contour'"),
    ('py:class', "'contourf'"),
    ('py:class', "'equal area'}"),
    ('py:class', "'imshow'"),
    ('py:class', "'matplotlib'}"),
    ('py:class', "'max'"),
    ('py:class', "'mean'"),
    ('py:class', "'min'"),
    ('py:class', "'minmax'"),
    ('py:class', "'right'}"),
    ('py:class', "'std'"),
    ('py:class', "'trapezoid'}"),
    ('py:class', "'trapezoid}"),
    ('py:class', "'y'}"),
    ('py:class', 'Corresponding isotropic stiffness tensor'),
    ('py:class', 'False'),
    ('py:class', 'FourthOrderTensor.'),
    ('py:class', 'Index in the vector'),
    ('py:class', 'Kelvin mapping'),
    ('py:class', 'Phase'),
    ('py:class', 'PlasticityCriterion'),
    ('py:class', 'Rotation'),
    ('py:class', 'SymmetricTensor'),
    ('py:class', 'Weighted average'),
    ('py:class', 'any'),
    ('py:class', 'array'),
    ('py:class', 'array-like'),
    ('py:class', 'array_like'),
    ('py:class', 'boolean'),
    ('py:class', 'cls'),
    ('py:class', 'damask._rotation.Rotation'),
    ('py:class', 'default'),
    ('py:class', "default 'x'"),
    ('py:class', 'default 1.'),
    ('py:class', 'default 50'),
    ('py:class', 'default False'),
    ('py:class', 'default None'),
    ('py:class', 'default Triclinic'),
    ('py:class', 'file'),
    ('py:class', 'floats'),
    ('py:class', 'iterable'),
    ('py:class', 'length 6'),
    ('py:class', 'matplotlib.axes._subplots.AxesSubplot'),
    ('py:class', 'matplotlib.pyplot.figure'),
    ('py:class', 'ndarray'),
    ('py:class', 'np.array'),
    ('py:class', 'np.ndarray'),
    ('py:class', 'optional'),
    ('py:class', 'orix Orientation'),
    ('py:class', 'orix.crystal_map.phase_list.Phase'),
    ('py:class', 'orix.phase.point_group.PointGroup'),
    ('py:class', 'orix.vector.Vector3D'),
    ('py:class', 'parameter to check'),
    ('py:class', 'plotly.graph_objs._figure.Figure'),
    ('py:class', 'pymatgen.analysis.elasticity.Strain'),
    ('py:class', 'pymatgen.analysis.elasticity.Stress'),
    ('py:class', 'pymatgen.analysis.elasticity.elastic.ElasticTensor'),
    ('py:class', 'pymatgen.core.tensors.Tensor'),
    ('py:class', 'scipy.spatial.Rotation'),
    ('py:class', 'self'),
    ('py:class', 'step size to use for finite differences'),
    ('py:class', "str {'Lambert'"),
    ('py:class', "str {'Voigt'"),
    ('py:class', "str {'exact'"),
    ('py:class', "str {'x'"),
    ('py:class', 'the full sphere.'),
    ('py:class', 'tuple/list'),
    ('py:class', 'whether to take consider the upper half-domain only'),
    ('py:class', "{'Voigt'"),
    ('py:class', "{'left'"),
    ('py:class', "{'plotly'"),
    ('py:obj', 'Piola_Kirchoff_1'),
    ('py:obj', 'Piola_Kirchoff_2'),
    ('py:obj', '__mul__'),
    ('py:obj', 'deviatoricPart'),
    ('py:obj', 'identity_tensor'),
    ('py:obj', 'skewPart'),
    ('py:obj', 'sphericalPart'),
]
autosectionlabel_prefix_document = True

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
}
pio.renderers.default = 'sphinx_gallery'


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/DorianDepriester/Elasticipy/blob/main/src/{filename}.py"
