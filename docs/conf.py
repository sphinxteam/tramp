import sphinx_rtd_theme
import sphinx_fontawesome
from sphinx_gallery.sorting import ExplicitOrder

# -- Project information -----------------------------------------------------
project = 'TRAMP'
copyright = '2020, Antoine Baker, Benjamin Aubin, Florent Krzakala, Lenka Zdeborova'
author = 'Antoine Baker, Benjamin Aubin, Florent Krzakala, Lenka Zdeborova'
release = '0.1'
version = '0.1'
language = 'en'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "nb2plots",
    "texext",
    'sphinx_rtd_theme',
    'recommonmark',
    'sphinx.ext.extlinks',
    'sphinx_fontawesome',
    'sphinx.ext.githubpages'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ["_static"]
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
locale_dirs = ['locale/']
gettext_compact = False

master_doc = 'index'
# 'monokai','solarized-light', 'tango',  'solarized-dark', 'pastie'
pygments_style = 'tango'  # 'monokai'  # 'solarized-dark'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': True,
    'sticky_navigation': False,
    'includehidden': False,
    'logo_only': True,
    'navigation_depth': 5,
    'display_version': True,
    'prev_next_buttons_location': 'top',
    'style_external_links': True,
    'navigation_depth': 4,
    'titles_only': False
}

html_logo = "_templates/logo_light.png"
html_show_sourcelink = True

# sphinx-gallery configuration ##
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../examples/'],
    "subsection_order": ExplicitOrder(
        [
            "../examples/sparse",
            "../examples/GLM",
        ]
    ),
    'gallery_dirs': ['gallery'],
    'backreferences_dir': 'modules/backreferences',
    'doc_module': ('tramp'),
    'image_scrapers': ('matplotlib'),
}


html_css_files = [
    'css/customtheme.css',
]
html_js_files = [
]


latex_elements = {
    'preamble': r'\usepackage{tikz}',
}
