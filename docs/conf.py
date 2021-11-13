import sphinx_rtd_theme
import sphinx_fontawesome
from sphinx_gallery.sorting import ExplicitOrder

# -- Project information -----------------------------------------------------
project = 'Tree-AMP'
copyright = '2021, Tree-AMP developers'
author = 'Tree-AMP developers'
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
    "texext.math_dollar",
    'sphinx_rtd_theme',
    'sphinx.ext.extlinks',
    'sphinx_fontawesome',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
]

# generate autosummary pages
autosummary_generate = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# Add any paths that contain templates here, relative to this directory.
html_static_path = ["_static"]
source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '*.md']
locale_dirs = ['locale/']
gettext_compact = False

master_doc = 'index'
# 'friendly', 'tango',  'paraiso-dark', 'pastie', ...
pygments_style = 'friendly'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

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

html_logo = "_static/logo_light.png"
html_show_sourcelink = True

# sphinx-gallery configuration ##
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../examples/'],
    "subsection_order": ExplicitOrder(
        [
            "../examples/sparse",
            "../examples/glm",
            "../examples/vae_prior",
        ]
    ),
    'gallery_dirs': ['gallery'],
    'backreferences_dir': 'modules/backreferences',
    'doc_module': ('tramp'),
    'image_scrapers': ('matplotlib'),
}

# show section and code author
show_authors = True

# Custom theme
html_copy_source = False

# Copy button
copybutton_prompt_text = ">>> "

# show todolist
todo_include_todos = True
