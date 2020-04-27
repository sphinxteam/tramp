import sphinx_rtd_theme
import sphinx_fontawesome

# -- Project information -----------------------------------------------------
project = 'TRAMP'
copyright = '2020, Antoine Baker, Benjamin Aubin, Florent Krzakala, Lenka Zdeborova'
author = 'Antoine Baker, Benjamin Aubin, Florent Krzakala, Lenka Zdeborova'
release = '0.1'
language = 'en'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    # 'recommonmark',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.napoleon',
    'sphinxjp.themes.basicstrap',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx_fontawesome'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ["_static"]
source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
locale_dirs = ['locale/']
gettext_compact = False

master_doc = 'index'
suppress_warnings = ['image.nonlocal_uri']
# 'monokai','solarized-light', 'tango',  'solarized-dark', 'pastie'
pygments_style = 'solarized-dark'


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
    'prev_next_buttons_location': 'none',
    'style_external_links': True,
    'style_nav_header_background': '#118ab2',
    'navigation_depth': 4,
    'titles_only': False

}


html_logo = "_templates/logo_light.png"
html_show_sourcelink = True

## sphinx-gallery configuration ##
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../../examples/sparseRegression'],
    'gallery_dirs': ['gallery/sparseRegression'],
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('tramp')
}

html_css_files = [
    'css/custom.css',
]
html_js_files = [
    'js/custom.js',
]
# html_style = 'css/customtheme.css'
