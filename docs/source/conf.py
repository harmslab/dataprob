# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'dataprob'
copyright = '2025, Michael J. Harms'
author = 'Michael J. Harms'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "github_url": "https://github.com/harmslab/dataprob",
}
