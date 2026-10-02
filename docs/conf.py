# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import bayesian_listener  # noqa

# -- Project information
project = 'bayesian_listener'
copyright = f'2025–{datetime.date.today().year}, Roberto Barumerli, Fabian Brinkmann, Emanuele Zanoni, Anton Hoyer'
author = 'Roberto Barumerli, Fabian Brinkmann, Emanuele Zanoni, Anton Hoyer'
project_slug = 'bayesian_listener'

version = bayesian_listener.__version__
release = bayesian_listener.__version__

# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.bibtex',
    'sphinx.ext.doctest',
]

bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
highlight_language = 'python3'
add_function_parentheses = False
todo_include_todos = False

# Hide parent class/module prefixes in the right-hand "On this page" TOC.
# Body headings and anchors remain fully qualified for cross-references.
toc_object_entries_show_parents = 'hide'

# Treat undefined cross-references as errors (also catches stale :class:/:func: mentions).
nitpicky = True

# Suppress nitpicky errors for symbols whose libraries do not publish a Sphinx
# inventory file (no objects.inv).  Without this list, every reference to
# pyfar.Coordinates, pybads.BADS, etc. would raise a build error even though the
# names are correct.
nitpick_ignore = [
    ('py:class', 'pybads.BADS'),
    ('py:class', 'spharpy'),
    ('py:class', 'spaudiopy'),
    ('py:class', 'Barumerli2023'),
    ('py:class', 'bayesian_listener.auditory_representation._AuditoryRepresentation'),
    # Numpy-style shape annotations parsed as classes by napoleon.
    ('py:class', 'ndarray'),
    ('py:class', 'shape'),
    ('py:class', 'array-like'),
    ('py:class', 'scalar'),
    ('py:class', 'optional'),
    # Free shape symbols.
    ('py:class', 'M'),
    ('py:class', '3'),
    ('py:class', 'n_src'),
    ('py:class', 'n_grid'),
    # Path leaks (pathlib).
    ('py:class', 'Path'),
    # Generic Python type words.
    ('py:class', 'callable'),
    ('py:class', 'function'),
    # Module-level data referenced by docstrings.
    ('py:data', 'METRIC_FUNCTIONS'),
    # Submodule references (autodoc does not register module objects unless
    # an explicit automodule directive is used; we never use one).
    ('py:mod', 'bayesian_listener'),
    ('py:mod', 'bayesian_listener.metrics'),
    ('py:mod', 'bayesian_listener.fitting'),
    ('py:mod', 'bayesian_listener.utils'),
    ('py:mod', 'bayesian_listener.resample'),
    ('py:mod', 'bayesian_listener.plotting'),
    ('py:mod', 'bayesian_listener.auditory_representation'),
]

# Napoleon's "default=..." and "{'a', 'b'}" enum syntax in type fields produce
# stray cross-reference attempts; suppress them all with a regex match on the
# parsed token text rather than enumerating each one.
import re as _re
nitpick_ignore_regex = [
    ('py:class', _re.compile(r'^default=.*$')),
    ('py:class', _re.compile(r'^\{.*$')),    # opening of an enum literal
    ('py:class', _re.compile(r'^.*\}$')),    # closing of an enum literal
    ('py:class', _re.compile(r"^['\"].*['\"]$")),  # bare quoted literals
    ('py:class', _re.compile(r'^\d.*$')),    # numeric defaults like "700.0"
]

# -- autodoc
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

# -- Napoleon (NumPy-style docstrings)
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy':  ('https://numpy.org/doc/stable/', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pyfar':  ('https://pyfar.readthedocs.io/en/stable/', None),
    'sofar':  ('https://sofar.readthedocs.io/en/stable/', None),
}

# -- HTML output
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_favicon = '_static/favicon.png'
html_logo = '_static/favicon.png'
html_title = 'Bayesian Listener'

# Furo theme options.
# Palette: dark teal / slate family.
# Light mode primary #0e6e7e: contrast ratio ~5.9:1 against white (WCAG AA pass).
# Dark  mode primary #4ecdc4: contrast ratio ~9.8:1 against Furo dark bg (WCAG AA pass).
# -- linkcheck
# DOI redirects through to publisher landing pages are normal; treat them
# as OK rather than as warnings.
linkcheck_allowed_redirects = {
    r'https://doi\.org/.*': r'https://.*',
}
# These URLs are either:
#   - blocked from automated HEAD/GET requests by the publisher (403),
#   - fronted by an institutional portal that times out under linkcheck's
#     30 s budget (EU joinup),
#   - upstream resources that move occasionally (Gräf homepage, OSF DOI
#     for the SONICOM dataset, spaudiopy GitHub mirror).
# In every case the URL has been verified manually and the underlying
# resource is reachable from a normal browser.
linkcheck_ignore = [
    r'^https://doi\.org/10\.1121/1\.427147$',
    r'^https://joinup\.ec\.europa\.eu/.*',
    r'^https://homepage\.univie\.ac\.at/.*',
    r'^https://doi\.org/10\.17605/OSF\.IO/M36C2$',
    r'^https://github\.com/chris-hold/spaudiopy.*',
]
linkcheck_timeout = 30
linkcheck_retries = 2

html_theme_options = {
    'light_css_variables': {
        'color-brand-primary':        '#0e6e7e',
        'color-brand-content':        '#0e6e7e',
        'color-highlight-on-target':  '#ddf2f5',
    },
    'dark_css_variables': {
        'color-brand-primary':        '#4ecdc4',
        'color-brand-content':        '#4ecdc4',
        'color-highlight-on-target':  '#1a3c42',
    },
    'navigation_with_keys': True,
    'top_of_page_button': 'edit',
    'sidebar_hide_name': False,
    # Required by Furo to render the edit button on each page.
    'source_repository': 'https://github.com/robaru/bayesian_listener/',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/robaru/bayesian_listener',
            'html': (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                '3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13'
                '-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 '
                '2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
                '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82'
                '.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 '
                '2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 '
                '3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 '
                '1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z">'
                '</path></svg>'
            ),
            'class': '',
        },
    ],
}

