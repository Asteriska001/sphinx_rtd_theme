Installation
============

How to install and use the framework
--------------------------------

Install the ``sphinx_rtd_theme`` package (or add it to your ``requirements.txt`` file):

.. code:: console

    $ git clone https://github.com/Asteriska001/astBugDetection
    $ cd astBugDetection
    $ pip install -e .

open the Python interpreter  and run the following code to check:

.. code:: python

    from framework import show_models
    show_models()

If the framework is installed successfully, you will see the following output

.. code:: console

        No.  Model Names
    -----  -------------
        1  GNNReGVD
        2  Devign
        3  LineVD
        4  BLSTM
        ......

.. seealso::
    :ref:`supported-models`
        Supported detection models, like Devign and VulDeePecker.

    :ref:`supported-datasets`
        Supported detection vulnerability datasets, like SARD/NVD and REVEAL.

    :ref:`supported-preprocess/representations`
        Supported data preprocess methods and representations, like Normalization and Graph extractors.

   ..
      comment about this note: it's possibly not necessary to add the theme as an extension.
      Rather, this is an issue caused by setting html_theme_path.
      See: https://github.com/readthedocs/readthedocs.org/pull/9654

this docs are constructing.
.. _howto_upgrade:

How to upgrade
--------------

Adding ``sphinx-rtd-theme`` to your project's dependencies will make pip install the latest compatible version of the theme.

If you want to test a **pre-release**, you need to be explicit about the version you specify.
Otherwise, pip will ignore pre-releases. Add for instance ``sphinx-rtd-theme==1.1.0b3`` to test a pre-release.

.. tip::
    We recommend that you pin the version of Sphinx that your project is built with.
    We won't release sphinx-rtd-theme without marking its compatibility with Sphinx. So if you do not pin ``sphinx-rtd-theme`` itself, you will always get the *latest compatible* release.
    
    More information is available in Read the Docs' documentation on :doc:`rtd:guides/reproducible-builds`.


Via Git or Download
-------------------

.. warning::

   Installing directly from the repository source is deprecated and is not
   recommended. Static assets won't be included in the repository starting in
   release :ref:`roadmap-release-3.0.0`.

Symlink or subtree the ``sphinx_rtd_theme/sphinx_rtd_theme`` repository into your documentation at
``docs/_themes/sphinx_rtd_theme`` then add the following two settings to your Sphinx
``conf.py`` file:

.. code:: python

    html_theme = "sphinx_rtd_theme"
    html_theme_path = ["_themes", ]
