ASTERIA-Detection Docs
==========================

`ASTERIA Detection`_ aims to transform the landscape of security in computing systems 
by providing a comprehensive framework for vulnerability detection. 
This cutting-edge architecture relies on deep learning methodologies, 
offering extensible and customizable solutions at the forefront of technology.

.. _ASTERIA-Detection: https://github.com/Asteriska001/ASTERIA-Detection 


Key Features
-----------------
**Extensible Resourceful Intelligence**: Incorporates AI and machine learning algorithms that can be easily extended and adapted to various security testing scenarios.
**Rich and Diverse Data Preprocessing Mechanisms**: Including code representation methods, graph neural networks, and sequence representation methods, allowing the free combination of preprocessing steps and adjustment of the processing flow.
**Adaptable Framework**: Tailored to meet individual requirements with built-in support for various vulnerability detection models.
**Abundant Datasets**: Offers a wide variety of datasets to train and test the models, providing a versatile environment for experimentation.
**Ease of Use**: Designed with the user in mind, ASTERIA offers an intuitive interface that makes implementing and modifying state-of-the-art (SOTA) vulnerability detection models a breeze.


User Guide
----------------

:doc:`installing`
    How to install this vulnerability detection framework.

:doc:`configuring`
    Task configuration and customization options.

:ref:`supported-models`
    Supported detection models, like Devign and VulDeePecker.

:ref:`supported-datasets`
    Supported detection vulnerability datasets, like SARD/NVD and REVEAL.

:ref:`supported-preprocess/representations`
    Supported data preprocess methods and representations, like Normalization and Graph extractors.

Development
-----------

:doc:`contributing`
    How to contribute changes to the theme.

:doc:`Development guidelines <development>`
    Guidelines the theme developers use for developing and testing changes.

`Read the Docs contributor guide`_
    Our contribution guidelines extend to all projects maintained by Read the
    Docs core team.

:doc:`changelog`
    The theme development changelog.

:doc:`Demo documentation <demo/structure>`
    The theme's styleguide test environment, where new changes are tested.


.. _Read the Docs contributor guide: https://docs.readthedocs.io/en/stable/contribute.html


.. Hidden TOCs

.. toctree::
   :caption: Theme Documentation
   :maxdepth: 2
   :hidden:

   installing
   configuring
   development
   contributing

.. toctree::
   :maxdepth: 1
   :hidden:

   changelog

.. toctree::
    :maxdepth: 2
    :numbered:
    :caption: Demo Documentation
    :hidden:

    demo/structure
    demo/demo
    demo/lists_tables
    demo/api

.. toctree::
    :maxdepth: 3
    :numbered:
    :caption: This is an incredibly long caption for a long menu
    :hidden:

    demo/long
    
.. toctree::
    :maxdepth: 3
    :caption: Breadcrumbs

    demo/level1/index.rst
