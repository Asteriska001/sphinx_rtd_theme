Development
===========

The framework developers follow the guidelines below for development and release planning. 

.. _supported-models:

Supported Models
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Model Name
     - Description
   * - Devign
     - https://github.com/epicosy/devign
   * - ReGVD
     - ReGVD(Region-based Graph Vulnerability Detection) https://github.com/daiquocnguyen/GNN-ReGVD
   * - LineVul
     - https://github.com/awsm-research/LineVul/
   * - VulDeePecker
     - VulDeePecker是一种深度学习系统，能够自动检测给定源代码中的漏洞，通过长短时记忆网络（LSTM）进行特征学习。
   * - TextCNN
     - TextCNN是一种采用卷积神经网络（CNN）对文本数据进行处理的模型，可应用于代码的漏洞检测，通过文本特征实现。
   * - VulCNN
     - VulCNN是一种专门针对漏洞检测的卷积神经网络模型，通过分析代码的结构和语义信息来识别潜在的安全漏洞。

.. _supported-datasets:

Supported datasets
----------------------

The framework officially supports the following datasets:

.. list-table:: Supported datasets
    :header-rows: 1
    :widths: 10, 10

    * - Dataset
      - Description
    * - REVEAL
      - 
    * - SARD/NVD
      - 

this docs are constructing.

Roadmap
-------

We currently have several releases planned on our development roadmap. Backward
incompatible changes, deprecations, and major features are noted for each of
these releases.

Releases follow `semantic versioning`_, and so it is generally recommended that
authors pin dependency on ``sphinx_rtd_theme`` to a version below the next major
version:

.. code:: console

    $ pip install "sphinx_rtd_theme<2.0.0"

.. _semantic versioning: http://semver.org/

.. _roadmap-release-1.0.0:

1.0.0
~~~~~

:Planned release date: August 2021

This release will be a slightly backwards incompatible release to follow the
:ref:`release-0.5.2` release. It will drop support for Sphinx 1.6, which is a rather old
release at this point.

This version will add official support for the Sphinx 4.x release series and
it resolves bugs with the latest release of Docutils, version 0.17.

Starting with this release, several deprecation warnings will be emitted at
build time:

Direct installation is deprecated
    Support for direct installation through GitHub is no longer a suggested
    installation method. In an effort to ease maintenance, compiled assets will
    eventually be removed from the theme repository. These files will only be
    included in the built packages/releases available on PyPI.

    We plan to start putting development releases up on PyPI more frequently, so
    that installation from the theme source repository is no longer necessary.

    Built assets are tentatively planned to be removed in version :ref:`roadmap-release-3.0.0`:.

HTML4 support is deprecated
    Support for the Sphinx HTML4 writer will be removed in the :ref:`roadmap-release-2.0.0`
    release.

.. _roadmap-release-1.1.0:

1.1.0
~~~~~

:Planned release date: 2021 Q3

We aim to follow up release :ref:`release-1.0.0` with at least one bug fix release in
the 1.x release series. The 1.1 release will not be adding any major features
and will instead mark the last release targeting projects with old dependencies
like Sphinx 1.8, HTML4, or required support for IE11.

.. _roadmap-release-2.0.0:

2.0.0
~~~~~

:Planned release date: 2022 Q1

This release will mark the beginning of a new round of feature development, as
well as a number of backward incompatible changes and deprecations.

Of note, the following backwards incompatible changes are planned for this
release:

Sphinx 1.x, Sphinx 2.x, and Docutils 0.16 will not be tested
    Official support will drop for these version, though they may still continue
    to work. Theme developers will not be testing these versions any longer.

HTML4 support will be removed
    Starting with this release, we will only support the HTML5 writer output,
    and builds attempting to use the HTML4 writer will fail. If you are still
    using the HTML4 writer, or have the ``html4_writer = True`` option in your
    Sphinx configuration file, you will need to either remove this option or pin
    your dependency to ``sphinx_rtd_theme<=2.0.0`` until you can.

    This option was suggested in the past to work around issues with HTML5
    support and should no longer be required to use a modern combination of this
    theme and Sphinx.

.. _roadmap-release-3.0.0:

3.0.0
~~~~~

This release is not yet planned, however there are plans to potentially replace
Wyrm with Bootstrap in a release after 2.0.

Also tentatively planned for this release is finally removing built CSS and
JavaScript assets from our repository. This will remove the ability to install
the package directly from GitHub, and instead users will be advised to install
development releases from PyPI
