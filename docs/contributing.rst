..
  Generated from CONTRIBUTING.md. Do not edit!

Contributing 
=============

**Table of Contents**

-  `How to Contribute <#how-to-contribute>`__
-  `Pull Requests <#pull-requests>`__
-  `Code Style <#code-style>`__

--------------

How to Contribute
-----------------

We want to make contributing to ASTERIA as easy and transparent as
possible. The most helpful ways to contribute are:

1. Provide feedback.

   -  `Report
      bugs <https://github.com/Asteriska001/ASTERIA-Detection>`__.
      In particular, it’s important to report any crash or correctness
      bug. We use GitHub issues to track public bugs. Please ensure your
      description is clear and has sufficient instructions to be able to
      reproduce the issue.
   -  Report issues when the documentation is incomplete or unclear, or
      an error message could be improved.
   -  Make feature requests. Let us know if you have a use case that is
      not well supported, including as much detail as possible.

2. Contribute to the ASTERIA ecosystem.

   -  Pull requests. Please see below for details. The easiest way to
      get stuck is to grab an `unassigned “Good first issue”
      ticket <https://github.com/Asteriska001/ASTERIA-Detection/issues?q=is%3Aopen+is%3Aissue+no%3Aassignee+label%3A%22Good+first+issue%22>`__!
   -  Add new features not on `the
      roadmap <https://github.com/Asteriska001/ASTERIA-Detection/about.html#roadmap>`__.
      Examples could include adding support for new models, producing
      research results using ASTERIA, etc.

Pull Requests
-------------

We actively welcome your pull requests.

1. Fork `the repo <https://github.com/Asteriska001/ASTERIA-Detection>`__
   and create your branch from ``development``.
2. Follow the instructions for `building from
   source <https://github.com/Asteriska001/ASTERIA-Detection/blob/development/INSTALL.md>`__
   to set up your environment.
3. If you’ve added code that should be tested, add tests.
4. If you’ve changed APIs, update the
   `documentation <https://github.com/Asteriska001/ASTERIA-Detection/tree/development/docs/source>`__.
5. Ensure the ``make test`` suite passes.
6. Make sure your code lints (see `Code Style <#code-style>`__ below).


Code Style
----------

We want to ease the burden of code formatting using tools. Our code
style is simple:

-  Python:
   `black <https://github.com/psf/black/blob/master/docs/the_black_code_style.md>`__
   and `isort <https://pypi.org/project/isort/>`__.
-  C++: `Google C++
   style <https://google.github.io/styleguide/cppguide.html>`__ with 100
   character line length and ``camelCaseFunctionNames()``.

We use `pre-commit <https://pre-commit.com/>`__ to ensure that code is
formatted prior to committing. Before submitting pull requests, please
run pre-commit. See
`INSTALL.md <https://github.com/Asteriska001/ASTERIA-Detection/blob/development/INSTALL.md>`__
for installation and usage instructions.

Other common sense rules we encourage are:

-  Prefer descriptive names over short ones.
-  Split complex code into small units.
-  When writing new features, add tests.
-  Make tests deterministic.
-  Prefer easy-to-use code over easy-to-read, and easy-to-read code over
   easy-to-write.
