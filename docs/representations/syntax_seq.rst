Syntax Sequence representations method
==========================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Imported Modules
----------------

.. code-block:: python

   from framework.representations.extractors import clang_driver_scoped_options
   from framework.representations.extractors.extractors import Visitor, ClangDriver, ClangExtractor, clang
   from framework.representations import common

Visitors
--------

1. **SyntaxSeqVisitor**

   A Visitor subclass that populates a list with names of TokenInfo objects it visits.

   Attributes:
      - S: List to store visited token names.

   Methods:
      - **visit(v)**
        :param v: Visited token. Appends its name to the S list if it's an instance of clang.seq.TokenInfo.

2. **SyntaxTokenkindVisitor**

   A Visitor subclass that populates a list with kinds of TokenInfo objects it visits.

   Attributes:
      - S: List to store visited token kinds.

   Methods:
      - **visit(v)**
        :param v: Visited token. Appends its kind to the S list if it's an instance of clang.seq.TokenInfo.

3. **SyntaxTokenkindVariableVisitor**

   A Visitor subclass that processes TokenInfo objects with more detailed logic based on their kind and name.

   Attributes:
      - S: List to store processed tokens.

   Methods:
      - **visit(v)**
        :param v: Visited token. Depending on its kind and name, either appends its name or kind to the S list.

Builders
--------

**SyntaxSeqBuilder**

   A RepresentationBuilder subclass tailored to extract syntax sequences.

   Attributes:
      - __clang_driver: An instance of ClangDriver.
      - __extractor: An instance of ClangExtractor.

   Methods:
      - **__init__(clang_driver=None)**
        :param clang_driver: (Optional) An instance of ClangDriver. If not provided, a default one with specific settings is created.

      - **string_to_info(src, additional_include_dir=None, filename=None)**
        :param src: Source code string to process.
        :param additional_include_dir: (Optional) Additional include directory.
        :param filename: (Optional) Filename for the source.
        :return: Extracted information from the source string.

      - **info_to_representation(info, visitor=SyntaxTokenkindVariableVisitor)**
        :param info: Information extracted from the source.
        :param visitor: A visitor class to process the information. Defaults to SyntaxTokenkindVariableVisitor.
        :return: Sequence representation of the information.
