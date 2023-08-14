LLVM-Seq representation method
===============================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Imported Modules
----------------

.. code-block:: python

   from framework.representations.extractors import clang_driver_scoped_options
   from framework.representations.extractors.extractors import Visitor, ClangDriver, LLVMIRExtractor, llvm
   from framework.representations import common

Utility Functions
-----------------

These functions provide utilities for string manipulations.

- **merge_after_element_on_condition(elements, element_conditions)**
   
  Merges consecutive elements based on given conditions.

  :param elements: List of elements to merge
  :param element_conditions: List of conditions to check
  :return: List of merged elements
  
  **Example:** Given a list `['a', 'b', 'c', 'a', 'e']` and conditions `['a']`, it returns `['ab', 'c', 'ae']`.

- **filer_elements(elements, element_filter)**
  
  Filters out specified elements from a list.

  :param elements: List of elements to filter
  :param element_filter: List of elements to remove
  :return: Filtered list of elements

  **Example:** Given a list `['a', ' ', 'c']` and filters `[' ']`, it returns `['a', 'c']`.

- **strip_elements(elements, element_filters)**
   
  Strips specified characters from each element of a list.

  :param elements: List of elements to strip
  :param element_filters: List of characters to strip
  :return: List of stripped elements

  **Example:** Given a list `['a', ' b', 'c']` and filters `[' ']`, it returns `['a', 'b', 'c']`.

- **strip_function_name(elements)**

  Modifies function names in a specific format in a list of elements.

  :param elements: List of elements with function names
  :return: List of elements with modified function names

- **transform_elements(elements)**

  Applies a series of transformations to the input list of elements.

  :param elements: List of elements to transform
  :return: Transformed list of elements

LLVMSeqVisitor Class
--------------------

A Visitor class for LLVM sequences.

Attributes:
   
   - S: A list to store the transformed sequence

Methods:

- **visit(v)**

  Processes the visited item based on its type.

  :param v: The visited item
  

LLVMSeqBuilder Class
--------------------

Class to build LLVM sequences from source code.

Attributes:

   - clang_driver: An instance of ClangDriver
   - extractor: An instance of LLVMIRExtractor

Methods:

- **__init__(clang_driver=None)**

  Initializes the LLVMSeqBuilder.

  :param clang_driver: (Optional) An instance of ClangDriver
  
- **string_to_info(src, additional_include_dir=None, filename=None)**

  Converts a source string to information.

  :param src: Source code as a string
  :param additional_include_dir: (Optional) Additional include directory
  :param filename: (Optional) Name of the file
  :return: Extracted information from the source

- **info_to_representation(info, visitor=LLVMSeqVisitor)**

  Converts the provided info to its representation using the visitor.

  :param info: Information to be processed
  :param visitor: (Optional) A visitor instance, defaults to LLVMSeqVisitor
  :return: A representation of the info
