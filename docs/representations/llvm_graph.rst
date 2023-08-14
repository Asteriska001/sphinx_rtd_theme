LLVM-graph representation method
=================================

This module provides functionality to build graph representations based on LLVM's intermediate representation (IR). The representations can be used for tasks like vulnerability detection.

Imports
-------
The module leverages NetworkX for graph manipulation and several custom extractor classes.

.. code-block:: python

   import networkx as nx
   from framework.representations.extractors import clang_driver_scoped_options
   ...

Classes
-------

LLVMCDFGVisitor
^^^^^^^^^^^^^^^^
Constructs a control and data flow graph representation.

.. class:: LLVMCDFGVisitor

   .. method:: __init__(self)

      Initializes the visitor with an empty directed graph and sets the edge types.

   .. method:: visit(self, v)

      :param v: A node representing an LLVM element (e.g., Function, BasicBlock, Instruction).
      :type v: llvm.graph.Element
      
      Visits the LLVM elements and builds the graph nodes and edges based on control and data flow.

LLVMCDFGCallVisitor
^^^^^^^^^^^^^^^^^^^
Similar to `LLVMCDFGVisitor` but also includes call edges for function calls.

.. class:: LLVMCDFGCallVisitor

   (Similar to LLVMCDFGVisitor methods but also handles function call relationships)

LLVMCDFGPlusVisitor
^^^^^^^^^^^^^^^^^^^
Extends `LLVMCDFGVisitor` to include basic block relationships.

.. class:: LLVMCDFGPlusVisitor

   (Similar to LLVMCDFGVisitor methods but with additional handling for basic blocks)

LLVMProGraMLVisitor
^^^^^^^^^^^^^^^^^^^
Builds a program graph representation considering control, data, and call flows.

.. class:: LLVMProGraMLVisitor

   (Methods similar to previous visitors but with specific adjustments for program graphs)

LLVMGraphBuilder
^^^^^^^^^^^^^^^^
Uses LLVM's intermediate representation (IR) to build various graph representations.

.. class:: LLVMGraphBuilder

   .. method:: __init__(self, clang_driver=None)

      :param clang_driver: Optional driver for extracting LLVM IR.
      :type clang_driver: ClangDriver or None
      
      Initializes the graph builder with an optional clang driver.

   .. method:: string_to_info(self, src, additional_include_dir=None, filename=None)

      :param src: Source code to be transformed.
      :type src: str
      :param additional_include_dir: Additional directories to search for includes.
      :type additional_include_dir: str or None
      :param filename: Optional name for the source file.
      :type filename: str or None
      :return: Graph representation of the provided source code.
      :rtype: Graph

      Transforms source code into its graph representation.

   .. method:: info_to_representation(self, info, visitor=LLVMCDFGVisitor)

      :param info: LLVM information.
      :type info: llvm.graph.Info
      :param visitor: Type of graph visitor to use.
      :type visitor: Visitor
      :return: Graph representation.
      :rtype: common.Graph

      Converts LLVM information into the desired graph representation.

