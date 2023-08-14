representations/common.py
=============

This module provides classes to handle abstract syntax tree graphs and sequences using the `networkx` library and visualizations using the `pygraphviz` library.

RepresentationBuilder
---------------------

.. class:: RepresentationBuilder()

    A class to maintain and operate on the tokens for a representation.

    .. method:: num_tokens()

       :returns: The number of unique tokens.
       :rtype: int

    .. method:: get_tokens()

       :returns: A list of all the unique tokens.
       :rtype: list

    .. method:: print_tokens()

       Prints a table view of the tokens along with their node IDs and counts.

Sequence
--------

.. class:: Sequence(S: str, token_types: list)

    Represents a sequence of tokens.

    .. method:: get_token_list()

       :returns: List of integers representing the sequence of tokens.
       :rtype: list[int]

    .. method:: size()

       :returns: Size of the sequence.
       :rtype: int

    .. method:: draw(width: int = 8, limit: int = 30, path: str = None)

       Visualizes the sequence.

       :param width: The width of the graph layout.
       :param limit: Limit to the number of tokens displayed.
       :param path: File path to save the graph visualization. If None, the visualization will not be saved.
       :returns: None

Graph
-----

.. class:: Graph(graph: networkx.Graph, node_types: list, edge_types: list)

    A representation of a graph with methods to manipulate and visualize it.

    .. method:: get_node_str_list()

       :returns: List of node attributes in string format.
       :rtype: list[str]

    .. method:: get_node_list()

       :returns: List of integers representing the nodes.
       :rtype: list[int]

    .. method:: get_edge_list()

       :returns: List of edges with source node, edge type, and target node.
       :rtype: list[tuple]

    .. method:: get_leaf_node_list()

       Returns the node indices for leaves of the graph.
       
       :returns: An ordered list of node indices.
       :rtype: list[int]

    .. method:: map_to_leaves(relations: dict = None)

       Map inner nodes of the graph to leaf nodes.

       :param relations: Specifies which edges indicate a parent-child relationship. If None, default relations will be used.
       :returns: A new graph where the mapping is applied.
       :rtype: Graph

    .. method:: size()

       :returns: The number of nodes in the graph.
       :rtype: int

    .. method:: draw(path: str = None, with_legend: bool = False, align_tokens: bool = True)

       Visualizes the graph.

       :param path: File path to save the graph visualization. If None, the visualization will not be saved.
       :param with_legend: Boolean indicating if legend should be added.
       :param align_tokens: Boolean indicating if tokens should be aligned.
       :returns: None
```

Detailed Code Explanation
-------------------------
- `RepresentationBuilder`: Provides methods to maintain and print a token representation.

- `Sequence`: Represents a sequence of tokens and allows you to visualize it.

- `Graph`: Represents a graph that offers functionalities like retrieving node and edge lists, mapping inner nodes to leaf nodes, and visualizing the graph. The visualization colors different types of edges with distinct colors and provides a legend for edge types.

The given code heavily utilizes the `networkx` library for graph manipulations and `pygraphviz` for visualizations. The graphs used are MultiDiGraphs (i.e., multi-directed graphs) that can have multiple edges between nodes. The visualizations offer the ability to include legends, and for the Graph, there's a special treatment for "leaf nodes", presumably nodes that represent the end of a branch in the graph.

Note: Make sure you have installed and properly set up the `networkx` and `pygraphviz` libraries before using this module.