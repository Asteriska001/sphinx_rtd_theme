ast_graphs.py Documentation
===========================

The `ast_graphs.py` module provides utilities to build Abstract Syntax Tree (AST) graphs from code sources using Clang, with specific focuses on various relations such as data references, control flows, and tokens.

Functions
---------

- **filter_type(type: str) -> str**
  Filters the input type string and maps it to a more generic type representation.
  
  Parameters:
  
  - `type (str)`: Input type string.
  
  Returns:
  
  - `str`: A filtered type string.

- **add_ast_edges(g: nx.MultiDiGraph, node)**
  Adds AST edges to the graph representing parent-child relationships in the AST.
  
  Parameters:
  
  - `g (nx.MultiDiGraph)`: The graph to which edges will be added.
  - `node`: A node object that may contain AST relations.

- **add_ref_edges(g: nx.MultiDiGraph, node)**
  Adds edges representing data references for a given node.
  
  Parameters:
  
  - `g (nx.MultiDiGraph)`: The graph to which edges will be added.
  - `node`: A node object that may contain data reference relations.

- **add_cfg_edges(g: nx.MultiDiGraph, node)**
  Adds edges for control flow relations for a given node.
  
  Parameters:
  
  - `g (nx.MultiDiGraph)`: The graph to which edges will be added.
  - `node`: A node object that may contain control flow relations.

- **add_token_ast_edges(g: nx.MultiDiGraph, node)**
  Adds edges connecting tokens to the nearest AST node covering them.
  
  Parameters:
  
  - `g (nx.MultiDiGraph)`: The graph to which edges will be added.
  - `node`: A node object that may contain token information.

Classes
-------

- **ASTVisitor**
  A visitor class to build a graph with edges representing AST relationships.
  
  Methods:
  
  - `visit(v)`: Visits a node `v` and adds relevant AST edges to the graph.

- **ASTDataVisitor**
  Extends `ASTVisitor` and additionally captures data reference relationships.
  
  Methods:
  
  - `visit(v)`: Visits a node `v` and adds relevant AST and data reference edges to the graph.

- **ASTDataCFGVisitor**
  Extends `ASTDataVisitor` and additionally captures control flow relationships.
  
  Methods:
  
  - `visit(v)`: Visits a node `v` and adds AST, data reference, and control flow edges to the graph.

- **ASTDataCFGTokenVisitor**
  Extends `ASTDataCFGVisitor` and captures token relationships.
  
  Methods:
  
  - `visit(v)`: Visits a node `v` and adds AST, data reference, control flow, and token edges to the graph.

- **ASTGraphBuilder**
  Main class to build AST graphs.
  
  Methods:
  
  - `string_to_info(src, additional_include_dir=None, filename=None)`: Extracts graph information from a source string.
  - `info_to_representation(info, visitor=ASTDataVisitor)`: Converts extracted information into a graph representation.

