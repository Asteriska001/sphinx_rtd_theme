Model/GNNReGVD Reference
----------

**Imports**: The initial block of code imports all necessary libraries and modules to be used throughout the code.

.. code:: python

    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import copy
    import torch.nn.functional as F
    from torch.nn import CrossEntropyLoss, MSELoss
    from .modules.GNN.modulesGNN import *
    from .modules.GNN.utils import preprocess_features, preprocess_adj
    from .modules.GNN.utils import *
    from framework.models.modules.transformers.transformers import *


**Device Configuration**:
This line checks if CUDA (used for GPU computations) is available. If it is, it sets the device to "cuda", otherwise it defaults to "cpu".
.. code:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

**Model Class**:
.. code:: python

    class Model(nn.Module):

The `Model` class extends `nn.Module`, which is the base class for all neural network modules in PyTorch.

* **__init__()**: The constructor initializes the model with encoder, configuration, tokenizer, and other arguments.
    - **Parameters**: 
        - encoder: the encoding mechanism (like BERT, RoBERTa etc.)
        - config: configuration object with various model settings.
        - tokenizer: tool for tokenizing input data.
        - args: other arguments.
    - **Attributes**:
        - encoder: the model's encoder
        - config: the model's configuration
        - tokenizer: the model's tokenizer
        - args: other attributes for the model

* **forward()**: The main function where forward propagation happens.
    - **Parameters**: 
        - input_ids: IDs representing input tokens
        - labels: labels for the input data.
    - **Output**: 
        - If labels are provided: return binary cross-entropy loss and probabilities
        - If no labels: return probabilities.

**PredictionClassification Class**:
.. code:: python

    class PredictionClassification(nn.Module):

This class is a sentence-level classification head.

* **__init__()**: The constructor initializes layers like dense, dropout and out_proj based on given configuration and arguments.
    - **Parameters**: 
        - config: configuration object.
        - args: other arguments.
        - input_size: Size of input features. Defaults to `args.hidden_size` if not provided.

* **forward()**: Forward propagation through layers.
    - **Parameters**: features
    - **Output**: logits after passing through all layers.

**GNNReGVD Class**:
.. code:: python

    class GNNReGVD(nn.Module):


This is a Graph Neural Network model for vulnerability detection.

* **__init__()**: The constructor initializes various components of the model like encoder, tokenizer, configurations, embeddings, GNN layers and classifier.
    - **Parameters**: 
        - encoder: the encoding mechanism.
        - config: configuration object.
        - tokenizer: tool for tokenizing input data.
        - args: other arguments.

* **forward()**: The main function where forward propagation happens.
    - **Parameters**: 
        - input_ids: IDs representing input tokens.
        - labels: labels for the input data.
    - **Output**: 
        - If labels are provided: return binary cross-entropy loss and probabilities.
        - If no labels: return probabilities.

----------

**Available Interface Functions**:
1. `Model.__init__(encoder, config, tokenizer, args)`
2. `Model.forward(input_ids=None, labels=None)`
3. `PredictionClassification.__init__(config, args, input_size=None)`
4. `PredictionClassification.forward(features)`
5. `GNNReGVD.__init__(encoder, config, tokenizer, args)`
6. `GNNReGVD.forward(input_ids=None, labels=None)`

----------
