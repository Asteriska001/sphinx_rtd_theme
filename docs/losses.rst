Available Loss methods
======================

The code presents a module for defining and obtaining various loss functions catered to the deep learning framework for vulnerability detection.

.. code:: python

    import torch
    from torch import nn, Tensor
    from torch.nn import functional as F

- **Import Statements**:
  - Import necessary libraries from the PyTorch framework. These libraries will be used to define custom loss classes.

.. code:: python

    __all__ = ['CrossEntropy', 'MSELoss', 'BCELoss', 'NLLLoss', 'KLDivLoss', 'HingeLoss', 'SmoothL1Loss']

- **`__all__` Declaration**:
  - A list of strings indicating all the available loss function names that are implemented in this module. This is used for validation purposes later on.

Each of the following classes defines a specific loss function:

1. **CrossEntropy**:
   - Implements the Cross-Entropy Loss, commonly used for classification tasks.

.. code:: python

    class CrossEntropy(nn.Module):
        def forward(self, input: Tensor, target: Tensor):
            return F.cross_entropy(input, target)


2. **MSELoss**:
   - Implements the Mean Squared Error Loss, commonly used for regression tasks.

.. code:: python

    class MSELoss(nn.Module):
        def forward(self, input: Tensor, target: Tensor):
            return F.mse_loss(input, target)


3. **BCELoss**:
   - Implements the Binary Cross-Entropy Loss with Logits, used for binary classification tasks.

.. code:: python

    class BCELoss(nn.Module):
        def forward(self, input: Tensor, target: Tensor):
            return F.binary_cross_entropy_with_logits(input, target)


4. **NLLLoss**:
   - Implements the Negative Log-Likelihood Loss, used in conjunction with log softmax.

.. code:: python

    class NLLLoss(nn.Module):
        def forward(self, input: Tensor, target: Tensor):
            return F.nll_loss(input, target)


5. **KLDivLoss**:
   - Implements the Kullback-Leibler Divergence Loss, used to measure how one probability distribution diverges from a second, expected probability distribution.

.. code:: python

    class KLDivLoss(nn.Module):
        def forward(self, input: Tensor, target: Tensor):
            return F.kl_div(input, target)


6. **HingeLoss**:
   - Implements the Hinge Embedding Loss, used often in ranking problems.

.. code:: python

    class HingeLoss(nn.Module):
        def forward(self, input: Tensor, target: Tensor):
            return F.hinge_embedding_loss(input, target)


7. **SmoothL1Loss**:
   - Implements the Smooth L1 Loss or Huber Loss, which is less sensitive to outliers than the Mean Squared Error Loss.

.. code:: python

    class SmoothL1Loss(nn.Module):
        def forward(self, input: Tensor, target: Tensor):
            return F.smooth_l1_loss(input, target)


.. code:: python

    def get_loss(loss_fn_name: str = 'CrossEntropy'):
        assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
        return eval(loss_fn_name)()


- **get_loss Function**:
  - This function is used to retrieve the desired loss function based on its name (`loss_fn_name`).
  - An assertion checks if the provided loss function name is available in the module. If not, it raises an exception.
  - The `eval` function dynamically evaluates the provided loss name string and returns the corresponding loss function.


The `losses.py` module offers a variety of loss functions that can be easily accessed and utilized within a deep learning framework. By wrapping PyTorch's functional API into distinct classes, the module allows for a modular and clean approach to handling loss functions for different tasks. The `get_loss` function further streamlines the process by enabling easy retrieval of a specified loss function based on its name. This design makes it simple to extend the module by adding more loss functions if needed in the future.