schedulers.py Documentation
==============================

The `schedulers.py` module provides learning rate scheduling strategies for deep learning training loops. Learning rate scheduling is crucial for adapting the learning rate during training to ensure efficient and effective convergence. This module specifically offers strategies that cater to vulnerability detection tasks within deep learning frameworks.

Imports:
--------

.. code-block:: python

    import torch
    import math
    from torch.optim.lr_scheduler import _LRScheduler

These imports enable:

- Core PyTorch functionalities.
- Mathematical functions for specific scheduling strategies.
- PyTorch's base learning rate scheduler.

Class Definitions:
------------------

1. **PolyLR**
--------------

Polynomial decay learning rate scheduler.

.. code-block:: python

    class PolyLR(_LRScheduler):
        ...

- **`__init__`**:

    Initializes the `PolyLR` object.

    **Parameters**:

    - `optimizer`: Optimizer instance.
    - `max_iter`: Maximum number of iterations.
    - `decay_iter`: Iteration interval for decay.
    - `power`: Polynomial power for decay.
    - `last_epoch`: The last epoch (used for resuming training).

- **`get_lr`**:

    Computes and returns the learning rate for the current iteration based on polynomial decay.

2. **WarmupLR**
---------------

Base class for learning rate schedulers with warmup phase.

.. code-block:: python

    class WarmupLR(_LRScheduler):
        ...

- **`__init__`**:

    Initializes the `WarmupLR` object.

    **Parameters**:

    - `optimizer`: Optimizer instance.
    - `warmup_iter`: Number of warmup iterations.
    - `warmup_ratio`: Initial learning rate ratio during warmup.
    - `warmup`: Warmup type (`linear` or `exp`).
    - `last_epoch`: The last epoch (used for resuming training).

- **`get_lr`**:

    Computes and returns the learning rate for the current iteration based on warmup ratio or main ratio.

- **`get_lr_ratio`**:

    Chooses between warmup and main ratio depending on the iteration.

- **`get_main_ratio`**:

    Placeholder for the main learning rate calculation after warmup. Not implemented in this base class and must be implemented in derived classes.

- **`get_warmup_ratio`**:

    Computes the learning rate ratio during the warmup phase based on the specified strategy (`linear` or `exp`).

3. **WarmupPolyLR**
-------------------

Learning rate scheduler with warmup and polynomial decay.

.. code-block:: python

    class WarmupPolyLR(WarmupLR):
        ...

4. **WarmupExpLR**
-------------------

Learning rate scheduler with warmup and exponential decay.

.. code-block:: python

    class WarmupExpLR(WarmupLR):
        ...

5. **WarmupCosineLR**
-------------------

Learning rate scheduler with warmup and cosine annealing.

.. code-block:: python

    class WarmupCosineLR(WarmupLR):
        ...

Function Definitions:
---------------------

**get_scheduler**
-----------------

Factory function to retrieve the desired scheduler instance.

.. code-block:: python

    def get_scheduler(scheduler_name: str, optimizer, max_iter: int, power: int, warmup_iter: int, warmup_ratio: float):
        ...

**Parameters**:

- `scheduler_name`: Name of the desired scheduler.
- `optimizer`: Optimizer instance.
- Others: Various parameters depending on the scheduler type.

**Returns**:

- An instance of the desired learning rate scheduler.

Test Script:
------------

The script at the end of the module:

.. code-block:: python

    if __name__ == '__main__':
        ...

is a simple test script to visualize the behavior of the `WarmupPolyLR` scheduler. It initializes a sample neural network, defines an optimizer, and utilizes the scheduler for a specified number of iterations. The learning rates are then plotted to show the scheduling behavior.

Reference
---------

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Learning rate scheduling strategies: https://arxiv.org/abs/1706.02515

---
