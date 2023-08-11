**Metrics.py Documentation**
===========================

The `Metrics.py` module is designed for evaluating model predictions in the context of a deep learning framework aimed at vulnerability detection. It offers a variety of metrics, primarily used in classification tasks, to determine the performance of the model.

Imports:
----------

.. code-block:: python

    import torch
    from torch import Tensor
    from typing import Tuple,List
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score

These imports include:

- Essential PyTorch functions and types.
- Python's type hinting module for better clarity in function signatures.
- Scikit-learn's metrics for efficient and reliable computation of classification metrics.

Class Definition: Metrics
--------------------------

.. code-block:: python

    class Metrics:
        ...

This class provides a suite of methods for calculating various performance metrics, including accuracy, F1-score, recall, precision, ROC AUC score, and PR AUC score.

### Initializer: `__init__`

.. code-block:: python

    def __init__(self, num_classes: int, device) -> None:
        ...

Initializes the `Metrics` object with a specified number of classes and a device (`cpu` or `cuda`). This method sets up empty data structures to hold predictions and targets for metric calculations.

**Parameters**:

- `num_classes` (*int*): The number of classes in the classification task.
- `device` (*torch.device*): The device (CPU or CUDA) to allocate tensors on.

### Method: `update`

.. code-block:: python

    def update(self, pred: Tensor, target: Tensor) -> None:
        ...

Updates internal data structures with new model predictions and ground truth targets.

**Parameters**:

- `pred` (*Tensor*): Model predictions.
- `target` (*Tensor*): Ground truth labels.

### Metric Calculation Methods:

1. **`compute_acc`**:

    Computes the overall accuracy of the model's predictions.

    **Returns**:

    - *float*: Overall accuracy.

2. **`compute_f1`**:

    Computes the macro-average F1 score.

    **Returns**:

    - *float*: Macro-average F1 score.

3. **`compute_rec`**:

    Computes the macro-average recall.

    **Returns**:

    - *float*: Macro-average recall.

4. **`compute_prec`**:

    Computes the macro-average precision.

    **Returns**:

    - *float*: Macro-average precision.

5. **`compute_roc_auc`**:

    Computes the ROC AUC score.

    **Returns**:

    - *float*: ROC AUC score.

6. **`compute_pr_auc`**:

    Computes the average precision score.

    **Returns**:

    - *float*: Average precision score.


Reference
----------

For this module:

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Scikit-learn Metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

---

This concludes the comprehensive documentation of the `Metrics.py` module.