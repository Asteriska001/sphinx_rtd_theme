Training module References
=================

1. **Import Statements**: 
   - Essential libraries and modules are imported for building, training, and evaluating the model.

.. code:: python

    import torch
    import argparse
    import yaml
    import time
    import multiprocessing as mp
    from tabulate import tabulate
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter
    from torch.cuda.amp import GradScaler, autocast
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DistributedSampler, RandomSampler
    from torch import distributed as dist

    from framework.models import *
    from framework.datasets import *
    from framework.model import get_model
    from framework.dataset import get_dataset
    from framework.losses import get_loss
    from framework.schedulers import get_scheduler
    from framework.optimizers import get_optimizer
    from framework.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
    from val import evaluate

2. **Ordered Loading of YAML files**: 
   - This section provides a custom function to load YAML files in an ordered way, preserving the order of elements in dictionaries.

    .. code:: python

        from collections import OrderedDict

        def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
            class OrderedLoader(Loader):
                pass
            def construct_mapping(loader, node):
                loader.flatten_mapping(node)
                return object_pairs_hook(loader.construct_pairs(node))
            OrderedLoader.add_constructor(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                construct_mapping)
            return yaml.load(stream, OrderedLoader)


3. **Main Function (main)**:
   - The core function of the script responsible for preparing and training the model.

    .. code:: python

        def main(cfg, gpu, save_dir):


   a. **Initialization**: Retrieves the configuration, sets the device, splits the configuration into specific parts, and sets the number of epochs and learning rate.

   b. **Data Preparation**: 
      - Defines training and validation datasets using a function `get_dataset`.

   c. **Model Initialization**: 
      - Retrieves and prints the model, moves it to the appropriate device, and initializes Distributed Data Parallel if needed.

   d. **Data Loading**:
      - Defines the data loaders for training and validation.

   e. **Training Configuration**:
      - Initializes the loss function, optimizer, scheduler, gradient scaler, and TensorBoard writer.

   f. **Training Loop**: 
      - Includes the main loop for training epochs, where each batch of data is processed through the model, and loss is calculated and back-propagated.

   g. **Evaluation and Saving**: 
      - Periodically evaluates the model on the validation set and saves the best performing model.

   h. **Logging**: 
      - Logs the best accuracy and total training time.

3. **Script Execution (if __name__ == '__main__')**:
   - Defines and parses command-line arguments.
   - Loads the configuration file, fixes seeds, and sets up CUDA.
   - Calls the main function for execution.


The code is a typical deep learning training script, customized for a specific vulnerability detection task. It uses PyTorch as the deep learning framework, supporting distributed training with advanced features like mixed-precision training and custom data loading. The configuration is highly modular, allowing customization via a YAML file. 