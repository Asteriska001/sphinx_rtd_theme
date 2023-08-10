Configuration File Guide
========================

This configuration file is used for the framework targeting vulnerability detection tasks. The sections below detail the purpose of each configuration field.

We will use a configuration file  ``regvd_reveal.yaml``  as an example

.. code:: console

    #raw command line: --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
    #	--do_eval --do_test --do_train --train_data_file=../dataset/train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test.jsonl \
    #	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
    #	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
    #	--seed 123456 2>&1 | tee $logp/training_log.txt

    DEVICE          : cuda              # device used for training and evaluation (cpu, cuda, cuda0, cuda1, ...)
    SAVE_DIR        : 'output'         # output folder name used for saving the model, logs and inference results

    MODEL:                                    
    NAME          : GNNReGVD                                        # name of the model you are using
    BACKBONE      :
    PARAMS:                                                   # model variant
        encoder       : 
        config        :
        tokenizer     : 'roberta'
        args          :
            config_name : ''
            gnn     : 'ReGGNN'
            feature_dim_size: 768
            hidden_size: 256
            num_GNN_layers: 2
            model_name_or_path: microsoft/graphcodebert-base
            remove_residual: False
            att_op: 'mul'
            num_classes : 2
            format: 'uni'
            window_size: 5
    PRETRAINED    : 'checkpoints/backbones/xx.pth'              # backbone model's weight 

    DATASET:
    NAME          : REVEAL                                         # dataset name to be trained with (camvid, cityscapes, ade20k)
    ROOT          : 'data/ADEChallengeData2016'                         # dataset root path
    PARAMS:
        tokenizer   : 'roberta'
        args        :
        train_data_file: '../dataset/train.jsonl'
        eval_data_file : '../dataset/valid.jsonl'
        test_data_file : '../dataset/test.jsonl'
        block_size     : 400
        training_percent: 1.0
    PREPROCESS:
        ENABLE      : False #True
        COMPOSE     : [ 
                    "Normalize",
                    "PadSequence",
                    "OneHotEncode"
            ]

    TRAIN:
    INPUT_SIZE    : 128
    BATCH_SIZE    : 128               # batch size used to train
    EPOCHS        : 2             # number of epochs to train
    EVAL_INTERVAL : 50              # evaluation interval during training
    AMP           : false           # use AMP in training
    DDP           : false           # use DDP training

    LOSS:
    NAME          : CrossEntropy          # loss function name (ohemce, ce, dice)
    CLS_WEIGHTS   : false            # use class weights in loss calculation

    OPTIMIZER:
    NAME          : adamw           # optimizer name
    LR            : 0.001           # initial learning rate used in optimizer
    WEIGHT_DECAY  : 0.01            # decay rate used in optimizer 

    SCHEDULER:
    NAME          : warmuppolylr    # scheduler name
    POWER         : 0.9             # scheduler power
    WARMUP        : 0              # warmup epochs used in scheduler
    WARMUP_RATIO  : 0.1             # warmup ratio
    
    EVAL:
    INPUT_SIZE    : 128
    MODEL_PATH    : ''  # trained model file path
    INPUT_SIZE    : 128                                                         # evaluation input size            

    TEST:
    MODEL_PATH    : ''  # trained model file path
    FILE          : 'assests/codes'                                                         # filename or foldername 
    INPUT_SIZE    : 128                                                            # inference input size


Device Configuration
--------------------

- **DEVICE**: The device used for training and evaluation (e.g., cpu, cuda, cuda0, cuda1, etc.).
- **SAVE_DIR**: Output folder name used for saving the model, logs, and inference results.

Model Configuration
-------------------

MODEL
^^^^^^

- **NAME**: The name of the model you are using.
- **BACKBONE**: The backbone part of the model.
- **PARAMS**: Model parameters.
    - **encoder**: Encoder.
    - **config**: Configuration.
    - **tokenizer**: Tokenizer for handling text.
    - **args**: Other arguments.
- **PRETRAINED**: Pretrained weights path for the backbone model.

Dataset Configuration
---------------------

DATASET
^^^^^^^

- **NAME**: The dataset name to be trained with.
- **ROOT**: The root path of the dataset.
- **PARAMS**: Other parameters.
- **PREPROCESS**: Preprocessing parameters.

Training Configuration
----------------------

TRAIN
^^^^^

- **INPUT_SIZE**: Input size.
- **BATCH_SIZE**: Batch size used to train.
- **EPOCHS**: Number of epochs to train.
- **EVAL_INTERVAL**: Evaluation interval during training.
- **AMP**: Whether to use AMP in training.
- **DDP**: Whether to use Distributed Data Parallel (DDP) training.

Loss Configuration
------------------

LOSS
^^^^

- **NAME**: Loss function name.
- **CLS_WEIGHTS**: Whether to use class weights in loss calculation.

Optimizer Configuration
-----------------------

OPTIMIZER
^^^^^^^^^

- **NAME**: Optimizer name.
- **LR**: Initial learning rate used in the optimizer.
- **WEIGHT_DECAY**: Decay rate used in the optimizer.

Scheduler Configuration
-----------------------

SCHEDULER
^^^^^^^^^

- **NAME**: Scheduler name.
- **POWER**: Scheduler power.
- **WARMUP**: Warmup epochs used in the scheduler.
- **WARMUP_RATIO**: Warmup ratio.

Evaluation Configuration
------------------------

EVAL
^^^^

- **INPUT_SIZE**: Input size.
- **MODEL_PATH**: Trained model file path.
- **INPUT_SIZE**: Evaluation input size.

Testing Configuration
---------------------

TEST
^^^^

- **MODEL_PATH**: Trained model file path.
- **FILE**: Filename or folder name.
- **INPUT_SIZE**: Inference input size.

Finally, the configuration can be used for model training or inference by simply entering the following command. 

.. code:: python

    #training
    python tools/train.py --cfg regvd_reveal.yaml
    #evaluation
    python tools/val.py --cfg regvd_reveal.yaml
    #inference
    python tools/infer.py --cfg regvd_reveal.yaml
