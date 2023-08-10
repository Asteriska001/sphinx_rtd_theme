Folder structure
================

.. code:: console
    
    .
    ├── LICENSE
    ├── README.md
    ├── configs                 #configuration examples
    │   ├── devign_reveal.yaml  #devign + reveal example
    │   ├── linevd_reveal.yaml  #linevd + reveal example
    │   ├── regvd_reveal.yaml   #regvd + reveal example
    │   └── vulDeepecker_reveal.yaml
    ├── external                #external components
    │   └── MSR_20_CODE_DATASET #MSR_20_CODE_DATASET.
    │       ├── LICENSE
    │       ├── README.md
    │       ├── all_c_cpp_release2.0.csv
    │       ├── notebooks
    │       │   ├── AllProjects2Lang.ipynb
    │       │   ├── Android_csv.csv.ipynb
    │       │   ├── Chrome_csv.ipynb
    │       │   ├── ImageMagick_csv.ipynb
    │       │   ├── Linux_csv.ipynb
    │       │   ├── all_cpp_c_project_with_chrome_android.ipynb
    │       │   ├── exploreAllCVEDetailsCSV.ipynb
    │       │   ├── exploreAndroidCSV.ipynb
    │       │   ├── exploreChromeCSV.ipynb
    │       │   ├── get_response.ipynb
    │       │   ├── map_chrome_csv_to_cve.ipynb
    │       │   └── statistics_plot.ipynb
    │       ├── other_data
    │       │   ├── all_CVE_details_output.csv
    │       │   ├── all_linkNotNull.csv
    │       │   ├── all_linkNotNull_withProject.csv
    │       │   └── bugs.csv
    │       └── scripts
    │           ├── get_commit_info.py
    │           └── scrape_all_the_cve.py
    ├── framework               #framework main source code
    │   ├── __init__.py
    │   ├── dataset.py          #framework dataset mod.
    │   ├── datasets            #framework datasets dir. Storing datasets' implementations
    │   │   ├── __init__.py
    │   │   └── reveal.py       #the REVEAL Dataset Implementation
    │   ├── errors              #framework errors dir. Storing errors' implementations
    │   │   ├── __init__.py
    │   │   ├── dataset_errors.py #dataset errors implementation
    │   │   └── download_errors.py #download errors implementation
    │   ├── losses.py           #framework losses mod. including losses implementations and some loss related methods
    │   ├── metrics.py          #framework metrics mod. including metrics implementations and some metrics related methods
    │   ├── model.py            #framework model mod. including model related methods
    │   ├── models              #framework models dir. Storing models' implementations
    │   │   ├── Concoction.py   #the Concoction Implementation
    │   │   ├── Devign.py       #the Devign Implementation
    │   │   ├── GNNReGVD.py     #the GNNReGVD Implementation
    │   │   ├── TextCNN.py      #the TextCNN Implementation
    │   │   ├── VulDeePecker.py #the VuldeePecker Implementation
    │   │   ├── __init__.py
    │   │   ├── lineVul.py
    │   │   └── modules         #some network componments may be used
    │   │       ├── GNN
    │   │       │   ├── modulesGNN.py
    │   │       │   └── utils.py
    │   │       └── transformers
    │   │           └── transformers.py
    │   ├── optimizers.py       #framework optimizers mod. including optimizers implementations and optimizers related methods.
    │   ├── preprocess.py       #framework preprocess mod. including preprocess implementations and preprocess related methods
    │   ├── representations     #framework representations dir. Storing representations' Implementations
    │   │   ├── __init__.py
    │   │   ├── ast_graphs.py   #ast graph representation module. 
    │   │   ├── ast_graphs_test.py
    │   │   ├── common.py
    │   │   ├── common_test.py
    │   │   ├── extractors      #representation extractors. including the "Clang" and 'LLVM_IR'
    │   │   │   ├── CMakeLists.txt
    │   │   │   ├── __init__.py
    │   │   │   ├── clang_ast
    │   │   │   │   ├── CMakeLists.txt
    │   │   │   │   ├── clang_extractor.cc
    │   │   │   │   ├── clang_extractor.h
    │   │   │   │   ├── clang_extractor_test.cc
    │   │   │   │   ├── clang_graph_frontendaction.cc
    │   │   │   │   ├── clang_graph_frontendaction.h
    │   │   │   │   ├── clang_seq_frontendaction.cc
    │   │   │   │   └── clang_seq_frontendaction.h
    │   │   │   ├── common
    │   │   │   │   ├── clang_driver.cc
    │   │   │   │   ├── clang_driver.h
    │   │   │   │   ├── clang_driver_test.cc
    │   │   │   │   ├── common_test.h
    │   │   │   │   └── visitor.h
    │   │   │   ├── extractors.cc
    │   │   │   ├── extractors_test.py
    │   │   │   └── llvm_ir
    │   │   │       ├── CMakeLists.txt
    │   │   │       ├── llvm_extractor.cc
    │   │   │       ├── llvm_extractor.h
    │   │   │       ├── llvm_extractor_test.cc
    │   │   │       ├── llvm_graph_funcinfo.cc
    │   │   │       ├── llvm_graph_funcinfo.h
    │   │   │       ├── llvm_graph_pass.cc
    │   │   │       ├── llvm_graph_pass.h
    │   │   │       ├── llvm_pass_test.cc
    │   │   │       ├── llvm_seq_pass.cc
    │   │   │       └── llvm_seq_pass.h
    │   │   ├── llvm_graphs.py
    │   │   ├── llvm_graphs_test.py
    │   │   ├── llvm_seq.py
    │   │   ├── llvm_seq_test.py
    │   │   ├── syntax_seq.py
    │   │   ├── syntax_seq_test.py
    │   │   └── vectorizers         #vectorizers representation module.
    │   │       ├── vectorizer.py
    │   │       └── word2vec.py
    │   ├── schedulers.py           #framework scheduler module. including schedulers Implementation and some methods.
    │   └── utils
    │       ├── __init__.py
    │       └── utils.py
    ├── notebooks                   #Easy to use Jupyter notebooks
    │   └── tutorials.ipynb
    ├── scripts                     #some useful scripts for the framework
    │   └── Normalization
    │       ├── Normalization.py
    │       └── clean_gadget.py
    ├── setup.py                    #framework's setup script.
    └── tools                       #the main interface of the framework to train/val models.
        ├── export.py               #export model weights.
        ├── train.py                #training interface.
        └── val.py                  #evaluation interface.


In the folder structure above:

- ``framework/*`` is the core source code of the framework.
- ``framework/models`` 、 ``framework/datasets`` and ``framework/representations`` is the directory where store the core components's Implementation. The addition and development of the core model with other components should be modified here.
- ``tools/train.py`` and ``val.py`` are the core entry point for users to interact with the framework and contains the complete model training as well as validation process, if you want to have a more complete understanding of the framework, you can browse them.
- ``notebooks/`` This directory hosts a number of pre-built jupyter notebooks to get you up and running and learning the framework.