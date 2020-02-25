# Estimate article body for a given title

## Currently implemented algorithms

- Latent Dirichlet Allocation (LDA)
- Doc2Vec


## Structure

- main.py

    This sarves as an *interface* of this software.

- engine.py

    This module is a container for various models.
    Each model is supposed to provide functionalities regarding the model such as train, evaluate, and predict.
    Also, each model class should have unified interface to ensure that models can be easily swapped.

- data.py

    This module provides functions for loading/storing data.

- utils.py

    Utility module.


## Usage

If you launch ```main.py```, it'll load data, train a model, evaluate it, and finally make a prediction on test data.

### Example

```bash
python3 main.py --model_type doc2vec
```

### Options
If you launch ```main.py``` with ```--help``` option, it'll show available options.

```bash
python3 main.py --help

# usage: python3 main.py [-h] [--data_path DATA_PATH]
#                        [--train_data_path TRAIN_DATA_PATH]
#                        [--val_data_path VAL_DATA_PATH]
#                        [--test_data_path TEST_DATA_PATH]
#                        [--output_path OUTPUT_PATH] [--cache_dir CACHE_DIR]
#                        [--model_type MODEL_TYPE]
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --data_path DATA_PATH
#                         Path to the doc/title data file.
#                         Default: data/exam_data1.json
#   --train_data_path TRAIN_DATA_PATH
#                         Path to the train data file.
#                         Default: data/train_q.json
#   --val_data_path VAL_DATA_PATH
#                         Path to the validation data file.
#                         Default: data/val_q.json
#   --test_data_path TEST_DATA_PATH
#                         Path to the test data file.
#                         Default: data/test_q.json
#   --output_path OUTPUT_PATH
#                         Path to the model_output dir.
#                         Default: ./temp_output
#   --cache_dir CACHE_DIR
#                         Wehre to store/load the cache directory.
#                         Default: disable cache
#   --model_type MODEL_TYPE
#                         Mdoel selection.. [lda, doc2vec]
#                         Default: lda
```
