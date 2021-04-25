# Electromyogram-based classification of hand and finger gestures using artificial neural networks

The codes for `2.6 Modeling` in Materials and Methods of the paper are available.

*** Note that all codes are executable ONLY if your own data exist

## Datasets

- Here, the datasets we used in the paper can not be released for personal information protection.

- Instead, you can identify a sample dataset. Please refer to `sample_dataset.csv`
  - `sample_dataset.csv` shows the examples of the datasets used for modeling (Note that this is not a real subject's dataset)
  - In the "label" column, 0: rest, 1: rock, 2: scissor, 3: paper, 4: one, 5: three, 6: four, 7: good, 8: okay, 9: gun 

## Modeling

### 1. ANN gridsearch

- This code covers the gridsearch process and training with the best params for ANN

- ANN gridsearch was conducted with TensorFlow 2.0

- See `ANN_gridsearch.py`

- You can see the sample tensorboard results of gridsearch with the mean of cv accuracies [here](https://tensorboard.dev/experiment/OlkPqHnqSv6LG4QlVDDwkQ/).  
  (this is the same result with `tensorboard --logdir=./logs/hparam_tuning_results` if you run the code with your own datasets)
- You can see the sample tensorboard results of gridsearch with accuracy and loss changes according to epochs for each cv [here](https://tensorboard.dev/experiment/0JqopeYzRI2aYjKUdhVW6w/).  
  (this is the same result with `tensorboard --logdir=./logs/hparam_tuning` if you run the code with your own datasets)

### 2. SVM, RF, LR gridsearch

- This code covers the gridsearch process and training with the best params for SVM, RF, and LR

- ML gridsearch was conducted with scikit-learn ver. 0.23.2

- See `ML_gridsearch.py`
