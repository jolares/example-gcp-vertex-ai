"""A training pipeline for models built with Tensorflow"""
import os

import pandas as pd
from envyaml import EnvYAML

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .typings import TrainerConfig


print(tf.__version__)


## Load Job Config

ENV = os.environ('ENV', 'local')

CONFIG_FILE=os.environ.get(
    'TRAIN_CONFIG_FILE',
    f'{os.getcwd()}/config.yml'
)
config: TrainerConfig = EnvYAML(
    yaml_file=CONFIG_FILE,
    env_file=f'{os.getcwd()}/{ENV}.env',
)


## Load the dataset

data_config: TrainerConfig.DataConfig = config['data']

# Downloads the dataset to filesystem
dataset_path = keras.utils.get_file(
    data_config['filename'],
    data_config['uri'],
)

print(f"dataset_path: {dataset_path}")

# Imports the dataset to pandas dataframe
column_names = data_config['attributes']
target_name = data_config['target']
dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values = "?", comment='\t',
    sep=" ", skipinitialspace=True
)

dataset.tail()


## Pre-Process the Data

# Determines rows with empty values
dataset.isna().sum()

# Drops rows with empty values
dataset = dataset.dropna()

# One-Hot encodes categorical column "Origin"
dataset['Origin'] = dataset['Origin'].map({
  1: 'USA',
  2: 'Europe',
  3: 'Japan'
})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

dataset.tail()

## Construct Training, Validation, and Test Sets

train_data = dataset.sample(
  frac=data_config['train_size'],
  random_state=config['random_seed']
)
test_data = dataset.drop(train_data.index)

# ## Explore the data

# # Have a quick look at the joint distribution of a few pairs of columns from the training set.
# # Also look at the overall statistics:

# train_stats = train_data.describe()
# train_stats.pop(target_name)
# train_stats = train_stats.transpose()

# print(train_stats)


## Separate features and labels

train_labels = train_data.pop(target_name)
test_labels = test_data.pop(target_name)

# """ Normalize the data

# Look again at the `train_stats` block above and note how different the ranges of each feature are.

# It is good practice to normalize features that use different scales and ranges. Although the model *might* converge without feature normalization, it makes training more difficult, and it makes the resulting model dependent on the choice of units used in the input.

# Note: Although we intentionally generate these statistics from only the training dataset, these statistics will also be used to normalize the test dataset. We need to do that to project the test dataset into the same distribution that the model has been trained on.
# """

# def norm(x):
#   return (x - train_stats['mean']) / train_stats['std']

# train_data = norm(train_data)
# train_data = norm(test_data)

# """This normalized data is what we will use to train the model.

# Caution: The statistics used to normalize the inputs here (mean and standard deviation) need to be applied to any other data that is fed to the model, along with the one-hot encoding that we did earlier.  That includes the test set as well as live data when the model is used in production.

# ## The model

# ### Build the model

# Let's build our model. Here, we'll use a `Sequential` model with two densely connected hidden layers, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, `build_model`, since we'll create a second model, later on.
# """

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

  model.compile(
    loss='mse',
    optimizer=optimizer,
    metrics=['mae', 'mse']
  )
  return model

model = build_model()

"""### Inspect the model

Use the `.summary` method to print a simple description of the model
"""

model.summary()

"""Now try out the model. Take a batch of `10` examples from the training data and call `model.predict` on it.

It seems to be working, and it produces a result of the expected shape and type.

### Train the model

Train the model for 1000 epochs, and record the training and validation accuracy in the `history` object.

Visualize the model's training progress using the stats stored in the `history` object.

This graph shows little improvement, or even degradation in the validation error after about 100 epochs. Let's update the `model.fit` call to automatically stop training when the validation score doesn't improve. We'll use an *EarlyStopping callback* that tests a training condition for  every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training.

You can learn more about this callback [here](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
"""

## Train the model

train_config: TrainerConfig.TrainConfig = config['train']

model = build_model()

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10 # The amount of epochs to check for improvement
)

early_history = model.fit(
    train_data, train_labels,
    epochs=train_config['n_epochs'],
    validation_split = data_config['validation_size']
    callbacks=[early_stop]
)

## Export and save trained model

model.save(
    f"{train_config['output_path']}"
)