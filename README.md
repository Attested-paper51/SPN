# SPN

**Link to Google Colab SPN Implementation:** [Google Colab Link](https://colab.research.google.com/drive/17N0gQ9FjhaTSPKXqTvw0Gnm4jfS14t3F?usp=sharing)

### Running the SPN Implementation in Google Colab

1. Click on the [Google Colab Link](https://colab.research.google.com/drive/17N0gQ9FjhaTSPKXqTvw0Gnm4jfS14t3F?usp=sharing) to open the notebook.

2. Once the notebook is open, go through each cell one by one and execute them by clicking on the play button next to each cell.

3. Make sure to execute the cells in sequential order, as some cells may depend on variables or functions defined in previous cells.

4. As you execute each cell, observe the output and any instructions provided in comments or Markdown cells within the notebook.

5. If any packages need to be installed, Colab will prompt you to install them when you run the corresponding cell. You only need libspn-keras and TensorFlow.

6. Once all cells have been executed, you should have the SPN model set up and ready for training and evaluation.

7. You can then proceed with training the model on the provided dataset and evaluating its performance as described in the notebook.


**Dataset used:** mnist_*.csv is a small sample of the MNIST database, which is described at: [MNIST Dataset Description](http://yann.lecun.com/exdb/mnist/)

## Implementation Details

- `!pip install libspn-keras tensorflow-datasets`: This line installs the required Python packages, namely libspn-keras and tensorflow-datasets, using pip.

- `import libspn_keras as spnk`: This imports the libspn-keras library and aliases it as spnk for easier reference.

- `print(spnk.get_default_sum_op())`: This line prints the default sum operation used in the library.

- `spnk.set_default_sum_op(spnk.SumOpGradBackprop())`: It sets the default sum operation to use gradient backpropagation for backpropagating gradients during training.

- `from tensorflow import keras`: This imports the Keras module from TensorFlow.

- `spnk.set_default_accumulator_initializer(...)`: This sets the default initializer for accumulator nodes in the SPN. Accumulator nodes are used for summing up values in the SPN.

- `import tensorflow_datasets as tfds`: This imports the TensorFlow Datasets module, which provides convenient access to various datasets.

- `batch_size = 32`: Sets the batch size for training.

- `normalize = spnk.layers.NormalizeStandardScore(...)`: Defines a normalization layer to normalize input data to have zero mean and unit variance.

- `def take_first(a, b): ...`: Defines a function take_first to be used for preprocessing data.

- `normalize.adapt(...): Adapts the normalization layer to the training data.

- `sum_product_network = keras.Sequential([...])`: Defines the architecture of the SPN model using a Sequential API from Keras.

- `sum_product_network.summary()`: Prints a summary of the model architecture.

- `mnist_train = tfds.load(name="mnist", split="train", as_supervised=True)...`: Loads the MNIST training dataset.

- `mnist_test = tfds.load(name="mnist", split="test", as_supervised=True)...`: Loads the MNIST test dataset.

- `optimizer = keras.optimizers.Adam(learning_rate=1e-2)`: Defines the Adam optimizer with a specified learning rate.

- `metrics = [keras.metrics.SparseCategoricalAccuracy()]`: Defines the evaluation metric for the model.

- `loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)`: Defines the loss function for the model.

- `sum_product_network.compile(...)`: Compiles the model with the specified optimizer, loss function, and metrics.

- `sum_product_network.fit(...)`: Trains the model on the MNIST training dataset.

- `sum_product_network.evaluate(...)`: Evaluates the trained model on the MNIST test dataset.

## Overview

Overall, this code sets up a SPN model for digit classification on the MNIST dataset, compiles it, trains it, and evaluates its performance.

