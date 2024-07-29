# 3rd-Year-Project-
Part of the 3rd year project at UCT, I had to create a machine learning algorithm to classify signal and background processes.

## Dataset

There's two files named 'tWZ+ttZ (1).csv' and 'ttZ+bkg.csv' in the repository, use the pandas.read_csv() function to read them to pandas dataframes (pandas.DataFrame()). You can read more in the documentation at https://pandas.pydata.org/.

## Selection Criteria

There's criteria that can be applied to selected the input features that may be of use or have a more meaningful physical interpretation (i.e choosing only kinematic variables), and certain features have been dropped to avoid trianing statistical noise instead or meaningful physical quantities.

## Spliting Training, Testing and Validation Datasets

Use the sklearn.model_selection.train_test_split() twice to split the dataset into 3, 60% being the training dataset, 20% being the testing dataset and the other 20% being the validation dataset. Read about more on the train_test_split() function from the sklearn website https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.

## Scaling the Data

You might also want to scale the features of your data to take on values from 0 to 1 by using a Standard scaler from sklearn, use it to fit and transform the data, more can be found in the documentation. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html.

# Model

A lot of frameworks can be used to define the model, i.e. Tensorflow, Keras, PyTorch and so on. The code on the notebook is executed using Keras and Tensorflow as it's backend, more can be found at the respective sites. https://keras.io/ and https://www.tensorflow.org/.

## Keras Model

We use the keras Sequential() model and added 4 layers, 1 input, 2 hidden and 1 output. the Dropout() layer is added to turn off a percentage of the neurons at each layer to add randomness to the model and a percentage is specified, Only 10% = 0.1 was selected for this model. The Dropout() layer is very useful to avoid overfitting.

A kernel initializer wasn't chosen but any can be chosen from the GLOROT-Uniform, Uniform and so on. This is a way to initialize the weights of the before training. A callback is included that utilizes Early_Stopping to stop the model when certain requirements aren't met, but this is a very stringent approach and does not show the full trend of the validation loss and the loss as functions of epochs. As such is was not utilized.

The Adam optimizer was also chosen as the optimizer and the binary_crossentropy was selected as the loss function. The machine learning model uses backpropagation to find parameters fo the weights, biases (multiparameters) that minimize the loss function.

The mean-absolute-error (mae) metric was also included as it is a multidimension equivalent of normalised residues, and if the sum of residues approaches zero overall, the relationship between the hyperparameters, input features and output label is somehow justified.

The model is then trained with the validation set assigned as the validation set.

