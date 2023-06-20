# Binary-classification-model-using-a-deep-neural-network.
## Application :
You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

The business team has given you a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.

## Technologies 
In this challenge, you’ll use Jupyter Lab and the following python libraries:
- Pandas
- scikit-learn
     - [scikit metrics] (https://scikit-learn.org/stable/modules/model_evaluation.html)
     - imbalanced-learn
     - linear model
     - train test split
     - Standard Scaler OneHotEncoder
 
## Installation Guide
 ## To check that scikit-learn and hvPlot are installed in your Conda dev environment, complete the following steps:
 ## 1. Activate your Conda dev environment (if it isn’t already) by running the following in your terminal:

 `conda activate dev`
## 2. When the environment is active, run the following in your terminal to check if the scikit-learn, itensorflow and keras libraries are installed on your machine:

`conda list scikit-learn
python -c "import tensorflow as tf;print(tf.__version__)"
python -c "import tensorflow as tf;print(tf.keras.__version__)"`

## If you see scikit-learn, itensorflow and keras listed in the terminal, you’re all set!
### 1. Install scikit-learn
`pip install -U scikit-learn`

### 2. Install tensorflow
`pip install --upgrade tensorflow`

## Usage
## To use this application, simply clone the repository and open jupyter lab from git bash by running the following command:
`jupyter lab`

## Instructions:
The steps for this challenge are broken out into the following sections:

- Prepare the data for use on a neural network model.
- Compile and evaluate a binary classification model using a neural network.
- Optimize the neural network model.

## PreProcess the dataset :
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), preprocess the dataset so that you can use it to compile and evaluate the neural network model later.
Open the starter code file, and complete the following data preparation steps:
- 1) Read the `applicants_data.csv` file into a Pandas DataFrame.
- 2) Drop the irrelevant columns from the dataframe.
- 3) Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.
- 4) Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables using `concat()` function.
- 5) Using the preprocessed data, create the features (X) and target (y) datasets.
- 6) Split the features and target sets into training and testing datasets.
- 7) Use scikit-learn's StandardScaler to scale the features data.

## Compile and Evaluate a Binary Classification Model Using a Neural Network
To do so, complete the following steps:

- Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
- Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
- Evaluate the model using the test data to determine the model’s loss and accuracy.
- Save and export your model to an HDF5 file, and name the file AlphabetSoup.h5.

## Optimize the Neural Network Model 
To do so, complete the following steps:
- Define at least two new deep neural network models (resulting in the original model, plus two optimization attempts). With each, try to improve on your first model’s predictive accuracy.
- After finishing your models, display the accuracy scores achieved by each model, and compare the results.
- Save each of your models as an HDF5 file.


## Contributors
Brought to you by Malika Ajmera


    

