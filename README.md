# Dota 2 Draft Strength Prediction With Keras

The goal of this project is to decide which draft is better in a Dota 2 match based on the heroes picked in the draft. The repository contains datasets that are cleaned and stored as .csv files, it has a pickle file with unstructured data obtained from VPGame Inc., jupyter notebook contains detailed information for loading the data and running the model.
For more information on the purpose and incentive of this project please check this blog - https://medium.com/@shrinivasshetty21/deep-learning-with-keras-on-dota-2-statistics-9a937434f9e0.

## Getting Started

Use the following instructions to make your system ready to run the code.

### Dependencies

Project is run using:
- Windows 10
- Python 3.5
- TensorFlow 1.11.0
- Keras 2.2.4

### Installations

A requirements.txt is added to the repository which can be used to install the dependencies using the following code.

```
$ conda create --name <env> --file <this file>
```

Example

```
$ conda create --name Dota2WR --file requirements.txt
```

## Inside the Repo

The repository contains jupyter notebooks that are used to structure the data obtained from Dota_2.vp pickle file. 
The .csv files contain the clean structured data that is used by Jupyter Notebook to train the model and perform predictions. 
The repository also contains a hero lookup table to decode Hero ID for practical implementations if required.
Description and purpose for each file is given below.
All the files required for testing the model is in Testing Folder.
  
### Files
- Hero Lookup Table : Contains information of Hero Name analogous to Hero ID.

### Folders
##### Data Processing Notebooks
- Data Preparation Dota 2 Neural Network.ipynb : Used for structuring data as per Machine Learning Model requirement. 
- Data Preparation Dota 2.ipynb : Used for structuring data obtained from Dota_2.vp pickle file.
- Dota_2.vp : Data provided by VPGame, Inc. for pursuing similar project, now made available to public.
- Dota_2_Cleaned_Data.csv : Contains a structured form of data obtained from pickle file.
- Dota_2_Model_Data.csv : Contains a structured form of data useful for the model.

##### Deep Learning Model Notebooks
- Predicting Dota 2 Draft Strength Using Keras.ipynb : Jupyter Notebook implementing the project.
- Team Input Model.ipynb : A Jupyter Notebook for tweaking Team Model as per requirement without using any model functions like ffnn_model.py.
- Training & Testing Model.ipynb : A Jupyter Notebook for training and testing the model. Make sure all the files are in the same directory before running this notebook.
- Hero Input Model.ipynb : A Jupyter Notebook for tweaking Hero Model as per requirement without using any model functions like ffnn_model.py.

##### Training Scripts
- ffnn_model.py : Feed Forward Neural Network Model function that returns the defined model.
- lstm_model.py : LSTM Model function that returns the defined model.
- lstm_ffnn_model.py : Hybrid Model function that returns the defined model.
- training_model.py : A python function to train the model.

##### Image
- Contains .png files for results of different models.

##### Testing Script
- Contains .h5 files for training the model.
- testing_model.py : A python function to test the model.

## Training the Model

I have already trained all the models and stored the weights for different model in the repository in Testing Folder.
But you can call the function and train the models if you like using the following code.

```
# First Input to the function = Model Name : 'FFNN', 'LSTM', 'LSTM_FFNN'
# Second Input to the function = Model Type : 'Hero', 'Team'
# Example:
training_model('FFNN', 'Hero')
```

## Testing the Model

The Testing Folder contains the testing_model along with the weights for each model.
You can call the function using following lines of code.

```
# First Input to the function = Model Name : 'FFNN', 'LSTM', 'LSTM_FFNN'
# Second Input to the function = Model Type : 'Hero', 'Team'
# Third Input Your Team Hero ID's = List containing Hero ID's : [8, 36, 119, 27, 75]
# Fourth Input Enemy Team Hero ID's = List containing Hero ID's : [17, 20, 71, 47, 109]
# Example:
testing_model('FFNN', 'Hero', [8, 36, 119, 27, 75], [17, 20, 71, 47, 109])
```
Output:
```
My Team Strength: 67%
Enemy Team Strength: 33%
```
### Additional Notes

This project was an inspiration from the AlphaMao project at VPGame, Inc.
If you are interested you can check their code at https://github.com/vpus/dota2-win-rate-prediction-v1.
If you have anything you'd like to discuss about the project. Feel free to contact me at shrinivasshetty21@gmail.com.
