# Dota 2 Draft Strength Prediction With Keras

The goal of this project is to decide which draft is better in a Dota 2 match based on the heroes picked in the draft. The repository contains datasets that are cleaned and stored as .csv files, it has a pickle file with unstructured data obtained from VPGame Inc., jupyter notebook contains detailed information for loading the data and running the model.
For more information on the purpose and incentive of this project please check this blog - https://medium.com/@shrinivasshetty21/deep-learning-with-keras-on-dota-2-statistics-part-1-9e889ab25e47.

## Getting Started

Use the following instructions to make your system ready to run the code.

### Dependencies

Project is run using:
- Windows 10
- Python 3.5/3.6
- scipy=1.1.0
- scikit-learn=0.19.1
- pandas=0.23.0
- numpy=1.14.3
- anaconda
- tensorflow
- keras
- cycler
- matplotlib
- pickle

### Installing

A requirements.txt is added to the repository which can be used to install the dependencies using the following code.

```
$ conda create --name <env> --file <this file>
```

Example:

```
$ conda create --name Dota2WR --file requirements.txt
```

## Inside the Repo

The repository contains jupyter notebooks that are used to structure the data obtained from Dota_2.vp pickle file. 
The .csv files contain the clean structured data that is used by Jupyter Notebook to train the model and perform predictions. 
The repository also contains a hero lookup table to decode Hero ID for practical implementations if required.
Description and purpose for each file is given below.
  
### Files:

- Data Preparation Dota 2 Neural Network.ipynb : Used for structuring data as per Machine Learning Model requirement. 
- Data Preparation Dota 2.ipynb : Used for structuring data obtained from Dota_2.vp pickle file.
- Dota_2.vp : Data provided by VPGame, Inc. for pursuing similar project, now made available to public.
- LICENSE : License for using the data used in this project.
- Dota_2_Cleaned_Data.csv : Contains a structured form of data obtained from pickle file.
- Dota_2_Model_Data.csv : Contains a structured form of data useful for the model.
- Hero Lookup Table : Contains information of Hero Name analogous to Hero ID.
- Predicting Dota 2 Draft Strength Using Keras.ipynb : Jupyter Notebook implementing the project.
- ffnn_model.py : Feed Forward Neural Network Model function that returns the defined model.
- lstm_model.py : LSTM Model function that returns the defined model.
- lstm_ffnn_model.py : Hybrid Model function that returns the defined model.
- Team Input Model.ipynb : A Jupyter Notebook for tweaking Team Model as per requirement without using any model functions like ffnn_model.py.
- Hero Input Model.ipynb : A Jupyter Notebook for tweaking Hero Model as per requirement without using any model functions like ffnn_model.py.
- Image files containing results of different models.

## Testing Code: 

After training the model you can use the following script to call the model you have trained to return draft strenght in percentage.
Here the model name is Model_T as trained in the Jupyter Notebook.

### Example:

```
# After training your model run the following code.
# Replace with your squad hero id here.
My_Team = np.asarray([[8, 36, 119, 27, 75]])
# Replace with Enemy squad hero id here.
Enemy_Team = np.asarray([[17, 20, 71, 47, 109]])
Draft_Strength = Model_T.predict([My_Team, Enemy_Team], batch_size=None, verbose=0, steps=None)
print('My Team Strength: '+ str(Draft_Strength[0][0]*100) + '%')
print('Enemy Team Strength: '+ str(100 - Draft_Strength[0][0]*100) + '%')
```

### Additional Notes:

This project was an inspiration from the AlphaMao project at VPGame, Inc.
If you are interested you can check their code at https://github.com/vpus/dota2-win-rate-prediction-v1.
If you have anything you'd like to discuss about the project. Feel free to contact me at shrinivasshetty21@gmail.com.
