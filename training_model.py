"""" This module is used to train and store model. """
import pandas as pd
import numpy as np
from ffnn_model import ffnn_hero_model, ffnn_team_model
from lstm_model import lstm_hero_model, lstm_team_model
from lstm_ffnn_model import lstm_ffnn_hero_model, lstm_ffnn_team_model

def training_model(model_name, model_type):
    """" Trains the data and returns the model for testing. """
    if model_type == 'Hero' or model_type == 'Team':
        print('Training Begins ...')
    else:
        print('Please Input Model Type as "Team" or "Hero"')
    if model_type == 'Hero':
        dota_2_df = pd.read_csv('Dota_2_Cleaned_Data.csv')
        dota_2_df_label = pd.read_csv('Dota_2_Model_Data.csv')
        label_data = dota_2_df_label.iloc[::2]
        dota_2_df = dota_2_df.head(500000)
        label_data = label_data.head(50000)['Win/Loss']
        label_data = label_data.values
        label_data[label_data >= 1] = 1
        Hero_1 = dota_2_df.iloc[::10, :]['Hero_ID']
        Hero_1 = Hero_1.values
        Hero_2 = dota_2_df.iloc[1::10, :]['Hero_ID']
        Hero_2 = Hero_2.values
        Hero_3 = dota_2_df.iloc[2::10, :]['Hero_ID']
        Hero_3 = Hero_3.values
        Hero_4 = dota_2_df.iloc[3::10, :]['Hero_ID']
        Hero_4 = Hero_4.values
        Hero_5 = dota_2_df.iloc[4::10, :]['Hero_ID']
        Hero_5 = Hero_5.values
        Hero_6 = dota_2_df.iloc[5::10, :]['Hero_ID']
        Hero_6 = Hero_6.values
        Hero_7 = dota_2_df.iloc[6::10, :]['Hero_ID']
        Hero_7 = Hero_7.values
        Hero_8 = dota_2_df.iloc[7::10, :]['Hero_ID']
        Hero_8 = Hero_8.values
        Hero_9 = dota_2_df.iloc[8::10, :]['Hero_ID']
        Hero_9 = Hero_9.values
        Hero_10 = dota_2_df.iloc[9::10, :]['Hero_ID']
        Hero_10 = Hero_10.values
        Permutation = np.random.permutation(Hero_1.shape[0])
        Hero_1 = Hero_1[Permutation]
        Hero_2 = Hero_2[Permutation]
        Hero_3 = Hero_3[Permutation]
        Hero_4 = Hero_4[Permutation]
        Hero_5 = Hero_5[Permutation]
        Hero_6 = Hero_6[Permutation]
        Hero_7 = Hero_7[Permutation]
        Hero_8 = Hero_8[Permutation]
        Hero_9 = Hero_9[Permutation]
        Hero_10 = Hero_10[Permutation]
        label_data = label_data[Permutation]
        Hero_1_Tr = Hero_1[0:40000]
        Hero_2_Tr = Hero_2[0:40000]
        Hero_3_Tr = Hero_3[0:40000]
        Hero_4_Tr = Hero_4[0:40000]
        Hero_5_Tr = Hero_5[0:40000]
        Hero_6_Tr = Hero_6[0:40000]
        Hero_7_Tr = Hero_7[0:40000]
        Hero_8_Tr = Hero_8[0:40000]
        Hero_9_Tr = Hero_9[0:40000]
        Hero_10_Tr = Hero_10[0:40000]
        Train_L = label_data[0:40000]
        Hero_1_Ts = Hero_1[40000:]
        Hero_2_Ts = Hero_2[40000:]
        Hero_3_Ts = Hero_3[40000:]
        Hero_4_Ts = Hero_4[40000:]
        Hero_5_Ts = Hero_5[40000:]
        Hero_6_Ts = Hero_6[40000:]
        Hero_7_Ts = Hero_7[40000:]
        Hero_8_Ts = Hero_8[40000:]
        Hero_9_Ts = Hero_9[40000:]
        Hero_10_Ts = Hero_10[40000:]
        Test_L = label_data[40000:]
    if model_type == 'Team':
        Dota_2_DF = pd.read_csv('Dota_2_Model_Data.csv')
        Dota_2_DF = Dota_2_DF.head(100000)
        Team_1 = Dota_2_DF.iloc[::2]
        Team_2 = Dota_2_DF.iloc[1::2]
        label_data = Team_1[['Win/Loss']]
        label_data = label_data.values
        label_data[label_data >= 1] = 1
        Team_1 = Team_1[['Hero_1', 'Hero_2', 'Hero_3', 'Hero_4', 'Hero_5']]
        Team_2 = Team_2[['Hero_1', 'Hero_2', 'Hero_3', 'Hero_4', 'Hero_5']]
        Team_1 = Team_1.values
        Team_2 = Team_2.values
        Permutation = np.random.permutation(Team_1.shape[0])
        Team_1 = Team_1[Permutation]
        Team_2 = Team_2[Permutation]
        label_data = label_data[Permutation]
        Training_Team_1 = Team_1[:40000]
        Training_Team_2 = Team_2[:40000]
        Train_L_T = label_data[:40000]
        Testing_Team_1 = Team_1[40000:]
        Testing_Team_2 = Team_2[40000:]
        Test_L_T = label_data[40000:]
    if model_name == 'FFNN' and model_type == 'Hero':
        model_t, batch_size = ffnn_hero_model()
        model_t.fit([Hero_1_Tr, Hero_2_Tr, Hero_3_Tr, Hero_4_Tr, Hero_5_Tr, Hero_6_Tr, Hero_7_Tr, Hero_8_Tr, Hero_9_Tr, Hero_10_Tr], Train_L, epochs=5, verbose=1, batch_size=batch_size, validation_data=([Hero_1_Ts, Hero_2_Ts, Hero_3_Ts, Hero_4_Ts, Hero_5_Ts, Hero_6_Ts, Hero_7_Ts, Hero_8_Ts, Hero_9_Ts, Hero_10_Ts], Test_L))
        model_t.save(model_name + model_type + '.h5')
    if model_name == 'LSTM' and model_type == 'Hero':
        model_t, batch_size = lstm_hero_model()
        model_t.fit([Hero_1_Tr, Hero_2_Tr, Hero_3_Tr, Hero_4_Tr, Hero_5_Tr, Hero_6_Tr, Hero_7_Tr, Hero_8_Tr, Hero_9_Tr, Hero_10_Tr], Train_L, epochs=5, verbose=1, batch_size=batch_size, validation_data=([Hero_1_Ts, Hero_2_Ts, Hero_3_Ts, Hero_4_Ts, Hero_5_Ts, Hero_6_Ts, Hero_7_Ts, Hero_8_Ts, Hero_9_Ts, Hero_10_Ts], Test_L))
        model_t.save(model_name + model_type + '.h5')
    if model_name == 'LSTM_FFNN' and model_type == 'Hero':
        model_t, batch_size = lstm_ffnn_hero_model()
        model_t.fit([Hero_1_Tr, Hero_2_Tr, Hero_3_Tr, Hero_4_Tr, Hero_5_Tr, Hero_6_Tr, Hero_7_Tr, Hero_8_Tr, Hero_9_Tr, Hero_10_Tr], Train_L, epochs=5, verbose=1, batch_size=batch_size, validation_data=([Hero_1_Ts, Hero_2_Ts, Hero_3_Ts, Hero_4_Ts, Hero_5_Ts, Hero_6_Ts, Hero_7_Ts, Hero_8_Ts, Hero_9_Ts, Hero_10_Ts], Test_L))
        model_t.save(model_name + model_type + '.h5')
    if model_name == 'FFNN' and model_type == 'Team':
        model_t, batch_size = ffnn_team_model()
        model_t.fit([Training_Team_1, Training_Team_2], Train_L_T, epochs=5, verbose=1, batch_size=batch_size, validation_data=([Testing_Team_1, Testing_Team_2], Test_L_T))
        model_t.save(model_name + model_type + '.h5')
    if model_name == 'LSTM' and model_type == 'Team':
        model_t, batch_size = lstm_team_model()
        model_t.fit([Training_Team_1, Training_Team_2], Train_L_T, epochs=5, verbose=1, batch_size=batch_size, validation_data=([Testing_Team_1, Testing_Team_2], Test_L_T))
        model_t.save(model_name + model_type + '.h5')
    if model_name == 'LSTM_FFNN' and model_type == 'Team':
        model_t, batch_size = lstm_ffnn_team_model()
        model_t.fit([Training_Team_1, Training_Team_2], Train_L_T, epochs=5, verbose=1, batch_size=batch_size, validation_data=([Testing_Team_1, Testing_Team_2], Test_L_T))
        model_t.save(model_name + model_type + '.h5')
