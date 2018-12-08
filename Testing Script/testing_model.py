""" This module prints the draft strength for the hero lineup. """
from tensorflow.python.keras.models import load_model
import numpy as np
def testing_model(model_name, model_type, my_team, enemy_team):
    """ This module prints the draft strength for both teams. """
    if type(my_team) != list and len(my_team) != 5:
        print("Pass input for your team as a list of strings with 5 Hero ID's")
        return
    if type(enemy_team) != list and len(enemy_team) != 5:
        print("Pass input for enemy team as a list of strings with 5 Hero ID's")
        return
    if model_type == 'Hero':
        model_t = load_model(model_name + model_type + '.h5')
        my_team_hero_a = np.asarray([[my_team[0]]])
        my_team_hero_b = np.asarray([[my_team[1]]])
        my_team_hero_c = np.asarray([[my_team[2]]])
        my_team_hero_d = np.asarray([[my_team[3]]])
        my_team_hero_e = np.asarray([[my_team[4]]])
        enemy_team_hero_a = np.asarray([[enemy_team[0]]])
        enemy_team_hero_b = np.asarray([[enemy_team[1]]])
        enemy_team_hero_c = np.asarray([[enemy_team[2]]])
        enemy_team_hero_d = np.asarray([[enemy_team[3]]])
        enemy_team_hero_e = np.asarray([[enemy_team[4]]])
        draft_strength = model_t.predict([my_team_hero_a, my_team_hero_b, my_team_hero_c, my_team_hero_d, my_team_hero_e, enemy_team_hero_a, enemy_team_hero_b, enemy_team_hero_c, enemy_team_hero_d, enemy_team_hero_e], batch_size=None, verbose=0, steps=None)
        print('My Team Strength: '+ str(draft_strength[0][0]*100) + '%')
        print('Enemy Team Strength: '+ str(100 - draft_strength[0][0]*100) + '%')
    if model_type == 'Team':
        model_t = load_model(model_name + model_type + '.h5')
        my_team_a = np.asarray([my_team])
        enemy_team_b = np.asarray([enemy_team])
        draft_strength = model_t.predict([my_team_a, enemy_team_b], batch_size=None, verbose=0, steps=None)
        print('My Team Strength: '+ str(draft_strength[0][0]*100) + '%')
        print('Enemy Team Strength: '+ str(100 - draft_strength[0][0]*100) + '%')