import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import math

combats = pd.read_csv('./data/combats.csv')
pokemon =  pd.read_csv('./data/pokemon.csv')
# tests = pd.read_csv('./data/tests.csv')
matchups = pd.read_csv('./data/matchups.csv')
# dataset = pd.read_csv('./data/data.csv')

# combats_matrix = np.array(combats.values, 'int')
# pokemon_matrix = np.array(pokemon.values)
# tests_matrix = np.array(tests.values, 'int')
# matchup_matrix = np.array(matchups.values)

# print(list(combats.columns))
# print(combats_matrix)
# print(list(pokemon.columns))
# print(pokemon_matrix[265])
# print(pokemon_matrix[297])
# print(tests_matrix)
# print(matchup_matrix)

# double super effective matchup is 2.5

def write_dataset(combats, pokemon, matchup, filepath):
    combats = np.array(combats.values, 'int')    
    pokemon = np.array(pokemon.values)

    combats -= 1
    for i in range(pokemon.shape[0]):
        for j in range(2,4):
            if type(pokemon[i,j])== float : pokemon[i,j] = 'skip'
            pokemon[i,j] = pokemon[i,j].lower()

    stat_diff = [[pokemon[i[0],j] - pokemon[i[1],j] for j in range(4,10)] for i in combats]

    matchup_vals = [[float(matchup[k][matchup['attacker']==j]) for j in pokemon[i[0],2:4] for k in pokemon[i[1],2:4]] for i in combats]
    win = [1 if i[2]==i[0] else 0 for i in combats]

    cols = ['hp','atk','def','spatk','spdef','spd','t11','t12','t21','t22','win']
    # cols = ['hp','atk','def','spatk','spdef','spd','win']
    d = np.c_[stat_diff, matchup_vals, win]
    # d = np.c_[stat_diff, win]
    data = pd.DataFrame(data = d, columns=cols)
    data.to_csv(filepath, index=False)

    return None

def normalize(data):
    data = data/abs(data.max(axis=0))
    # print(data)
    # print(target)
    return data


def load_data():
    dataset = pd.read_csv('./data/data.csv')
    
    if type(dataset) == pd.DataFrame:
        dataset = np.array(dataset.values, 'half')
    
    data = dataset[:,:-1]
    target = dataset[:,-1]

    data_normalized = normalize(data)

    #adding ones to data (Beta_0)
    one = np.ones((len(data_normalized),1))
    data_normalized = np.append(one, data_normalized, axis=1)
    # print(data_normalized)
    # print(target)
    return data, data_normalized, target

if __name__ == "__main__":
    print(">> writing dataset")
    write_dataset(combats, pokemon, matchups,'data/data.csv')
# normalize(dataset)