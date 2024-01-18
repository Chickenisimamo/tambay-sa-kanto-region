import pandas as pd
import numpy as np
import sys

import random
import math


# tests = pd.read_csv('./data/tests.csv')

# combats_matrix = np.array(combats.values, 'int')
# pokemon_matrix = np.array(pokemon.values)
# tests_matrix = np.array(tests.values, 'int')
# matchup_matrix = np.array(matchups.values)

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
    d = np.c_[stat_diff, matchup_vals, win]
    data = pd.DataFrame(data = d, columns=cols)
    data.to_csv(filepath, index=False)

    return None

def normalize(data):
    data = data/abs(data.max(axis=0))
    return data

def load_data():
    dataset = pd.read_csv('./data/data.csv')
    
    data_cols = list(dataset.columns)
    dataset = np.array(dataset.values, 'half')
    
    data = dataset[:,:-1]
    target = dataset[:,-1]

    data_normalized = normalize(data)

    #adding ones to data (Beta_0)
    one = np.ones((len(data_normalized),1))
    data_normalized = np.append(one, data_normalized, axis=1)
    return data, data_normalized, target, data_cols[:-1]

if __name__ == "__main__":
    print(">> loading raw data...")
    combats = pd.read_csv('./data/combats.csv')
    pokemon =  pd.read_csv('./data/pokemon.csv')
    matchups = pd.read_csv('./data/matchups.csv')

    if 'pcols' in sys.argv:
        print(list(combats.columns))
        print(list(pokemon.columns))
        print(list(matchup.columns))


    filepath = 'data/data.csv'
    if 'f' in sys.argv:
        try:
            filepath = 'data/' + sys.argv[sys.argv.index('f')+1] + '.csv'
        except:
            print('>> [!] file error')
    
    print(">> writing dataset...")
    write_dataset(combats, pokemon, matchups, filepath)