import pandas as pd
import numpy as np
import sys

import random
import math

def write_dataset(combats, pokemon, matchup, filepath):
    # Get data
    combats = np.array(combats.values, 'int')    
    pokemon = np.array(pokemon.values)

    # Makes pokemon dataset index-0
    combats -= 1

    # Handles type matchups of the pokemons
    for i in range(pokemon.shape[0]):
        for j in range(2,4):
            if type(pokemon[i,j])== float : pokemon[i,j] = 'skip'
            pokemon[i,j] = pokemon[i,j].lower()

    matchup_vals = [[float(matchup[k][matchup['attacker']==j]) for j in pokemon[i[0],2:4] for k in pokemon[i[1],2:4]] for i in combats]

    # Creates and array with the differences of the stats of the two pokemons fighting
    stat_diff = [[pokemon[i[0],j] - pokemon[i[1],j] for j in range(4,10)] for i in combats]

    # Extracts the winning pokemon (0 for 1st pokemon ID, 1 for 2nd pokemon ID)
    win = [1 if i[2]==i[0] else 0 for i in combats]

    # Set column names
    cols = ['hp','atk','def','spatk','spdef','spd','t11','t12','t21','t22','win']

    # Concatenate relevant arrays
    d = np.c_[stat_diff, matchup_vals, win]

    # Convert to dataframe then write to csv
    data = pd.DataFrame(data = d, columns=cols)
    data.to_csv(filepath, index=False)

    # UwU
    print("Done writing UWu")

    return None

def normalize(data):
    # Normalize each column
    data = (data - data.min(axis=0))/(data.max(axis=0) - data.min(axis=0))
    return data

def load_data(filename = './data/data.csv'):
    # Get dataset
    dataset = pd.read_csv(filename)
    
    # Converting to numpy array
    data_cols = list(dataset.columns)
    dataset = np.array(dataset.values, 'half')

    # Normalize dataset
    data_normalized = normalize(dataset)

    # Adding ones to data (Beta_0)
    one = np.ones((len(data_normalized),1))
    data_normalized = np.append(one, data_normalized, axis=1)

    # Shuffles the dataset
    np.random.shuffle(data_normalized)
    
    # Split the dataset with training having 80% and testing set having 20%
    train_normalized = data_normalized[int(0.8*len(data_normalized)):]
    test_normalized = data_normalized[:int(0.8*len(data_normalized))]

    # Extract the target data from both sets
    train_data = train_normalized[:,:-1]
    train_target = train_normalized[:,-1]
    test_data = test_normalized[:,:-1]
    test_target = test_normalized[:,-1]

    # Return the relevant arrays
    return train_data, train_target, test_data, test_target, data_cols[:-1]

if __name__ == "__main__":
    # Get raw data
    print(">> loading raw data...")
    combats = pd.read_csv('./data/combats.csv')
    pokemon =  pd.read_csv('./data/pokemon.csv')
    matchups = pd.read_csv('./data/matchups.csv')

    # For debugging
    if 'pcols' in sys.argv:
        print(list(combats.columns))
        print(list(pokemon.columns))
        print(list(matchups.columns))

    # Setting where to put processed data
    filepath = 'data/data.csv'
    if 'f' in sys.argv:
        try:
            filepath = 'data/' + sys.argv[sys.argv.index('f')+1] + '.csv'
        except:
            print('>> [!] file error')
    
    # Write the data in our preferred format
    print(">> writing dataset...")
    write_dataset(combats, pokemon, matchups, filepath)
