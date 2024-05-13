import pandas as pd
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_closest_character(df, character_name):
    try:
        character_row = df[df['character'] == character_name].iloc[0]
    except IndexError:
        return f"No data available for character '{character_name}'."
    
    min_distance = float('inf')
    closest_character = None

    for index, row in df.iterrows():
        if row['character'] == character_name:
            continue  # Skip the selected character
        distance = calculate_distance(character_row['x'], character_row['y'], row['x'], row['y'])
        if distance < min_distance:
            min_distance = distance
            closest_character = row['character']

    return closest_character

def main():
    df = load_data('character_embeddings.csv')
    character_name = input("Enter the name of the Game of Thrones character: ")
    closest_character = find_closest_character(df, character_name)
    if isinstance(closest_character, str):
        print(f"The character most similar to {character_name} is {closest_character}.")
    else:
        print(closest_character)

if __name__ == '__main__':
    main()
