import csv
import numpy as np

def read_csv_to_array(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)

    # Convert the data list into a NumPy array
    array_data = np.array(data, dtype=float)

    return array_data

def load_a2a():
    a2a_path = "./data/Art_Art.csv"
    a2a_data = read_csv_to_array(a2a_path)
    return a2a_data

def load_a2r():
    a2r_path = "./data/Art_RealWorld.csv"
    a2r_data = read_csv_to_array(a2r_path)
    return a2r_data

def load_c2c():
    c2c_path = "./data/Clipart_Clipart.csv"
    c2c_data = read_csv_to_array(c2c_path)
    return c2c_data

def load_c2p():
    c2p_path = "./data/Clipart_RealWorld.csv"
    c2p_data = read_csv_to_array(c2p_path)
    return c2p_data

if __name__ == "__main__":
    a2a_path = "./data/Art_Art.csv"
    a2r_path = "./data/Art_RealWorld.csv"
    c2c_path = "./data/Clipart_Clipart.csv"
    c2p_path = "./data/Clipart_RealWorld.csv"

    a2a_data = read_csv_to_array(a2a_path)
    a2r_data = read_csv_to_array(a2r_path)
    c2c_data = read_csv_to_array(c2c_path)
    c2p_data = read_csv_to_array(c2p_path)