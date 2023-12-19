import csv
import numpy as np
from sklearn.metrics import accuracy_score


def read_csv_to_array(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)

    # Convert the data list into a NumPy array
    array_data = np.array(data, dtype=float)

    return array_data

def svm_classify(Xs, Ys, Xt, Yt, norm=False):
    from sklearn.svm import SVC
    model = SVC()
    Ys = Ys.ravel()
    Yt = Yt.ravel()
    if norm:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        Xt = scaler.fit_transform(Xt)
    model.fit(Xs, Ys)
    Yt_pred = model.predict(Xt)
    acc = accuracy_score(Yt, Yt_pred)
    # print(f'Accuracy using SVM: {acc * 100:.2f}%')
    return round(acc, 4)

def load_data(source, target):
    source_path = f"./data/{source}_{source}.csv"
    source_data = read_csv_to_array(source_path)
    target_path = f"./data/{source}_{target}.csv"
    target_data = read_csv_to_array(target_path)
    return source_data[:, :-1], source_data[:, -1], target_data[:, :-1], target_data[:, -1]

if __name__ == "__main__":
    a2a_path = "./data/Art_Art.csv"
    a2r_path = "./data/Art_RealWorld.csv"
    c2c_path = "./data/Clipart_Clipart.csv"
    c2p_path = "./data/Clipart_RealWorld.csv"

    a2a_data = read_csv_to_array(a2a_path)
    a2r_data = read_csv_to_array(a2r_path)
    c2c_data = read_csv_to_array(c2c_path)
    c2p_data = read_csv_to_array(c2p_path)