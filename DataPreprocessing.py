import numpy as np
import os
from PIL import Image
from random import shuffle
from sklearn.model_selection import train_test_split

Data_path = 'Data/PetImages'
Save_path = 'Data/Preprocessed'
Label_list = os.listdir(Data_path)

def one_hot_encode(inp:np.array):
    final_data = []
    a = inp.max()
    for i in inp:
        l = list(np.zeros(a))
        l[i] = 1
        final_data.append(l)
    return(np.array(final_data))

def run():
    training_data = []

    for label in Label_list:
        l = Label_list.index(label)
        data_path = os.path.join(Data_path, label)
        for item in os.listdir(data_path):
            img_path = os.path.join(data_path, item)
            img = np.array(Image.open(img_path))
            img = img/img.max()
            training_data.append((img, l))

    shuffle(training_data)
    features = []
    labels = []

    for record in training_data:
        features.append(record[0])
        labels.append(record[1])

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    filenames = ['x_train', 'x_test', 'y_train', 'y_test']
    data = [x_train, x_test, y_train, y_test]

    for i,j in zip(filenames, data):
        np.save(f'{Save_path}/{i}', j)

run()