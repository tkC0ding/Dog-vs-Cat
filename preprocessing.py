import os
import cv2
import numpy as np
import random

DATA_PATH = 'Data/PetImages'
CATEGORIES = os.listdir(DATA_PATH)
PREPROCESS_DATA_PATH = 'Data/Preprocessed'
IMG_SIZE = 50

training_data = []
def preprocess_data():
    for category in CATEGORIES:
        CLASS_INDEX = CATEGORIES.index(category)
        print('category :', category)
        for item in os.listdir(os.path.join(DATA_PATH, category)):
            try:
                img = cv2.imread(os.path.join(DATA_PATH, category, item), cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_resized, CLASS_INDEX])
            except Exception as e:
                pass
        print('preproess complete for category :', category)

preprocess_data()

random.shuffle(training_data)

X = []
Y = []

for item in training_data:
    X.append(item[0])
    Y.append(item[1])

features = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
labels = np.array(Y)

os.mkdir(PREPROCESS_DATA_PATH)

np.save(f'{PREPROCESS_DATA_PATH}/features', features)
np.save(f'{PREPROCESS_DATA_PATH}/labels', labels)

print(f'data has been saved as features and labels in directory\n{PREPROCESS_DATA_PATH}')