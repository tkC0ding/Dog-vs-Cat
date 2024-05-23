import os
import shutil

Data_path = 'Data/PetImages'
Train_path = 'Data/train'
valid_path = 'Data/validation'
Test_path = 'Data/test'

os.mkdir(Train_path)
os.mkdir(Test_path)
os.mkdir(valid_path)

Labels_list = os.listdir(Data_path)

for label in Labels_list:
    os.mkdir(os.path.join(Train_path, label))
    os.mkdir(os.path.join(Test_path, label))
    os.mkdir(os.path.join(valid_path, label))

test_size = 2490
validation_size = 3000

for label in Labels_list:
    p = os.path.join(Data_path, label)
    counter = 0
    for item in os.listdir(p):
        shutil.move(os.path.join(p, item), os.path.join(Test_path, label, item))
        counter += 1
        if(counter == test_size):
            break

for label in Labels_list:
    p = os.path.join(Data_path, label)
    for item in os.listdir(p):
        shutil.move(os.path.join(p, item), os.path.join(Train_path, label, item))

for label in Labels_list:
    p = os.path.join(Train_path, label)
    counter = 0
    for item in os.listdir(p):
        shutil.move(os.path.join(p, item), os.path.join(valid_path, label, item))
        counter += 1
        if(counter == validation_size):
            break

shutil.rmtree(Data_path, ignore_errors=True)