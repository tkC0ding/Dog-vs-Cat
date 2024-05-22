import os
import shutil

Data_path = 'Data/PetImages'
Train_path = 'Data/train'
Test_path = 'Data/test'

Labels_list = os.listdir(Data_path)

test_size = 2490

for label in Labels_list:
    p = os.path.join(Data_path, label)
    counter = 0
    for item in os.listdir(p):
        shutil.move(os.path.join(p, item), os.path.join(Test_path, label, item))
        counter += 1
        if(counter == test_size-1):
            break

for label in Labels_list:
    p = os.path.join(Data_path, label)
    for item in os.listdir(p):
        shutil.move(os.path.join(p, item), os.path.join(Train_path, label, item))

os.rmdir(Data_path)