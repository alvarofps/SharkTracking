import glob, os
from sklearn.model_selection import StratifiedKFold as skfold
import numpy as np

dataset_path = '/home/apena/Desktop/backup_Dataset/full_dataset'
#dataset_path = '/home/alvaro/PycharmProjects/Split/data'


full_data = []
labels = []
fold = skfold(3, True, 1)
folds = {}

def populate_data_labels():
    global full_data
    global labels

    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        image_path = dataset_path + "/" + title + '.png'
        full_data.append(image_path)
        if title.__contains__('negative'):
            labels.append(0)
        else:
            labels.append(1)


def split():
    global fold
    global folds
    full_data_out = np.array(full_data)
    labels_out = np.array(labels)
    counter = 0

    for train, test in fold.split(full_data_out, labels_out):
        folds[counter] = full_data_out[train], full_data_out[test]
        counter = counter + 1


def write_txt_files():
    global folds
    for i in range(len(folds)):
        file_train = open('train_'+str(i)+'.txt', 'w')
        file_test = open('test_'+str(i)+'.txt', 'w')

        train_files, test_files = folds[i]

        for file in train_files:
            file_train.write(str(file) + "\n")
        file_train.close()

        for file in test_files:
            file_test.write(str(file) + "\n")
        file_test.close()


populate_data_labels()
split()
write_txt_files()


