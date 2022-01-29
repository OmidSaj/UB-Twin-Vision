import matplotlib.pyplot as plt


# helper function for data visualization
def visualize_plt(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


"""
Created on Fri Sep 11 10:12:41 2020

@author: OMID
"""

import os
import shutil
import pickle
import numpy as np


def make_di_path(path_name):
    try:
        os.mkdir(path_name)
    except OSError:
        print("Creation of the directory %s failed" % path_name)
    else:
        print("Successfully created the directory %s " % path_name)


def wipe_dir(Model_dir):
    folder = Model_dir
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def delete_file(filePath):
    try:
        os.remove(filePath)
    except OSError:
        1 == 1  # Do nothing


def delete_folder(filePath):
    try:
        shutil.rmtree(filePath)
    except OSError as e:  ## if failed, report it back to the user ##
        print("This folder does not exist")


def save_pickle(data, filename):
    output = open(filename + '.pkl', 'wb')
    pickle.dump(data, output)
    output.close()


def load_pickle(filename):
    pkl_file = open(filename + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    return data


import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)