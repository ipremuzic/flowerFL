import os

import numpy as np

from leaf_utils import read_data


def get_dataset(dataset, use_val_set=False):
    """
    Get generated LEAF dataset optimized for federal learning.
    """
    eval_set = 'test' if not use_val_set else 'val'           #REMOVE
    
    # local path to train and test data dir
    train_data_dir = os.path.join('.','data', dataset, 'data', 'train') 
    test_data_dir = os.path.join('.', 'data', dataset, 'data', eval_set)

    clients, _ , train_data, test_data = read_data(train_data_dir, test_data_dir)

    return clients, train_data, test_data


def get_user_at_index(index, clients):
    """
    Ger user_id from users at index.
    """
    return clients[index] 


def get_data_for_client(client_id, clients, train_data, test_data):
    """
    Get train and test data for particular user/client.
    """
    for c in clients:
        if c == client_id:
            c_train_data, c_test_data = train_data[c], test_data[c]
            break

    return c_train_data, c_test_data