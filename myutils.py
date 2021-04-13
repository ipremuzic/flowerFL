import os

import numpy as np

from leaf_utils import read_data


def get_dataset(dataset, use_val_set=False):
    """
    Get generated LEAF dataset optimized for federal learning.
    """
    # local path to train and test data dir
    train_data_dir = os.path.join('.','data', dataset, 'data', 'train') 
    test_data_dir = os.path.join('.', 'data', dataset, 'data', 'test')

    clients, _ , train_data, test_data = read_data(train_data_dir, test_data_dir)

    return clients, train_data, test_data


def get_client_at_index(index, clients):
    """
    Ger client_id for client at index.
    """
    return clients[index] 


def get_data_for_client(cid):
    """
    Get train and test data for particular user/client.
    """
    clients, train_data, test_data = get_dataset("femnist")
    c = get_client_at_index(cid, clients)
    c_train_data, c_test_data = train_data[c], test_data[c]

    return c_train_data, c_test_data