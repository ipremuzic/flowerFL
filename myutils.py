import os

import numpy as np

from leaf_utils import read_data


def get_dataset(dataset, use_val_set=False):
    """
    Get generated LEAF dataset optimized for federal learning.
    """
    eval_set = 'test' if not use_val_set else 'val'           #REMOVE
    
    # local path to train and test data dir
    train_data_dir = os.path.join('..', 'quickstart_tensorflow','data', dataset, 'data', 'train') 
    test_data_dir = os.path.join('..', 'quickstart_tensorflow', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    return users, groups, train_data, test_data


def get_user_at_index(index, users):
    """
    Ger user_id at index.
    """
    return users[index] 


def get_data_for_client(user_id, users, groups, train_data, test_data):
    """
    Get train and test data for particular user/client.
    """
    for u in users:
        if u == user_id:
            u_train_data, u_test_data = train_data[u], test_data[u]
            break
    
    raw_train_x = u_train_data['x']
    raw_train_y = u_train_data['y']
    raw_test_x = u_test_data['x']
    raw_test_y = u_test_data['y']
    
    x_train = np.array(raw_train_x)
    y_train = np.array(raw_train_y)
    x_test = np.array(raw_test_x)
    y_test = np.array(raw_test_y)

    return (x_train, y_train), (x_test, y_test)