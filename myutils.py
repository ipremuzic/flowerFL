import os

import numpy as np


from model_utils import read_data

"""
Get generated LEAF dataset optimized for federal learning. 
"""
def get_dataset(dataset, use_val_set=False):
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'quickstart_tensorflow','data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'quickstart_tensorflow', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    #print("ISPIS: users {}, train_data {}, test_data {}".format(users, train_data, test_data))

    return users, groups, train_data, test_data


def get_user_at_index(index, users):
    if(index == 0):
        print("ISPIS: Index {} user {}".format(index, users[index]))
    return users[index] 


#TODO: Remove groups, dont need it.
"""
Get train and test data for particular user.
"""
def get_data_for_client(user_id, users, groups, train_data, test_data):
    if len(groups) == 0:
        print("ISPIS: groups == 0")
    for u in users:
        if u == user_id:
            u_train_data, u_test_data = train_data[u], test_data[u]
            print("ISPIS: nasao sam korisnika {}".format(user_id))
            break
    
    raw_train_x = u_train_data['x']

    print("ISPIS: raw_train_x data type = {}".format(type(raw_train_x)))
    
    raw_train_y = u_train_data['y']
    raw_test_x = u_test_data['x']
    raw_test_y = u_test_data['y']
    
    # this might be dataset and model specific?
    #TODO: razmislit o ovome
    x_train = np.array(raw_train_x)
    y_train = np.array(raw_train_y)
    x_test = np.array(raw_test_x)
    y_test = np.array(raw_test_y)

    print("ISPIS: x_train data type = {}".format(type(x_train)))

    return (x_train, y_train), (x_test, y_test)
    
