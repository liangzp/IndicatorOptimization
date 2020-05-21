# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:34:22 2020

@author: user1
"""

import pickle


def load_obj(name):  # 读取pickle文件
    if '.pkl' not in name:
        name+='.pkl'
    with open( name , 'rb') as f:
        return pickle.load(f)

f = '../data/fund_data.pkl'
d = load_obj(f)


