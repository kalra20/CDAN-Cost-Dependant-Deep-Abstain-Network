import json
import numpy as np
import os
from glob import glob
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def my_generator(func, x_train, y_train,batch_size,k):
    while True:
        res = func(x_train, y_train, batch_size
                   ).next()
        yield [[res[0]], [res[1],res[1]]]


