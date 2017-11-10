#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import pandas as pd

def loadMovieList():
    f = open('movie_ids.txt').readlines()
    n = 1682
    list = [0]*1682
    i = 0
    for line in f:
        if (i+1)/10 < 1:
            list[i] = line[2:]
        elif (i+1)/100 < 1:
            list[i] = line[3:]
        elif (i+1)/1000 < 1:
            list[i] = line[4:]
        else:
            list[i] = line[5:]
        i += 1
    # dict = {"list":list}
    # movieList = pd.DataFrame(dict)
    # return movieList
    return list