#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import matplotlib.pyplot as plt

# 现在显示的图片是一个100*400的，其中每一行都是一张20*20的图片，我们希望一百张能够四四方方地显示出来。
def displayData(x, example_width):
    # 把图片的宽度和高度搞出来
    if example_width not in dir():
        example_width = int(round(np.sqrt(x.shape[1])))
    [m,n] = np.shape(x)
    example_height = int(n/example_width)

    # 计算显示图片数量的行数和列数
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m/display_rows))

    # 下面开始将数组增加纬度，从这里开始和octave中不同，这也是第一次觉得numpy比octave好用的地方
    blank_list = [0]*m
    for i in range(0,m):
        blank_list[i] = x[i].reshape(example_height,example_width).T # 注意这里需要转置一下，因为数据是为了octave设计的，而octave中的reshap是优先填充列的

    # 现在已经创建了一个列表，列表中的每一个对象都是一张图片，已经转置好了，我们下面开始融合他们，先一行一行地来，然后把所有行融合在一起
    row_list = [0]*display_rows

    for r in range(0,display_cols):
        row_list[r] = blank_list[r*display_cols]

        for c in range(1,display_cols):
            row_list[r] = np.c_[(row_list[r],blank_list[r*display_cols+c])]
            #row_list[r] = np.c_[(row_list[r],-np.ones([example_height]).reshape(example_height,1))]
        if r ==0:
            display_array = row_list[0]
        if r >=1:
            display_array = np.r_[(display_array,row_list[r])]
            #display_array = np.r_[(display_array,-np.ones([example_width*display_cols]).reshape(example_width*display_cols,1))]

    # 开始画图了
    f1 = plt.figure(1)
    plt.imshow(display_array,cmap=plt.cm.gray)
    plt.show()


