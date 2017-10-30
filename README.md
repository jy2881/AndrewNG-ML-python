# AndrewNg-ML-python
Inplement the exercise code of octave into python

用python实现吴恩达在coursera上的machine learning课程

简单说一下我淌过的一些坑吧：

1. octave的所有序列索引都是从1开始的，而python是从0开始的，这个问题导致了我ex3中预测精度仅有86%，我当时就想，这个结果仿佛就是错了一列的样子。

2. 原本练习中用到的fminfun、fmincg这俩梯度下降的优化函数，在python中对应的是scipy.optimize库中的fmin、fmin_cg、minimize，还有很多优化算法，但是从结果上没觉得有什么区别，我这点数据也看不出效率问题。

3. octave中的标准差，开方里面是总和除以数量减一，numpy的里面是均值，所以改写的时候，记得设置axis=0，ddof=1.