
## 原理

连续投影算法（successive projections algorithm, SPA）是前向特征变量选择方法。

SPA利用向量的投影分析，通过将波长投影到其他波长上，比较投影向量大小，以投影向量最大的波长为待选波长，然后基于矫正模型选择最终的特征波长。
SPA选择的是含有最少冗余信息及最小共线性的变量组合。

该算法简要步骤如下:
![Image text](https://github.com/mepleleo/SPA/blob/main/%E7%AE%97%E6%B3%95%E6%AD%A5%E9%AA%A4.png)
-------------------------摘自《光谱及成像技术在农业中的应用》P130



## 注意事项

1. 光谱矩阵（m * n）== m行为样本，n列为波段 ==

2. 进行建模前需要对光谱进行 建模集测试集分割 与 数据归一化 ，可先进行分割再归一，也可以先归一再分割，下边为分割再归一

   ````python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import MinMaxScaler
   
   Xcal, Xval, ycal, yval = train_test_split(x, y, test_size=0.4, random_state=0)
   
   min_max_scaler = MinMaxScaler(feature_range=(-1, 1))  
   Xcal = min_max_scaler.fit_transform(Xcal)
   Xval = min_max_scaler.transform(Xval)
   ````

   

源链接：https://gitee.com/aBugsLife/SPA
````
（1）变量Xcal，Ycal指的是（Xcalibration和Ycalibration）, Xval个Yval指的是（Xvalidation和Yvalidation）,
    从变量的命名可以看出，Xcal和Ycal是需要计算的光谱矩阵（训练集），Xval和Yval是验证的光谱矩阵（测试集）。
（2）Xcal（训练集矩阵）和Xval（测试集矩阵）都是M*N的光谱矩阵（M为样本数，N为维度（波段））。
（3）Ycal和Yval在程序里没有注释，不是很清楚含义。Ycal和Yval都是M*1的维的矩阵，应该是是训练集和测试集的训练标签。
（4）分析前光谱数据需要平滑去噪，否则误差较大。
````
