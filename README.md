
## 原理

连续投影算法（successive projections algorithm, SPA）是前向特征变量选择方法。

SPA利用向量的投影分析，通过将波长投影到其他波长上，比较投影向量大小，以投影向量最大的波长为待选波长，然后基于矫正模型选择最终的特征波长。
SPA选择的是含有最少冗余信息及最小共线性的变量组合。

该算法简要步骤如下:
记初始迭代向量为 xk(0)，需要提取的变量个数为N,光谱矩阵为J列。

任选光谱矩阵的1列（第j列），把建模集的第j列赋值给xj，记为 xk(0)。

将未选入的列向量位置的集合记为s,
s={j,1≤j≤J,j∉{k(0),⋯,k(n−1)}}
分别计算xj对剩余列向量的投影：
Pxj=xj−(xTjxk(n−1))xk(n−1)(xTk(n−1)xk(n−1))−1,j∈s
提取最大投影向量的光谱波长，
k(n)=arg(max(|P(xj)|),j∈s)
令xj=px,j∈s。

n=n+1，如果n<N,则按公式（1）循环计算。

最后，提取出的变量为{xk(n)=0,⋯,N−1}。对应每一次循环中的k(0)和N，分别建立多元线性回归分析（MLR）模型，得到建模集交互验证均方根误差（RMSECV），对应不同的候选子集，其中最小的RMSECV值对应的k(0)和N就是最优值。一般SPA选择的特征波长分数N不能很大。-------------------------摘自《光谱及成像技术在农业中的应用》P130



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
