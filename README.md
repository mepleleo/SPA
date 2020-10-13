
## 原理

连续投影算法（successive projections algorithm， SPA） 是前向特征变量选择方法。SPA利用向量的投影分析，通过将波长投影到其他波长上，比较投影向量大小，以投影向量最大的波长为待选波长，然后基于矫正模型选择最终的特征波长。SPA选择的是含有最少冗余信息及最小共线性的变量组合。该算法简要步骤如下。

记初始迭代向量为 $x_{k(0)}$，需要提取的变量个数为$N$,光谱矩阵为$J$列。

1. 任选光谱矩阵的1列（第$j$列），把建模集的第$j$列赋值给$x_j$，记为 $x_{k(0)}$。

2. 将未选入的列向量位置的集合记为$s$,
   $$
   s=\lbrace j,1\leq{j}\leq{J}, j\notin \lbrace k(0), \cdots, k(n-1) \rbrace \rbrace
   $$

3. 分别计算$x_j$对剩余列向量的投影：
   $$
   P_{x_j} = x_j-(x^T_j x_{k(n-1)})x_{k(n-1)}(x^T_{k(n-1)}x_{k(n-1)})^{-1},j\in s
   $$

4. 提取最大投影向量的光谱波长，
   $$
   k(n) = arg(max(\| P_(x_j) \|), j \in s)
   $$

5. 令$x_j = p_x, j \in s$。

6. $n = n + 1$，如果$n < N$,则按公式（1）循环计算。

最后，提取出的变量为$\lbrace x_{k(n)} = 0, \cdots, N-1 \rbrace$。对应每一次循环中的$k(0)$和$N$，分别建立多元线性回归分析（MLR）模型，得到建模集交互验证均方根误差（RMSECV），对应不同的候选子集，其中最小的RMSECV值对应的$k(0)$和$N$就是最优值。一般SPA选择的特征波长分数$N$不能很大。

​                                                              -------------------------摘自《光谱及成像技术在农业中的应用》P130



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

   



