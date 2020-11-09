# 算法原理
两类线性可分的模式类：$w_1$，$w_2$，设$d(X) = W^TX$,其中$W = [w_1,w_2,...,w_n,w_{n+1}]^T$,$X=[x_1,x_2,...,x_n,1]^T$,$d(X)>0$.

感知器算法通过对已知类别的训练样本集的学习，寻找一个满足上式的权向量。

# 算法步骤
* 选择N个分属于$w_1$和$w_2$类的模式样本构成训练样本集
  $${X_1,...X_N}$$
  构成增广向量形式，并进行规范化处理。任取权向量初始值$W(1)$，开始迭代，迭代次数$k=1$。
* 用全部训练样本进行一轮迭代，计算${W^T(k)}X_i$的值，并修正权向量。
  分两种情况更新权向量的值：
  1. 若${W^T(k)}X_i \leqslant0$，分类器对第$i$个模式做了错误分类，权向量校正为$W(k+1)=W(k)+cX_i$  $c$：正的校正增量
  2. 若${W^T(k)}X_i>0$，分类正确，权向量不变：
   $$W(k+1)=W(k)$$
* 分析分类结果，只要有一个错误分类，回到第二步，直至所有样本正确分类

# 算法特点
收敛性：经过有限次迭代运算后，求出了一个使所有样本都能正确分类的$W$，则称算法是收敛的。
收敛条件：模式类别线性可分

# 算法代码1
编写感知器算法程序，求下列模式分类的解向量
~~~ 
w1 = [[0 , 0 , 0 , 1] , [1 , 0 , 0 , 1] , [1 , 0 , 1 , 1] , [1 , 1 , 0 , 1] , [0 , 0 , -1 , -1] , [0 , -1 , -1 , -1] , [0 , -1 , 0 , -1] , [-1 , -1 , -1 , -1]]
w = [-1 , -2 , -2 , 0]
flag = 0
res = [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]
while(flag != 8):
    flag = 0
    for i in range(8):
        res[i] = w[0] * w1[i][0] + w[1] * w1[i][1] + w[2] * w1[i][2] + w[3] * w1[i][3]
        if res[i] <= 0:
            w[0] = w[0]+w1[i][0]
            w[1] = w[1]+w1[i][1]
            w[2] = w[2]+w1[i][2]
            w[3] = w[3]+w1[i][3]
            flag = flag - 1
        else: 
            w = w
            flag = flag + 1
print(w)
~~~

# 算法代码2
~~~
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def main():
    path =  'data.txt'
    data = pd.read_csv(path, header=None, names=['x', 'y','z'])
    X = np.array(data.iloc[:,:])

    X = np.column_stack((X, np.ones((len(X),1))))   
    X[4:,:] = - X[4:,:]
    
    c = 1
    w = np.array([-1,-2,-2,0])
    flag = True
    cnt = 0
    while flag:
        cnt += 1
        flag = False
        for x in X:
            # print(x)

            if w @ x <= 0:
                w = w + c*x
                flag = True
            
    print(w)
    print(cnt)
    
    fig = plt.figure(figsize=(12, 8))
    
    # 创建 3D 坐标系
    ax = fig.gca(fc='whitesmoke',
                projection='3d' 
                )

    x1 = np.linspace(0, 2, 9)
    x2 = np.linspace(0, 2, 9)
    
    x1,x2 = np.meshgrid(x1, x2)
    x3 = (-w[3] - w[0]*x1 -w[1]*x2)/w[2]
    ax.plot_surface(X=x1,
                Y=x2,
                Z=x3,
                color='b',
                alpha=0.2
               )
    ax.set(xlabel='X',
       ylabel='Y',
       zlabel='Z',
       xlim=(0, 2),
       ylim=(0, 2),
       zlim=(0, 2),
       xticks=np.arange(0, 2, 1),
       yticks=np.arange(0, 2, 1),
       zticks=np.arange(0, 2, 1)
      )
    
    half = int(len(X)/2)
    X[4:,:] = - X[4:,:]
    x = X[:half, 0]  
    y = X[:half, 1]  
    z = X[:half, 2]  
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z,c='y',marker='o')

    x2 = X[half:, 0]  
    y2 = X[half:, 1]  
    z2 = X[half:, 2]  
    ax.scatter(x2, y2, z2,c='r',marker='x')
    for i_x, i_y,i_z in zip(X[:,0],X[:,1],X[:,2]):
        ax.text(i_x, i_y,i_z, '({}, {},{})'.format(int(i_x), int(i_y),int(i_z)))
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.show()

main()
~~~
# 应用场景
对于$M$类模式应存在$M$个判决函数：{$d_i$，$i=1,...M$}

*算法主要内容*
设有$M$种模式类别：$w_1$,$w_2$,...,$w_M$
设其权向量初值为：$W_j(1)$,$j=1,...,M$
训练样本为增广向量形式，但不需要规范化处理
第$k$次迭代时，一个属于$w_i$类的模式样本$X$被送入分类器，计算所有判别函数
$$d_j(k)=W_j^T(k)X,j=1,...,M$$
分两种情况修改权向量：
+ 若$d_i(k)>d_j(k),\forall j\ne i;j=1,2,...,M$则权向量不变；
  
  $$W_j(k+1)=W_j(k),j=1,2,...,M$$

+ 若第$l$个权向量使$d_i(k)\leqslant d_i(k)$，则相应的权向量做调整，即:
 
$$\left\{\begin{matrix}
W_i(k+1)=W_i(k)+cX,\\
Wl(k+1)=W_l(k)-cX,\\ 
W_j(k+1)=W_j(k),j\ne i,l
\end{matrix}\right.
其中c为正的校正增量
$$