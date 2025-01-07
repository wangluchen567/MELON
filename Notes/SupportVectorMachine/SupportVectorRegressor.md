# 支持向量机回归器

## 目标函数定义

给定数据$X = [x_1, x_2,..., x_n]^T, Y=[y_1,y_2,..., y_n]^T$，对于线性回归器，我们期望得到一个直线能尽可能的拟合数据，也就是得到的模型$f(x)=w^Tx+b$尽可能与真实值$y$接近，类比分类问题，可将目标函数定义为：
$$
\min_{w, b}\frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}l_\epsilon(f(x_i)-y_i)\\
$$
其中C为惩罚系数，$l_\epsilon(f(x_i)-y_i)$具体为：
$$
l_\epsilon(f(x_i)-y_i)=
\begin{cases}
0, \quad \quad \quad \quad \quad \quad \quad |f(x_i)-y_i|<\epsilon\\
|f(x_i)-y_i|-\epsilon, \quad |f(x_i)-y_i| \geq \epsilon\\
\end{cases}
$$
也就是说当预测值$f(x_i)$与真实值$y_i$之间的差异小于$\epsilon$时，损失为0，即不惩罚误差，否则计算损失值为$|f(x_i)-y_i|-\epsilon$.

接下来引入松弛变量$\xi_i$和$\hat\xi_i$，可将上式函数重写为：
$$
\min_{w, b, \xi_i, \hat\xi_i}\frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}(\xi_i + \hat\xi_i)\\
\text{s.t.} \quad f(x_i)-y_i \leq \epsilon + \xi_i, \quad y_i-f(x_i) \leq \epsilon + \hat\xi_i, 
\quad \xi_i \geq 0, \quad \hat\xi_i \geq 0
$$

## 拉格朗日函数定义

将得到的目标函数中每个约束条件都加入拉格朗日乘子，得到拉格朗日函数为：
$$
L(w,b,\xi,\hat\xi,\alpha,\hat\alpha,\beta,\hat\beta) = \frac{1}{2}\|w\|^2 + 
C \sum_{i=1}^{n} (\xi_i + \hat\xi_i) +
\sum_{i=1}^{n}\alpha_i[f(x_i)-y_i-\epsilon-\xi_i]+
\sum_{i=1}^{n}\hat\alpha_i[y_i-f(x_i)-\epsilon-\hat\xi_i]- 
\sum_{i=1}^{n}\beta_i\xi_i - \sum_{i=1}^{n}\hat\beta_i\hat\xi_i
$$
其中$\xi_i \geq 0$，$\hat\xi_i \geq 0$，$\alpha_i \geq 0$，$\hat\alpha_i \geq 0$，$\beta_i \geq 0$，$\hat\beta_i \geq 0$，从而原始目标函数转换为：
$$
\min_{w,b,\xi,\hat\xi} \max_{\alpha,\hat\alpha,\beta,\hat\beta}
L(w,b,\xi,\hat\xi,\alpha,\hat\alpha,\beta,\hat\beta)
$$
该问题的对偶问题为：
$$
\max_{\alpha,\hat\alpha,\beta,\hat\beta}\min_{w,b,\xi,\hat\xi}
L(w,b,\xi,\hat\xi,\alpha,\hat\alpha,\beta,\hat\beta)
$$
假设原问题的最优值为$d^*$，对偶问题的最优值为$p^*$，那么有$d^* \geq p^*$，（这里可以理解为凤头中的鸡尾(先最大再最小)一定大于等于凤尾中的鸡头(先最小后最大)）。若想要满足$d^* = p^*$，则需要满足该问题是凸优化问题，且还要满足KKT条件（Karush–Kuhn–Tucker conditions），所以有：
$$
\min_{w,b,\xi,\hat\xi} \max_{\alpha,\hat\alpha,\beta,\hat\beta}
L(w,b,\xi,\hat\xi,\alpha,\hat\alpha,\beta,\hat\beta)=\max_{\alpha,\hat\alpha,\beta,\hat\beta}\min_{w,b,\xi,\hat\xi}
L(w,b,\xi,\hat\xi,\alpha,\hat\alpha,\beta,\hat\beta)
$$
为了保证原问题与对偶问题求得的最优解相同，需要满足以下约束：
$$
\begin{align}
& ①\quad \alpha_i \geq 0, \quad \hat\alpha_i \geq 0, \quad \beta_i \geq 0, \quad \hat\beta_i \geq 0, & (乘子约束)\\
& ②\quad \xi_i \geq 0, \quad \hat\xi_i \geq 0, \quad f(x_i)-y_i \leq \epsilon + \xi_i, \quad y_i-f(x_i) \leq \epsilon + \hat\xi_i, & (原始约束)\\
& ③\quad \alpha_i[f(x_i)-y_i-\epsilon-\xi_i] = 0, \quad \hat\alpha_i[y_i-f(x_i)-\epsilon-\hat\xi_i]=0, \quad \beta_i\xi_i=0, \quad \hat\beta_i\hat\xi_i=0, & (KKT条件)\\
& ④\quad \frac{\partial L}{\partial w} = \frac{\partial L}{\partial b} = \frac{\partial L}{\partial \xi}= \frac{\partial L}{\partial \hat\xi}=0. & (KKT条件)\\
\end{align}
$$
将最后的关于偏导的KKT条件计算得到：
$$
\begin{align}
& \frac{\partial L}{\partial w} = w-\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i=0 &\Leftrightarrow 
\quad w=\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i \\
& \frac{\partial L}{\partial b} = \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0 &\Leftrightarrow 
\quad \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0\\
& \frac{\partial L}{\partial \xi} = C-\alpha_i-\beta_i=0 &\Leftrightarrow 
\quad \alpha_i+\beta_i=C\\
& \frac{\partial L}{\partial \hat\xi} = C-\hat\alpha_i-\hat\beta_i=0 &\Leftrightarrow 
\quad \hat\alpha_i+\hat\beta_i=C
\end{align}
$$
由于上式约束中的$\beta_i$和$\hat\beta_i$除了$\beta_i \geq 0$和$\hat\beta_i \geq 0$之外没有其他约束，所以约束可以转化为：
$$
\begin{align}
& \quad 0 \leq \alpha_i \leq C, \quad 0 \leq \hat\alpha_i \leq C,\\
& \quad \xi_i \geq 0, \quad \hat\xi_i \geq 0, \quad f(x_i)-y_i \leq \epsilon + \xi_i, \quad y_i-f(x_i) \leq \epsilon + \hat\xi_i,\\
& \quad \alpha_i[f(x_i)-y_i-\epsilon-\xi_i] = 0, \quad \hat\alpha_i[y_i-f(x_i)-\epsilon-\hat\xi_i]=0,\\
& \quad w=\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i, \quad \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0,
\quad (C-\alpha_i)\xi_i=0, \quad (C-\hat\alpha_i)\hat\xi_i=0.\\
\end{align}
$$
可以看到当且仅当$f(x_i)-y_i-\epsilon-\xi_i=0$时，$\alpha_i$取非零值，当且仅当$y_i-f(x_i)-\epsilon-\hat\xi_i=0$时，$\hat\alpha_i$取非零值。也就是说，仅当样本不落入$\epsilon$间的隔带中时，其对应的$\alpha_i$和$\hat\alpha_i$取非零值，而$f(x_i)-y_i-\epsilon-\xi_i=0$和$y_i-f(x_i)-\epsilon-\hat\xi_i=0$不能同时成立，所以$\alpha_i$和$\hat\alpha_i$至少有一个是0，从而导致$\xi_i$和$\hat\xi_i$至少有一个是0，如此一来完整的约束应该是：
$$
\begin{align}
& \quad 0 \leq \alpha_i \leq C, \quad 0 \leq \hat\alpha_i \leq C, \quad \alpha_i\hat\alpha_i=0, 
\quad \xi_i\hat\xi_i=0, \\
& \quad \xi_i \geq 0, \quad \hat\xi_i \geq 0, \quad f(x_i)-y_i \leq \epsilon + \xi_i, \quad y_i-f(x_i) \leq \epsilon + \hat\xi_i,\\
& \quad \alpha_i[f(x_i)-y_i-\epsilon-\xi_i] = 0, \quad \hat\alpha_i[y_i-f(x_i)-\epsilon-\hat\xi_i]=0,\\
& \quad w=\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i, \quad \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0,
\quad (C-\alpha_i)\xi_i=0, \quad (C-\hat\alpha_i)\hat\xi_i=0.\\
\end{align}
$$
将$w=\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i, \quad \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0,
\quad (C-\alpha_i)\xi_i=0, \quad (C-\hat\alpha_i)\hat\xi_i=0, \quad \beta_i=C-\alpha_i, \quad \hat\beta_i=C-\hat\alpha_i$代入得到约束对应的对偶问题的目标函数有：
$$
\max_{\alpha,\hat\alpha}L(\alpha,\hat\alpha)=-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}(\hat\alpha_i-\alpha_i)(\hat\alpha_j-\alpha_j)(\vec{x_i}^T·\vec{x_j})-\epsilon\sum_{i=1}^{n}(\hat\alpha_i+\alpha_i)+
\sum_{i=1}^{n}y_i(\hat\alpha_i-\alpha_i)\\
\text{s.t.} \quad \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0, \quad 0 \leq \alpha_i \leq C, \quad 0 \leq \hat\alpha_i \leq C
$$

考虑到$\vec{x_i}^T\cdot\vec{x_j}$为线性核函数，为了能推广到一般形式，也就是能使用非线性的核函数，将其替换为$K_{ij}$，关于各类核函数的定义如下：
$$
K_{ij}=K(\vec{x_i},\vec{x_j})=
\begin{align}
\begin{cases}
\vec{x_i}^T\cdot\vec{x_j} & (线性核)\\
(\gamma \vec{x_i}^T\cdot\vec{x_j}+r)^d & (多项式核)\\
exp(-\gamma \|\vec{x_i}-\vec{x_j}\|^2) & (高斯核/径向基核)\\
tanh(\gamma \vec{x_i}^T\cdot\vec{x_j}+r) & (Sigmoid核)
\end{cases}
\end{align}
$$
将$\vec{x_i}^T\cdot\vec{x_j}$替换为$K_{ij}$，并将最大化问题转换为最小化问题得到：
$$
\min_{\alpha,\hat\alpha}L(\alpha,\hat\alpha)=\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}(\hat\alpha_i-\alpha_i)(\hat\alpha_j-\alpha_j)K_{ij}+\epsilon\sum_{i=1}^{n}(\hat\alpha_i+\alpha_i)-
\sum_{i=1}^{n}y_i(\hat\alpha_i-\alpha_i)\\
\text{s.t.} \quad \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0, \quad 0 \leq \alpha_i \leq C, \quad 0 \leq \hat\alpha_i \leq C
$$
## 优化问题转化

很显然，该问题的优化函数与分类问题的优化函数完全不一样，它拥有两个无法再简化的乘子，无法直接使用SMO（Sequential Minimal Optimization）算法求解，为了使该问题适合SMO算法求解，这里需要将函数进行一定的变换，可将两个乘子$\alpha$和$\hat\alpha$合并成一个单独的乘子$\beta$（注意，这里的$\beta$与之前的乘子无关，是一个新引入的乘子），表示为：
$$
\beta = 
\begin{bmatrix}
\alpha \\
\hat\alpha
\end{bmatrix}_{2n \times 1}=
\begin{bmatrix}
\alpha_1 \\
\alpha_2 \\
... \\
\alpha_n \\
\hat\alpha_1 \\
\hat\alpha_2 \\
... \\
\hat\alpha_n \\
\end{bmatrix}
$$
那么优化问题可以转换为：
$$
\min_{\beta} L(\beta) = \frac{1}{2}\beta^T
\begin{bmatrix}
K&-K\\
-K&K
\end{bmatrix}\beta+
\begin{bmatrix}
\epsilon+y\\
\epsilon-y
\end{bmatrix}^T\beta\\
\text{s.t.} \quad 
\begin{bmatrix}
e\\
-e
\end{bmatrix}^T\beta=0, \quad 0 \leq \beta_i \leq C
$$
其中$K$为核函数求得的核矩阵(方阵)，其形状为$n\times n$，其中第$i$行，第$j$列元素为$K_{ij}$，所以$\begin{bmatrix}
K&-K\\
-K&K
\end{bmatrix}$形状为$2n\times 2n$

而$\begin{bmatrix}
\epsilon+y\\
\epsilon-y
\end{bmatrix}$是一个$2n\times 1$的矩阵，具体为：
$$
\begin{bmatrix}
\epsilon+y\\
\epsilon-y
\end{bmatrix}_{2n \times 1}=
\begin{bmatrix}
\epsilon+y_1\\
\epsilon+y_2\\
...\\
\epsilon+y_n\\
\epsilon-y_1\\
\epsilon-y_2\\
...\\
\epsilon-y_n\\
\end{bmatrix}_{2n \times 1}
$$
而在$\begin{bmatrix}
e\\
-e
\end{bmatrix}$中$e$表示形状为$n\times 1$的全1矩阵，$\begin{bmatrix}
e\\
-e
\end{bmatrix}$的前$n$个元素为$+1$，后$n$个元素为$-1$，具体为：
$$
\begin{bmatrix}
e\\
-e
\end{bmatrix}_{2n \times 1}=
\begin{bmatrix}
+1\\
+1\\
...\\
+1\\
-1\\
-1\\
...\\
-1\\
\end{bmatrix}_{2n \times 1}
$$
下面通过推导证明该转换方式是正确的：
$$
\begin{align}
L(\beta) = & \frac{1}{2}\beta^T
\begin{bmatrix}
K&-K\\
-K&K
\end{bmatrix}\beta+
\begin{bmatrix}
\epsilon+y\\
\epsilon-y
\end{bmatrix}^T\beta\\
= & \frac{1}{2}\begin{bmatrix}\alpha&\hat\alpha\end{bmatrix}\begin{bmatrix}K&-K\\-K&K\end{bmatrix}
\begin{bmatrix}\alpha\\\hat\alpha\end{bmatrix}+\begin{bmatrix}\epsilon+y&\epsilon-y\end{bmatrix}
\begin{bmatrix}\alpha\\\hat\alpha\end{bmatrix}\\
= & \frac{1}{2}\begin{bmatrix}\alpha K-\hat\alpha K&-\alpha K+\hat\alpha K\end{bmatrix}
\begin{bmatrix}\alpha\\\hat\alpha\end{bmatrix}+\alpha\cdot(\epsilon + y)+\hat\alpha\cdot(\epsilon - y)\\
= & \frac{1}{2}(\alpha^2\cdot K-2\alpha\hat\alpha \cdot K+\hat\alpha^2\cdot K)+
\alpha\cdot\epsilon + \alpha\cdot y+\hat\alpha\cdot\epsilon - \hat\alpha\cdot y\\
= & \frac{1}{2}(\alpha - \hat\alpha)^2\cdot K+
\epsilon\cdot(\alpha + \hat\alpha)+y\cdot(\alpha - \hat\alpha)\\
= & \frac{1}{2}(\hat\alpha - \alpha)^2\cdot K+
\epsilon\cdot(\hat\alpha + \alpha)-y\cdot(\hat\alpha - \alpha)\\
= & \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}(\hat\alpha_i-\alpha_i)(\hat\alpha_j-\alpha_j)K_{ij}+\epsilon\sum_{i=1}^{n}(\hat\alpha_i+\alpha_i)-
\sum_{i=1}^{n}y_i(\hat\alpha_i-\alpha_i)\\
= & L(\alpha, \hat\alpha)
\end{align}
$$
根据前面得到的转化结果，为方便后续求解，求解问题可进一步转化为：
$$
\begin{align}
\min_{\beta} L(\beta) = & \frac{1}{2}\beta^T\begin{bmatrix}K&-K\\-K&K\end{bmatrix}\beta+
\begin{bmatrix}\epsilon+y\\\epsilon-y\end{bmatrix}^T\beta\\
 = & \frac{1}{2}\beta^T(\begin{bmatrix}K&K\\K&K\end{bmatrix}\odot\begin{bmatrix}e&-e\\-e&e\end{bmatrix})
 \beta+\begin{bmatrix}\epsilon+y\\\epsilon-y\end{bmatrix}^T\beta\\
= & \frac{1}{2}\beta^T[\begin{bmatrix}K&K\\K&K\end{bmatrix}\odot(\begin{bmatrix}e\\-e\end{bmatrix}\begin{bmatrix}e\\-e\end{bmatrix}^T)]
\beta+\begin{bmatrix}\epsilon+y\\\epsilon-y\end{bmatrix}^T\beta\\
= & \frac{1}{2} (\beta \odot \begin{bmatrix}e\\-e\end{bmatrix})^T\begin{bmatrix}K&K\\K&K\end{bmatrix}(\beta \odot \begin{bmatrix}e\\-e\end{bmatrix}) + \begin{bmatrix}\epsilon+y\\\epsilon-y\end{bmatrix}^T\beta\\
& \text{s.t.} \quad \begin{bmatrix}e\\-e\end{bmatrix}^T\beta=0, \quad 0 \leq \beta_i \leq C
\end{align}
$$
为方便表示，我们令$Q = \begin{bmatrix}K&K\\K&K\end{bmatrix}$，$P=\begin{bmatrix}\epsilon+y\\\epsilon-y\end{bmatrix}$，$Z=\begin{bmatrix}e\\-e\end{bmatrix}$，那么有：
$$
\min_{\beta} L(\beta) = \frac{1}{2}(\beta\odot Z)^T\cdot Q\cdot (\beta\odot Z)+P^T\cdot\beta\\
\text{s.t.} \quad Z^T\cdot\beta=0, \quad 0 \leq \beta_i \leq C
$$
将其展开有：
$$
\min_{\beta} L(\beta) =  \frac{1}{2}\sum_{i=1}^{2n}\sum_{j=1}^{2n}\beta_i\beta_jz_iz_jQ_{ij}+
\sum_{i=1}^{2n}p_i\beta_i\\
\text{s.t.} \quad \sum_{i=1}^{2n}z_i\beta_i=0, \quad 0 \leq \beta_i \leq C
$$
可以看到，该形式与分类器得到的需要优化的拉格朗日函数非常相似，如此一来，转换后的问题就很容易使用SMO算法求解了。

## 算法求解准备

若要求解该最优化函数，只需要在不违反约束的情况下，想办法改变$\beta$的值，然后逐渐让该函数最小化，考虑到存在约束：
$$
\sum_{i=1}^{2n}z_i\beta_i=0, \quad z_i \in\{-1,1\}
$$
与分类器的优化相似，为了保证该约束可以一直成立，可以考虑只同时改变两个$\beta$的值，然后在保证不违反约束的情况下，使目标函数最小化，该思路就是由John Platt于1996年提出的称为SMO（Sequential Minimal Optimization）的算法。而在正式使用SMO算法之前，还需要先讨论在什么情况下是违反另一个约束的，以方便SMO算法优化求解。

考虑到还存在另一个约束：
$$
0 \leq \beta_i \leq C
$$
另外，因为
$$
\beta = 
\begin{bmatrix}
\alpha \\
\hat\alpha
\end{bmatrix}_{2n \times 1}=
\begin{bmatrix}
\alpha_1 \\
\alpha_2 \\
... \\
\alpha_n \\
\hat\alpha_1 \\
\hat\alpha_2 \\
... \\
\hat\alpha_n \\
\end{bmatrix}, \quad 0 \leq \alpha_i \leq C, \quad 0 \leq \hat\alpha_i \leq C
$$
所以我们这里需要根据$\alpha$的情况去讨论$\beta$的情况，之前得到的关于$\alpha$的约束有：
$$
\begin{align}
& \quad 0 \leq \alpha_i \leq C, \quad 0 \leq \hat\alpha_i \leq C, \quad \alpha_i\hat\alpha_i=0, 
\quad \xi_i\hat\xi_i=0, \\
& \quad \xi_i \geq 0, \quad \hat\xi_i \geq 0, \quad f(x_i)-y_i \leq \epsilon + \xi_i, \quad y_i-f(x_i) \leq \epsilon + \hat\xi_i,\\
& \quad \alpha_i[f(x_i)-y_i-\epsilon-\xi_i] = 0, \quad \hat\alpha_i[y_i-f(x_i)-\epsilon-\hat\xi_i]=0,\\
& \quad w=\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i, \quad \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0,
\quad (C-\alpha_i)\xi_i=0, \quad (C-\hat\alpha_i)\hat\xi_i=0.\\
\end{align}
$$
考虑到$\alpha_i\hat\alpha_i=0$，所以接下来只需要讨论约束的五种情况即可：

第一种情况是当$\hat\alpha_i=0$且$0<\alpha_i<C$时：
$$
\begin{align}
\begin{cases}
\hat\alpha_i=0  & \Rightarrow \hat\xi_i=0 & \Rightarrow f(x_i) \geq y_i - \epsilon\\
0<\alpha_i<C  & \Rightarrow \xi_i=0, f(x_i)-y_i-\epsilon-\xi_i=0 & \Rightarrow f(x_i) = y_i + \epsilon
\end{cases}
\end{align}
$$
第二种情况是当$\hat\alpha_i=0$且$\alpha_i=C$时：
$$
\begin{align}
\begin{cases}
\hat\alpha_i=0  & \Rightarrow \hat\xi_i=0 & \Rightarrow f(x_i) \geq y_i - \epsilon\\
\alpha_i=C  & \Rightarrow \xi_i\geq 0, f(x_i)-y_i-\epsilon-\xi_i=0 & \Rightarrow f(x_i) \geq y_i + \epsilon
\end{cases}
\end{align}
$$
前两种情况综合来看，当$\hat\alpha_i=0$且$\alpha_i>0$时，有$f(x_i) \geq y_i + \epsilon$，则满足约束，所以当违反前两种情况的约束时，有$f(x_i) < y_i + \epsilon$，而因为$p_j=\epsilon+y_i, z_j=+1, 1\leq j\leq n$，所以有$f(x_j) < z_jp_j$，两边同乘以$z_j$有：$z_jf(x_j) < p_j$.

第三种情况是$\alpha_i=0$且$0<\hat\alpha_i<C$：
$$
\begin{align}
\begin{cases}
\alpha_i=0  & \Rightarrow \xi_i=0 & \Rightarrow f(x_i) \leq y_i + \epsilon\\
0<\hat\alpha_i<C  & \Rightarrow \hat\xi_i=0, y_i-f(x_i)-\epsilon-\hat\xi_i=0 & \Rightarrow f(x_i) = y_i - \epsilon
\end{cases}
\end{align}
$$
第四种情况是$\alpha_i=0$且$\hat\alpha_i=C$：
$$
\begin{align}
\begin{cases}
\alpha_i=0  & \Rightarrow \xi_i=0 & \Rightarrow f(x_i) \leq y_i + \epsilon\\
\hat\alpha_i=C  & \Rightarrow \hat\xi_i\geq 0, y_i-f(x_i)-\epsilon-\hat\xi_i=0 & \Rightarrow f(x_i) \leq y_i - \epsilon
\end{cases}
\end{align}
$$
第三四种情况综合来看，当$\alpha_i=0$且$\hat\alpha_i>0$时，有$f(x_i) \leq y_i - \epsilon$，则满足约束，所以当违反前两种情况的约束时，有$f(x_i) > y_i - \epsilon$，而因为$p_j=\epsilon-y_i, z_j=-1, n < j\leq 2n$，所以有$f(x_j) > z_jp_j$，两边同乘以$z_j$有：$z_jf(x_j) < p_j$. ($z_j<0$所以变号)

第五种情况是$\hat\alpha_i=0$且$\alpha_i=0$：
$$
\begin{align}
\begin{cases}
\alpha_i=0  & \Rightarrow \xi_i=0 & \Rightarrow f(x_i) \leq y_i + \epsilon\\
\hat\alpha_i=0  & \Rightarrow \hat\xi_i= 0 & \Rightarrow f(x_i) \geq y_i - \epsilon
\end{cases}
\end{align}
$$
即满足该约束时$y_i-\epsilon \leq f(x_i) \leq y_i+\epsilon$，而对于$p_j$和$z_j$有：
$$
p_j=\begin{cases}\epsilon+y_i, \quad 1\leq j \leq n\\ \epsilon-y_i, \quad n < j \leq 2n\end{cases}\quad
z_j=\begin{cases}+1, \quad 1\leq j \leq n\\ -1, \quad n < j \leq 2n\end{cases}
$$
所以该约束可以转化为$z_jf(x_j)\leq p_j$，所以当违反该约束时有：$z_jf(x_j) > p_j$.

综上，当$\alpha_i$和$\hat\alpha_i$其中一个大于0时，即$\beta_j > 0$时，$z_jf(x_j) < p_j$则违反约束；当$\alpha_i$和$\hat\alpha_i$都为0时，即$\beta_j < C$时，$z_jf(x_j) > p_j$则违反约束，即下面两种情况下违反约束：
$$
\begin{cases}
\beta_j > 0, \quad z_jf(x_j) < p_j\\
\beta_j < C, \quad z_jf(x_j) > p_j \\
\end{cases}
$$
所以我们只需要将这些违反约束的$\beta$找出来进行调整修改，使其满足约束即可。

## SMO算法求解

### 确定$\beta$的范围

由于回归问题转化后优化函数与分类问题非常相似，所以在确定$\beta$范围时可以直接仿照分类问题确定$\alpha$范围进行实现。

在前面的介绍中可知，为了满足约束$\sum_{i=1}^{n}\beta_iz_i=0, \quad y_i \in \{-1,1\}$，SMO算法的求解思路是同时改变一对$\beta$的值，对函数进行最优化，这里假设要修改的$\beta$为$\beta_1$和$\beta_2$，修改前为$\beta_1^{old}$和$\beta_2^{old}$，修改后为$\beta_1^{new}$和$\beta_2^{new}$，为保证约束，修改前和修改后的$\beta$需要满足：
$$
\beta_1^{old}z_1 + \beta_2^{old}z_2 = \beta_1^{new}z_1 + \beta_2^{new}z_2=\zeta
$$
其中$\zeta$是一个常数，剩余的$\sum_{i=3}^{n}\beta_i=-\zeta$，由于不好同时求解$\beta_1$和$\beta_2$，所以可先求其中一个，另一个可以根据约束，通过求解第一个的变化情况得到，这里是先对$\beta_2$进行求解，而由于$0 \leq \beta_i \leq C$，所以在改变$\beta$值时，为保证约束，要将新得到的$\beta$值的范围进行裁剪，并且考虑到$z_i \in \{-1,1\}$，所以下面需要讨论两种情况：

第一种情况：$z_1 \neq z_2$，由于$z_1 \neq z_2$，所以有：
$$
\beta_1^{old} - \beta_2^{old} = \beta_1^{new} - \beta_2^{new}=\zeta
$$
而由于$0 \leq \beta_i \leq C$，所以有：
$$
0 \leq \beta_1^{new} \leq C, \quad 0 \leq \beta_2^{new} \leq C
$$
而$\beta_1^{new} = \beta_2^{new} + \zeta$，所以有
$$
0 \leq \beta_2^{new} + \zeta \leq C, \quad 0 \leq \beta_2^{new} \leq C
$$
即：
$$
-\zeta \leq \beta_2^{new} \leq C - \zeta, \quad 0 \leq \beta_2^{new} \leq C
$$
所以有：
$$
max(0, -\zeta) \leq \beta_2^{new} \leq min(C, C-\zeta)
$$
而因为：
$$
\beta_1^{old} - \beta_2^{old} = \beta_1^{new} - \beta_2^{new}=\zeta
$$
所以有：
$$
max(0, \beta_2^{old} - \beta_1^{old}) \leq \beta_2^{new} \leq min(C, C+\beta_2^{old} - \beta_1^{old})
$$
第二种情况：$z_1 = z_2$，由于$z_1 = z_2$，所以有：
$$
\beta_1^{old} + \beta_2^{old} = \beta_1^{new} + \beta_2^{new}=\zeta
$$
而由于$0 \leq \beta_i \leq C$，所以有：
$$
0 \leq \beta_1^{new} \leq C, \quad 0 \leq \beta_2^{new} \leq C
$$
而$\beta_1^{new} = \zeta - \beta_2^{new}$，所以有
$$
0 \leq \zeta - \beta_2^{new} \leq C, \quad 0 \leq \beta_2^{new} \leq C
$$
即：
$$
\zeta-C \leq \beta_2^{new} \leq \zeta, \quad 0 \leq \beta_2^{new} \leq C
$$
所以有：
$$
max(0, \zeta-C) \leq \beta_2^{new} \leq min(C, \zeta)
$$
而因为：
$$
\beta_1^{old} + \beta_2^{old} = \beta_1^{new} + \beta_2^{new}=\zeta
$$
所以有：
$$
max(0, \beta_2^{old} + \beta_1^{old} - C) \leq \beta_2^{new} \leq min(C, \beta_2^{old} + \beta_1^{old})
$$
综上，$\beta_2^{new}$的上下界为：
$$
L\leq \beta_2^{new} \leq H  \Leftrightarrow 
\begin{align}
\begin{cases}
L=max(0, \beta_2^{old} - \beta_1^{old}), H=min(C, C+\beta_2^{old} - \beta_1^{old}), & z_1 \neq z_2\\
L=max(0, \beta_2^{old} + \beta_1^{old} - C), H=min(C, \beta_2^{old} + \beta_1^{old}), & z_1 = z_2
\end{cases}
\end{align}
$$

### 更新$\beta$值

得到$\beta_2^{new}$的上下界后，接下来就是要得到$\beta_2^{new}$的值，然后利用得到的上下界进行裁剪。而我们之前得到的需要优化的函数为：
$$
\min_{\beta} L(\beta) =  \frac{1}{2}\sum_{i=1}^{2n}\sum_{j=1}^{2n}\beta_i\beta_jz_iz_jQ_{ij}+
\sum_{i=1}^{2n}p_i\beta_i\\
\text{s.t.} \quad \sum_{i=1}^{2n}z_i\beta_i=0, \quad 0 \leq \beta_i \leq C
$$
由于我们求解时只修改$\beta_1$和$\beta_2$，将与$\beta_1$和$\beta_2$有关的项提出有：
$$
\begin{align}
L(\beta) 
& = \sum_{i=1}^{2n}p_i\beta_i + \frac{1}{2}\sum_{i=1}^{2n}\sum_{j=1}^{2n}\beta_i\beta_jz_iz_jQ_{ij}\\
& = p_1\beta_1 + p_2\beta_2 + \sum_{i=3}^{2n}p_i\beta_i +
\frac{1}{2}\sum_{i=1}^{2n}[\sum_{j=1}^{2}z_iz_j\beta_i\beta_jQ_{ij}+
\sum_{j=3}^{2n}z_iz_j\beta_i\beta_jQ_{ij}]\\
& = p_1\beta_1 + p_2\beta_2 + \sum_{i=3}^{2n}p_i\beta_i + 
\frac{1}{2}\sum_{i=1}^{2}[\sum_{j=1}^{2}z_iz_j\beta_i\beta_jQ_{ij}+
\sum_{j=3}^{2n}z_iz_j\beta_i\beta_jQ_{ij}] + 
\frac{1}{2}\sum_{i=3}^{2n}[\sum_{j=1}^{2}z_iz_j\beta_i\beta_jQ_{ij}+
\sum_{j=3}^{2n}z_iz_j\beta_i\beta_jQ_{ij}]\\
& = p_1\beta_1 + p_2\beta_2 + \sum_{i=3}^{2n}p_i\beta_i + 
\frac{1}{2}\sum_{i=1}^{2}\sum_{j=1}^{2}z_iz_j\beta_i\beta_jQ_{ij} + 
\sum_{i=1}^{2}\sum_{j=3}^{2n}z_iz_j\beta_i\beta_jQ_{ij} + 
\frac{1}{2}\sum_{i=3}^{2n}\sum_{j=3}^{2n}z_iz_j\beta_i\beta_jQ_{ij}\\
& = p_1\beta_1 + p_2\beta_2 + 
\frac{1}{2}\beta_1^2Q_{11} + \frac{1}{2}\beta_2^2Q_{22} + z_1z_2\beta_1\beta_2Q_{12}+
z_1\beta_1\sum_{j=3}^{2n}\beta_jz_jQ_{1j} + z_2\beta_2\sum_{j=3}^{2n}\beta_jz_jQ_{2j} +
\frac{1}{2}\sum_{i=3}^{2n}\sum_{j=3}^{2n}z_iz_j\beta_i\beta_jQ_{ij} + \sum_{i=3}^{2n}p_i\beta_i
\end{align}
$$
为方便表示，这里定义：(注意这里$f(x_i)$的定义，考虑到具体的$\hat\alpha$和$\alpha$，需要定义为负的系数)
$$
f(x_i)=-\sum_{j=1}^{2n}\beta_jz_jQ_{ij}+b=\sum_{j=1}^{n}(\hat\alpha_j-\alpha_j)K_{ij}+b\\
v_i=\sum_{j=3}^{2n}\beta_jz_jQ_{ij}=-f(x_i)-\sum_{j=1}^{2}\beta_jz_jQ_{ij}-b=
-f(x_i)-\beta_1z_1Q_{1i}-\beta_2z_2Q_{2i}-b
$$
所以要优化的函数变为：
$$
\max_{\beta}L(\beta) = p_1\beta_1 + p_2\beta_2 + 
\frac{1}{2}\beta_1^2Q_{11} + \frac{1}{2}\beta_2^2Q_{22} + z_1z_2\beta_1\beta_2Q_{12}+
z_1\beta_1v_1 + z_2\beta_2v_2 + Constant(常数项)\\
\text{s.t.} \quad 0 \leq \beta_i \leq C, \quad \sum_{i=1}^{2n}\beta_iz_i=0
$$
其中$Constant$为常数项，因为与$\beta_1$和$\beta_2$无关，并且之后求导会直接变为0，这里不进行展开。

接下来将$\beta_1$用$\beta_2$表示，由于存在约束$\sum_{i=1}^{2n}\beta_iz_i=0$，所以令$\beta_1z_1+\beta_2z_2=\zeta$，两边同时乘以$z_1$有：$\beta_1z_1z_1+\beta_2z_2z_1=z_1\zeta$，而$z_1z_1=1$，令$\gamma=z_1\zeta$，$s=z_1z_2$，那么有：
$$
\beta_1 = \gamma - s \beta_2
$$
将$\beta_1 = \gamma - s \beta_2$及$s=z_1z_2$代入函数得：
$$
L(\beta_2) = p_1(\gamma - s\beta_2) + p_2\beta_2 + 
\frac{1}{2}(\gamma - s\beta_2)^2Q_{11} + \frac{1}{2}\beta_2^2Q_{22} + 
s(\gamma - s\beta_2)\beta_2Q_{12}+
z_1(\gamma - s\beta_2)v_1+z_2\beta_2v_2 + Constant(常数项)\\
$$
如此一来要优化的函数中就只剩下$\beta_2$一个变量了，该函数对$\beta_2$的偏导数为0时，该函数取得极值，所以下面需要对该函数求偏导（这里注意$s^2=z_1z_1z_2z_2=1$）：
$$
\begin{align}
\frac{\partial L(\beta_2)}{\partial \beta_2} & = -sp_1+p_2-\gamma sQ_{11}+\beta_2Q_{11}+\beta_2Q_{22}+
\gamma sQ_{12}-2\beta_2Q_{12}-z_2v_1+z_2v_2\\
& = -z_1z_2p_1+p_2-\gamma z_1z_2Q_{11}+\beta_2Q_{11}+\beta_2Q_{22}+
\gamma z_1z_2Q_{12}-2\beta_2Q_{12}-z_2v_1+z_2v_2\\
& = 0
\end{align}
$$
所以有：
$$
\beta_2 = \frac{z_2[z_1p_1-z_2p_2+z_1\gamma(Q_{11}-Q_{12})+v_1-v_2]}{Q_{11} + Q_{22} - 2Q_{12}}
$$
我们得到了$\beta_2$的值，之后需要根据这个值更新$\beta_2$，而根据之前的定义有：
$$
\begin{align}
\gamma & = \beta_1+ s\beta_2 \\
& = \beta_1^{new}+ s\beta_2^{new} \\
& = \beta_1^{old}+ s\beta_2^{old}\\
& =\beta_1^{new}+ z_1z_2\beta_2^{new}\\
& =\beta_1^{old}+ z_1z_2\beta_2^{old}
\end{align}
$$


如此一来，我们可以将之前的$\beta^{old}$代入，然后将$v_i=-f(x_i)-z_1\beta_1Q_{1i}-z_2\beta_2Q_{2i}-b$也代入，另外为了表示方便，这里令$\eta = Q_{11} + Q_{22} - 2Q_{12}$表示学习速率，另外还要注意$Q_{12}=Q_{21}$且$z_1z_1=z_2z_2=1$，所以有：
$$
\begin{align}
\beta_2^{new} & = \frac{z_2[z_1p_1-z_2p_2+z_1\gamma(Q_{11}-Q_{12})+v_1-v_2]}{Q_{11} + Q_{22} - 2Q_{12}}\\
& = z_2[z_1p_1-z_2p_2+z_1(\beta_1^{old}+ z_1z_2\beta_2^{old})(Q_{11}-Q_{12})-
f(x_1)-z_1\beta_1^{old}Q_{11}-z_2\beta_2^{old}Q_{12}-b+
f(x_2)+z_1\beta_1^{old}Q_{12}+z_2\beta_2^{old}Q_{22}+b]\frac{1}{\eta}\\
& = z_2[z_1p_1-z_2p_2+z_1\beta_1^{old}Q_{11}-z_1\beta_1^{old}Q_{12}+
z_2\beta_2^{old}Q_{11}-z_2\beta_2^{old}Q_{12}-
f(x_1)-z_1\beta_1^{old}Q_{11}-z_2\beta_2^{old}Q_{12}-b+
f(x_2)+z_1\beta_1^{old}Q_{12}+z_2\beta_2^{old}Q_{22}+b]\frac{1}{\eta}\\
& = z_2[(z_1p_1-f(x_1))-(z_2p_2-f(x_2))+z_2\beta_2^{old}Q_{11}+z_2\beta_2^{old}Q_{22}-
2 z_2\beta_2^{old}Q_{12}]\frac{1}{\eta}\\
& = z_2[(z_1p_1-f(x_1))-(z_2p_2-f(x_2))+z_2\beta_2^{old}\eta]\frac{1}{\eta}\\
& = \beta_2^{old} +\frac{z_2[(z_1p_1-f(x_1))-(z_2p_2-f(x_2))]}{\eta}
\end{align}
$$
这里令$E_i=z_ip_i-f(x_i)$表示误差项，$\eta = Q_{11} + Q_{22} - 2Q_{12}$表示学习速率，那么最后可得到$\beta_2$的更新公式为：
$$
\beta_2^{new} = \beta_2^{old} +\frac{z_2(E_1-E_2)}{\eta}
$$
在得到新的$\beta_2^{new}$之后，我们还要使用之前推出的上下界对其进行裁剪，所以裁剪后的结果为：
$$
\beta_2^{new,clipped}=
\begin{cases}
H, \quad \quad\beta_2^{new} > H\\
\beta_2^{new}, \quad L \leq \beta_2^{new} \leq H\\
L, \quad \quad\beta_2^{new} < L
\end{cases}
$$
我们得到了$\beta_2$，接下来需要根据$\beta_2$求得$\beta_1$，已知$\gamma = \beta_1 - s \beta_2=\beta_1 - z_1z_2 \beta_2$，所以有：
$$
\gamma =\beta_1^{old} - z_1z_2 \beta_2^{old}=\beta_1^{new} - z_1z_2 \beta_2^{new,clipped}
$$
最终得到$\beta_1^{new}$的值为：
$$
\beta_1^{new} =\beta_1^{old} + z_1z_2 (\beta_2^{old} -\beta_2^{new,clipped})
$$

### 更新$b$值

在更新了$\beta$值之后，还需要重新计算并更新阈值$b$的值，因为阈值$b$关系到输出函数$f(x)$和误差项$E$的计算，当$0 \leq \beta_1^{new} \leq C$时，对应的数据点是支持向量上的点，此时满足：
$$
|f(x_1)-y_1|=\epsilon
$$
将$w=\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i$代入有：
$$
|\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i\cdot x_1-y_1|=\epsilon
$$
将核函数由线性核函数推广到一般形式后有：
$$
|\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)K_{i1}-y_1|=\epsilon
$$
利用$\alpha$和$\beta$的关系将$\alpha$和$\hat\alpha$替换为$\beta$有：
$$
-\sum_{i=1}^{2n}\beta_iz_iQ_{i1}+b=
\begin{cases}
y_1+\epsilon, \quad \beta_1 \in \alpha\\
y_1-\epsilon, \quad \beta_1 \in \hat\alpha\\
\end{cases}=
\begin{cases}
z_1(\epsilon+y_1), \quad \beta_1 \in \alpha\\
z_1(\epsilon-y_1), \quad \beta_1 \in \hat\alpha\\
\end{cases}
=z_1p_1
$$
所以有：
$$
\begin{align}
b_1^{new} & = z_1p_1 + \sum_{i=1}^{2n}\beta_iz_iQ_{i1}\\
& = z_1p_1 + \sum_{i=3}^{2n}\beta_iz_iQ_{i1} + \beta_1^{new}z_1Q_{11} + \beta_2^{new}z_2Q_{12}
\end{align}
$$
其中前两项为：
$$
\begin{align}
z_1p_1 + \sum_{i=3}^{2n}\beta_iz_iQ_{i1} 
& = z_1p_1 + \sum_{i=3}^{2n}\beta_iz_iQ_{i1}+f(x_1)-f(x_1)\\
& = (z_1p_1 - f(x_1)) - \beta_1^{old}z_1Q_{11} - \beta_2^{old}z_2Q_{12} + b^{old}\\
& = E - \beta_1^{old}z_1Q_{11} - \beta_2^{old}z_2Q_{12} + b^{old}
\end{align}
$$
将其代入有：
$$
\begin{align}
b_1^{new} & = z_1p_1 + \sum_{i=3}^{2n}\beta_iz_iQ_{i1} + \beta_1^{new}z_1Q_{11} + \beta_2^{new}z_2Q_{12}\\
& = E_1 - \beta_1^{old}z_1Q_{11} - \beta_2^{old}z_2Q_{12}+b^{old} + \beta_1^{new}z_1Q_{11} + 
\beta_2^{new}z_2Q_{12}\\
& = b^{old} + E_1 + z_1(\beta_1^{new}-\beta_1^{old})Q_{11} + z_2(\beta_2^{new}-\beta_2^{old})Q_{12}
\end{align}
$$
同理可得：
$$
b_2^{new} = b^{old} + E_2 + z_1(\beta_1^{new}-\beta_1^{old})Q_{12} + z_2(\beta_2^{new}-\beta_2^{old})Q_{22}
$$
从理论上说，当$b_1^{new}$和$b_2^{new}$都有效的时候，两个乘子$\beta_1^{new}$和$\beta_2^{new}$对应的点都在间隔超平面上，一定有：
$$
b^{new} = b_1^{new} = b_2^{new}
$$
当都不满足的时候，SMO算法选择$b_1^{new}$和$b_2^{new}$的平均值(中点)作为新的阈值$b^{new}$，所以最终阈值$b^{new}$的取值为：
$$
b^{new} = 
\begin{cases}
b_1^{new}, \quad 0<\beta_1^{new}<C\\
b_2^{new}, \quad 0<\beta_2^{new}<C\\
\frac{b_1^{new}+b_2^{new}}2{}, otherwise.
\end{cases}
$$

### 优化步骤总结

梳理SMO算法的具体步骤如下：

1.找出违反约束的乘子$\beta_i$和$\beta_j$(违反下面的条件则违反约束)：
$$
\begin{cases}
\beta_i > 0, \quad z_if(x_i) < p_i \quad \Leftrightarrow \quad z_i(f(x_i) - z_ip_i) < 0 \quad \Leftrightarrow \quad z_iE_i > 0\\
\beta_i < C, \quad z_if(x_i) > p_i \quad \Leftrightarrow \quad z_i(f(x_i) - z_ip_i) > 0 \quad \Leftrightarrow \quad z_iE_i < 0\\
\end{cases}
$$

$$
\begin{cases}
\beta_j > 0, \quad z_jf(x_j) < p_j \quad \Leftrightarrow \quad z_j(f(x_j) - z_jp_j) < 0 \quad \Leftrightarrow \quad z_jE_j > 0\\
\beta_j < C, \quad z_jf(x_j) > p_j \quad \Leftrightarrow \quad z_j(f(x_j) - z_jp_j) > 0 \quad \Leftrightarrow \quad z_jE_j < 0\\
\end{cases}
$$

（这里为避免重复计算，在具体实现时可以先计算误差，再检查是否违反约束）

2.计算误差$E_i$和$E_j$：
$$
E_i = z_ip_i-f(x_i)=\sum_{j=1}^{2n}\beta_jz_jQ_{ij}-b+z_ip_i\\
E_j = z_jp_j-f(x_j)=\sum_{i=1}^{2n}\beta_iz_iQ_{ij}-b+z_jp_j
$$
3.计算$\beta_j$的上下界$L$和$H$：
$$
\begin{cases}
L=max(0, \beta_j^{old} - \beta_i^{old}), H=min(C, C+\beta_j^{old} - \beta_i^{old}), \quad z_i \neq z_j\\
L=max(0, \beta_j^{old} + \beta_i^{old} - C), H=min(C, \beta_j^{old} + \beta_i^{old}), \quad z_i = z_j
\end{cases}
$$
4.计算学习速率$\eta$：
$$
\eta = Q_{ii} + Q_{jj} - 2Q_{ij}
$$
5.更新$\beta_j$：
$$
\beta_j^{new} = \beta_j^{old} +\frac{z_j(E_i-E_j)}{\eta}
$$
6.根据上下界裁剪$\beta_j$：
$$
\beta_j^{new,clipped}=
\begin{cases}
H, \quad \quad\beta_j^{new} > H\\
\beta_j^{new}, \quad L \leq \beta_j^{new} \leq H\\
L, \quad \quad\beta_j^{new} < L
\end{cases}
$$
7.更新$\beta_i$：
$$
\beta_i^{new} =\beta_i^{old} + z_iz_j (\beta_j^{old} -\beta_j^{new,clipped})
$$
8.更新$b_i$和$b_j$：
$$
b_i^{new} = b^{old} + E_i + z_i(\beta_i^{new}-\beta_i^{old})Q_{ii} + z_j(\beta_j^{new}-\beta_j^{old})Q_{ij}\\
b_j^{new} = b^{old} + E_j + z_i(\beta_i^{new}-\beta_i^{old})Q_{ij} + z_j(\beta_j^{new}-\beta_j^{old})Q_{jj}
$$
9.更新$b$：
$$
b^{new} = 
\begin{cases}
b_i^{new}, \quad 0<\beta_i^{new}<C\\
b_j^{new}, \quad 0<\beta_j^{new}<C\\
\frac{b_i^{new}+b_j^{new}}2{}, otherwise.
\end{cases}
$$
PS:

这里注意一点，在推导过程中问题转换后其实不用再考虑$\alpha_i\hat\alpha_i=0$这个约束，具体原因是，由于与变量直接相关的只有$(\hat\alpha_i-\alpha_i)$，所以目标函数已经将$\alpha_i$和$\hat\alpha_i$视为了一个整体，每一对元素都可以通过等量偏移满足约束，假设这里得到的最优解为$\alpha_i^*$和$\hat\alpha_i^*$，$\alpha_i^*\hat\alpha_i^*\neq0$，那么我们可以将这一对变量进行偏移有：
$$
\hat\alpha_i^*-\alpha_i^* = (\hat\alpha_i^*+m_i)-(\alpha_i^*+m_i) = \hat\alpha_i-\alpha_i\\
\alpha_i^*\hat\alpha_i^*\neq0, \quad \alpha_i\alpha_i = 0
$$
那么对于最优函数值的影响是：
$$
\min_{\alpha,\hat\alpha}L = \min_{\alpha,\hat\alpha}L^*+\epsilon\sum_{i=1}^{n}m_i=\min_{\alpha,\hat\alpha}L^*+\epsilon M
$$

对最终结果的$w$影响为：(以线性核为例)
$$
w^*=\sum_{i=1}^{n}(\hat\alpha_i^*-\alpha_i^*)x_i=\sum_{i=1}^{n}[(\hat\alpha_i^*+m_i)-(\alpha_i^*+m_i)]x_i=\sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)x_i=w
$$
同理，对函数值$f(x)$与误差$E$也不会产生影响，也就是说变量偏移只会对最优函数值造成影响，而对最终得到的结果$w$和$b$没有影响。

## 参考资料

\[1\] [John C. Platt. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines](http://leap.ee.iisc.ac.in/sriram/teaching/MLSP_16/refs/SMO.pdf)

\[2\] [Python3《机器学习实战》学习笔记（八）：支持向量机原理篇之手撕线性SVM - 知乎](https://zhuanlan.zhihu.com/p/29604517)

\[3\] [Python3《机器学习实战》学习笔记（九）：支持向量机实战篇之再撕非线性SVM - 知乎](https://zhuanlan.zhihu.com/p/29872905)

\[4\] [深入理解拉格朗日乘子法（Lagrange Multiplier) 和KKT条件_真正理解拉格朗日乘子法和 kkt 条件-CSDN博客](https://blog.csdn.net/xianlingmao/article/details/7919597)

\[5\] [Python · SVM（二）· LinearSVM - 知乎](https://zhuanlan.zhihu.com/p/27293420)

\[6\] [Python · SVM（四）· SMO 算法 - 知乎](https://zhuanlan.zhihu.com/p/27662928)

\[7\] [Python3《机器学习实战》学习笔记（九）：支持向量机实战篇之再撕非线性SVM - 知乎](https://zhuanlan.zhihu.com/p/29872905)

\[8\] [支持向量机（SVM）和支持向量机回归（SVR） - 知乎](https://zhuanlan.zhihu.com/p/76609851)

\[9\] [SMO求解支持向量回归 - 邢存远的博客 | Welt Xing's Blog](https://welts.xyz/2021/09/16/svr/)
