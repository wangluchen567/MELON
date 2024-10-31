# 支持向量机分类器

## 目标函数定义

给定数据$X = [x_1, x_2,..., x_n]^T, Y=[y_1,y_2,..., y_n]^T \in \{-1, 1\}$，对于线性分类器，我们期望得到一个分隔线能将两类数据正确分隔，分隔线可定义为：
$$
w^TX+b=0
$$
其中$w^T$为分隔线的法向量方向，我们的目标是最大化所有分类数据到分隔线的距离，目标函数可以定义为：
$$
\max_{w, b} d = \frac{|w^TX+b|}{\|w\|}
$$
所以我们希望对于任意样本数据，都有：
$$
\begin{cases}
w^Tx_i+b > 0,\quad y_i=+1\\
w^Tx_i+b < 0,\quad y_i=-1
\end{cases}
$$
我们暂时考虑其为硬间隔（hard margin）问题，则必须有：
$$
\begin{cases}
\frac{|w^Tx_i+b|}{\|w\|} \geq d,\quad y_i=+1\\
\frac{|w^Tx_i+b|}{\|w\|} \leq d,\quad y_i=-1
\end{cases}
$$
其等价于：
$$
\begin{cases}
w^Tx_i+b \geq 1,\quad y_i=+1\\
w^Tx_i+b \leq 1,\quad y_i=-1
\end{cases}
$$
可以用一个公式来表示两种情况：
$$
y_i(w^Tx_i+b) \geq 1
$$
由于我们使用支持向量上的样本点对目标进行最大化，所以对任意支持向量的样本点有$|w^TX+b|=1$，然后我们将最大化问题转换为最小化问题为：
$$
\max_{w, b} d = \min_{w, b}\|w\|=\min_{w, b}\frac{1}{2}\|w\|^2\\
\text{s.t.} \quad y_i(w^Tx_i+b) \geq 1
$$
后面转换为平方的一半是为了方便之后的求导，并且最小化结果不变。

由于在真实数据中硬间隔可能无法使用，所以接下来放松约束，将问题考虑为软间隔（soft margin）问题：
$$
\min_{w, b, \xi}\frac{1}{2}\|w\|^2\\
\text{s.t.} \quad y_i(w^Tx_i+b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$
然而直接这样放松约束很可能会使得函数或模型变得“怠惰”，所以这里还要加入惩罚项，最终带惩罚项的目标函数为：
$$
\min_{w, b, \xi}\frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i\\
\text{s.t.} \quad y_i(w^Tx_i+b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

其中C为惩罚值，C越大惩罚越严重，模型对训练数据分类越精确，但泛化性会变差。

## 拉格朗日函数定义

将得到的目标函数中每个约束条件都加入拉格朗日乘子，得到拉格朗日函数为：
$$
L(w,b,\xi,\alpha, \beta) = \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i - \sum_{i=1}^{n}\alpha_i[y_i(w^Tx_i+b)-1+\xi_i]-\sum_{i=1}^{n}\beta_i\xi_i
$$
其中$\xi_i \geq 0$，$\alpha_i \geq 0$，$\beta_i \geq 0$，从而原始目标函数转换为：
$$
\min_{w, b, \xi} \max_{\alpha,\beta} L(w,b,\xi,\alpha, \beta)
$$
该问题的对偶问题为：
$$
\max_{\alpha,\beta} \min_{w, b, \xi} L(w,b,\xi,\alpha, \beta)
$$


假设原问题的最优值为$d^*$，对偶问题的最优值为$p^*$，那么有$d^* \geq p^*$，（这里可以理解为凤头中的鸡尾(先最大再最小)一定大于等于凤尾中的鸡头(先最小后最大)）。若想要满足$d^* = p^*$，则需要满足该问题是凸优化问题，且还要满足KKT条件（Karush–Kuhn–Tucker conditions），所以有：
$$
\min_{w, b, \xi} \max_{\alpha,\beta} L(w,b,\xi,\alpha, \beta) = \max_{\alpha,\beta} \min_{w, b, \xi} L(w,b,\xi,\alpha, \beta)
$$
为了保证原问题与对偶问题求得的最优解相同，需要满足以下约束：
$$
\begin{align}
& ①\quad \alpha_i \geq 0, \quad \beta_i \geq 0, & (乘子约束)\\
& ②\quad \xi_i \geq 0, \quad y_i(w^Tx_i+b)-1+\xi_i \geq 0, & (原始约束)\\
& ③\quad \alpha_i[y_i(w^Tx_i+b)-1+\xi_i] = 0, \quad \beta_i\xi_i=0, & (KKT条件)\\
& ④\quad \frac{\partial L}{\partial w} = \frac{\partial L}{\partial b} = \frac{\partial L}{\partial \xi}=0. & (KKT条件)\\
\end{align}
$$
将最后的关于偏导的KKT条件计算得到：
$$
\begin{align}
& \frac{\partial L}{\partial w} = w-\sum_{i=1}^{n}\alpha_iy_ix_i=0 &\Leftrightarrow 
\quad w=\sum_{i=1}^{n}\alpha_iy_ix_i \\
& \frac{\partial L}{\partial b} = \sum_{i=1}^{n}\alpha_iy_i=0 &\Leftrightarrow 
\quad \sum_{i=1}^{n}\alpha_iy_i=0\\
& \frac{\partial L}{\partial \xi} = C-\alpha_i-\beta_i=0 &\Leftrightarrow 
\quad \alpha_i+\beta_i=C
\end{align}
$$
将上面得到的结果代入到拉格朗日函数中可得：
$$
\begin{align}
\max_{\alpha,\beta} L(w,b,\xi,\alpha, \beta) 
& = \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i - \sum_{i=1}^{n}\alpha_i[y_i(w^Tx_i+b)-1+\xi_i]-\sum_{i=1}^{n}\beta_i\xi_i = \\
\max_{\alpha,\beta} L(\alpha, \beta) & = \frac{1}{2}\|\sum_{i=1}^{n}\alpha_iy_ix_i\|^2 + \sum_{i=1}^{n} (C-\alpha_i-\beta_i)\xi_i - \sum_{i=1}^{n}\alpha_iy_i(\sum_{j=1}^{n}\alpha_jy_jx_j)x_i-b\sum_{i=1}^{n}\alpha_iy_i+\sum_{i=1}^{n}\alpha_i\\
& = -\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_j(\vec{x_i}^T\cdot\vec{x_j})+\sum_{i=1}^{n}\alpha_i\\
\text{s.t.} & \quad \alpha_i \geq 0, \quad \beta_i \geq 0, \quad \alpha_i+\beta_i=C, \quad \sum_{i=1}^{n}\alpha_iy_i=0
\end{align}
$$
由于上式约束中的$\beta_i$除了$\beta_i \geq 0$之外没有其他约束，所以$\alpha_i+\beta_i=C$可以转化为$\alpha_i \leq C$。将约束进行转化后有：
$$
\begin{align}
\max_{\alpha}L(\alpha) 
& = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_j(\vec{x_i}^T\cdot\vec{x_j})\\
\text{s.t.} & \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n}\alpha_iy_i=0
\end{align}
$$

## 算法求解准备

若要求解该最优化函数，只需要在不违反约束的情况下，想办法改变$\alpha$的值，然后逐渐让该函数最大化，考虑到存在约束：
$$
\sum_{i=1}^{n}\alpha_iy_i=0, \quad y_i \in \{-1,1\}
$$
为了保证该约束可以一直成立，可以考虑只同时改变两个$\alpha$的值，然后在保证不违反约束的情况下，使目标函数最大化，该思路就是由John Platt于1996年提出的称为SMO（Sequential Minimal Optimization）的算法。SMO算法是将一个大的优化问题分解为多个小的优化问题来求解的。这些小的优化问题往往很容易求解，并且对它们进行顺序求解的结果与将它们作为整体来求解的结果是完全一致的。而在正式使用SMO算法之前，还需要先讨论在什么情况下是违反另一个约束的，以方便SMO算法优化求解。

考虑到还存在另一个约束：
$$
0 \leq \alpha_i \leq C
$$
我们将该约束分解成下面三种情况进行讨论，先看在满足什么情况下满足约束，就可以知道什么情况下违反约束了
$$
\begin{cases}
\alpha_i = 0\\
0 < \alpha_i < C\\
\alpha_i = C
\end{cases}
$$
另外，为了方便讨论各个情况，我们使用的是还未代入的各个约束，并且为了方便表示，这里定义特征到结果的输出函数为$f(x)=w^Tx+b$
$$
\begin{align}
& ①\quad \alpha_i \geq 0, \quad \beta_i \geq 0, & (乘子约束)\\
& ②\quad \xi_i \geq 0, \quad y_if(x_i)-1+\xi_i \geq 0, & (原始约束)\\
& ③\quad \alpha_i[y_if(x_i)-1+\xi_i] = 0, \quad \beta_i\xi_i=0, & (KKT条件)\\
& ④\quad w=\sum_{i=1}^{n}\alpha_iy_ix_i, \quad \sum_{i=1}^{n}\alpha_iy_i=0, \quad \alpha_i+\beta_i=C & (KKT条件)\\
\end{align}
$$
接下来讨论约束的三种情况：

第一种情况是$\alpha_i = 0$，而在KKT条件中有$\alpha_i+\beta_i=C$，所以有：$\beta_i=C>0$，而在KKT条件中有$\beta_i\xi_i=0$，可知$\beta_i$和$\xi_i$必有一个为0，而$\beta_i=C>0$，所以$\xi_i=0$.  而在原始约束中有$y_if(x_i)-1+\xi_i \geq 0$，即$y_if(x_i) \geq 1-\xi_i$，而$\xi_i=0$，所以有：$y_if(x_i) \geq 1$.

第二种情况是$0 < \alpha_i < C$，而在KKT条件中有$\alpha_i+\beta_i=C$，所以有：$0<\beta_i<C$，而在KKT条件中有$\beta_i\xi_i=0$，可知$\beta_i$和$\xi_i$必有一个为0，而$\beta_i>0$，所以$\xi_i=0$.  在KKT条件中还有$\alpha_i[y_if(x_i)-1+\xi_i] = 0$，也就是说$\alpha_i$和$y_if(x_i)-1+\xi_i$必有一个为0，而$0 < \alpha_i < C$，所以有$y_if(x_i)-1+\xi_i=0$，而$\xi_i=0$，所以有：$y_if(x_i) = 1$.

第三种情况是$\alpha_i = C$，而在KKT条件中有$\alpha_i+\beta_i=C$，所以有：$\beta_i=0$，而在KKT条件中有$\beta_i\xi_i=0$，可知$\beta_i$和$\xi_i$必有一个为0，而$\beta_i=0$，所以$\xi_i>0$.  在KKT条件中还有$\alpha_i[y_if(x_i)-1+\xi_i] = 0$，也就是说$\alpha_i$和$y_if(x_i)-1+\xi_i$必有一个为0，而$\alpha_i = C$，所以有$y_if(x_i)-1+\xi_i=0$，即$y_if(x_i) = 1-\xi_i$，而$\xi_i>0$，所以有：$y_if(x_i) \leq 1$.

综合上面三种情况，每种情况的约束需要满足的条件与具体含义为：
$$
\begin{align}
\begin{cases}
\alpha_i = 0 &\Leftrightarrow \quad y_if(x_i) \geq 1 \quad(样本在间隔超平面以外)\\
0 < \alpha_i < C &\Leftrightarrow \quad y_if(x_i) = 1 \quad(样本在间隔超平面上)\\
\alpha_i = C &\Leftrightarrow \quad y_if(x_i) \leq 1 \quad(样本在间隔超平面以外)
\end{cases}
\end{align}
$$
所以我们可以得到在下面两种种情况下，是违反约束的：
$$
\begin{cases}
\alpha_i > 0, \quad y_if(x_i) > 1\\
\alpha_i < C, \quad y_if(x_i) < 1 \\
\end{cases}
$$
所以我们只需要将这些违反约束的$\alpha$找出来进行调整修改，使其满足约束即可。

## SMO算法求解

### 确定$\alpha$值范围

在前面的介绍中可知，为了满足约束$\sum_{i=1}^{n}\alpha_iy_i=0, \quad y_i \in \{-1,1\}$，SMO算法的求解思路时同时改变一对$\alpha$的值，对函数进行最优化，这里假设要修改的$\alpha$为$\alpha_1$和$\alpha_2$，修改前为$\alpha_1^{old}$和$\alpha_2^{old}$，修改后为$\alpha_1^{new}$和$\alpha_2^{new}$，为保证约束，修改前和修改后的$\alpha$需要满足：
$$
\alpha_1^{old}y_1 + \alpha_2^{old}y_2 = \alpha_1^{new}y_1 + \alpha_2^{new}y_2=\zeta
$$
其中$\zeta$是一个常数，剩余的$\sum_{i=3}^{n}\alpha_i=-\zeta$，由于不好同时求解$\alpha_1$和$\alpha_2$，所以可先求其中一个，另一个可以根据约束，通过求解第一个的变化情况得到，这里是先对$\alpha_2$进行求解，而由于$0 \leq \alpha_i \leq C$，所以在改变$\alpha$值时，为保证约束，要将新得到的$\alpha$值的范围进行裁剪，并且考虑到$y_i \in \{-1,1\}$，所以下面需要讨论两种情况：

第一种情况：$y_1 \neq y_2$，由于$y_1 \neq y_2$，所以有：
$$
\alpha_1^{old} - \alpha_2^{old} = \alpha_1^{new} - \alpha_2^{new}=\zeta
$$
而由于$0 \leq \alpha_i \leq C$，所以有：
$$
0 \leq \alpha_1^{new} \leq C, \quad 0 \leq \alpha_2^{new} \leq C
$$
而$\alpha_1^{new} = \alpha_2^{new} + \zeta$，所以有
$$
0 \leq \alpha_2^{new} + \zeta \leq C, \quad 0 \leq \alpha_2^{new} \leq C
$$
即：
$$
-\zeta \leq \alpha_2^{new} \leq C - \zeta, \quad 0 \leq \alpha_2^{new} \leq C
$$
所以有：
$$
max(0, -\zeta) \leq \alpha_2^{new} \leq min(C, C-\zeta)
$$
而因为：
$$
\alpha_1^{old} - \alpha_2^{old} = \alpha_1^{new} - \alpha_2^{new}=\zeta
$$
所以有：
$$
max(0, \alpha_2^{old} - \alpha_1^{old}) \leq \alpha_2^{new} \leq min(C, C+\alpha_2^{old} - \alpha_1^{old})
$$
第二种情况：$y_1 = y_2$，由于$y_1 = y_2$，所以有：
$$
\alpha_1^{old} + \alpha_2^{old} = \alpha_1^{new} + \alpha_2^{new}=\zeta
$$
而由于$0 \leq \alpha_i \leq C$，所以有：
$$
0 \leq \alpha_1^{new} \leq C, \quad 0 \leq \alpha_2^{new} \leq C
$$
而$\alpha_1^{new} = \zeta - \alpha_2^{new}$，所以有
$$
0 \leq \zeta - \alpha_2^{new} \leq C, \quad 0 \leq \alpha_2^{new} \leq C
$$
即：
$$
\zeta-C \leq \alpha_2^{new} \leq \zeta, \quad 0 \leq \alpha_2^{new} \leq C
$$
所以有：
$$
max(0, \zeta-C) \leq \alpha_2^{new} \leq min(C, \zeta)
$$
而因为：
$$
\alpha_1^{old} + \alpha_2^{old} = \alpha_1^{new} + \alpha_2^{new}=\zeta
$$
所以有：
$$
max(0, \alpha_2^{old} + \alpha_1^{old} - C) \leq \alpha_2^{new} \leq min(C, \alpha_2^{old} + \alpha_1^{old})
$$
综上，$\alpha_2^{new}$的上下界为：
$$
L\leq \alpha_2^{new} \leq H  \Leftrightarrow 
\begin{align}
\begin{cases}
L=max(0, \alpha_2^{old} - \alpha_1^{old}), H=min(C, C+\alpha_2^{old} - \alpha_1^{old}), & y_1 \neq y_2\\
L=max(0, \alpha_2^{old} + \alpha_1^{old} - C), H=min(C, \alpha_2^{old} + \alpha_1^{old}), & y_1 = y_2
\end{cases}
\end{align}
$$

### 更新$\alpha$值

得到$\alpha_2^{new}$的上下界后，接下来就是要得到$\alpha_2^{new}$的值，然后利用得到的上下界进行裁剪。而我们之前得到的需要优化的函数为：
$$
\begin{align}
\max_{\alpha}L(\alpha) 
& = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_j(\vec{x_i}^T\cdot\vec{x_j})\\
\text{s.t.} & \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n}\alpha_iy_i=0
\end{align}
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


所以推广到一般形式有：
$$
\begin{align}
\max_{\alpha}L(\alpha) 
& = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_jK_{ij}\\
\text{s.t.} & \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n}\alpha_iy_i=0
\end{align}
$$
由于我们求解时只修改$\alpha_1$和$\alpha_2$，将与$\alpha_1$和$\alpha_2$有关的项提出有：
$$
\begin{align}
L(\alpha) 
& = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_jK_{ij}\\
& = \alpha_1 + \alpha_2 + \sum_{i=3}^{n}\alpha_i -
\frac{1}{2}\sum_{i=1}^{n}[\sum_{j=1}^{2}y_iy_j\alpha_i\alpha_jK_{ij}+
\sum_{j=3}^{n}y_iy_j\alpha_i\alpha_jK_{ij}]\\
& = \alpha_1 + \alpha_2 + \sum_{i=3}^{n}\alpha_i - 
\frac{1}{2}\sum_{i=1}^{2}[\sum_{j=1}^{2}y_iy_j\alpha_i\alpha_jK_{ij}+
\sum_{j=3}^{n}y_iy_j\alpha_i\alpha_jK_{ij}] - 
\frac{1}{2}\sum_{i=3}^{n}[\sum_{j=1}^{2}y_iy_j\alpha_i\alpha_jK_{ij}+
\sum_{j=3}^{n}y_iy_j\alpha_i\alpha_jK_{ij}]\\
& = \alpha_1 + \alpha_2 + \sum_{i=3}^{n}\alpha_i - 
\frac{1}{2}\sum_{i=1}^{2}\sum_{j=1}^{2}y_iy_j\alpha_i\alpha_jK_{ij} + 
\sum_{i=1}^{2}\sum_{j=3}^{n}y_iy_j\alpha_i\alpha_jK_{ij} - 
\frac{1}{2}\sum_{i=3}^{n}\sum_{j=3}^{n}y_iy_j\alpha_i\alpha_jK_{ij}\\
& = \alpha_1 + \alpha_2 - 
\frac{1}{2}\alpha_1^2K_{11} - \frac{1}{2}\alpha_2^2K_{22} - y_1y_2\alpha_1\alpha_2K_{12}-
y_1\alpha_1\sum_{j=3}^{n}\alpha_jy_jK_{1j}-y_2\alpha_2\sum_{j=3}^{n}\alpha_jy_jK_{2j} +
\sum_{i=3}^{n}\alpha_i -
\frac{1}{2}\sum_{i=3}^{n}\sum_{j=3}^{n}y_iy_j\alpha_i\alpha_jK_{ij}
\end{align}
$$
为方便表示，这里定义：
$$
f(x_i)=\sum_{j=1}^{n}\alpha_jy_jK_{ij}+b\\
v_i=\sum_{j=3}^{n}\alpha_jy_jK_{ij}=f(x_i)-\sum_{j=1}^{2}\alpha_jy_jK_{ij}-b=
f(x_i)-\alpha_1y_1K_{1i}-\alpha_2y_2K_{2i}-b
$$
所以要优化的函数变为：
$$
\max_{\alpha}L(\alpha) = \alpha_1 + \alpha_2 - 
\frac{1}{2}\alpha_1^2K_{11} - \frac{1}{2}\alpha_2^2K_{22} - y_1y_2\alpha_1\alpha_2K_{12}-
y_1\alpha_1v_1-y_2\alpha_2v_2 + Constant(常数项)\\
\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n}\alpha_iy_i=0
$$
其中$Constant$为常数项，因为与$\alpha_1$和$\alpha_2$无关，并且之后求导会直接变为0，这里不进行展开。

接下来将$\alpha_1$用$\alpha_2$表示，由于存在约束$\sum_{i=1}^{n}\alpha_iy_i=0$，所以令$\alpha_1y_1+\alpha_2y_2=\zeta$，两边同时乘以$y_1$有：$\alpha_1y_1y_1+\alpha_2y_2y_1=y_1\zeta$，而$y_1y_1=1$，令$\gamma=y_1\zeta$，$s=y_1y_2$，那么有：
$$
\alpha_1 = \gamma - s \alpha_2
$$
将$\alpha_1 = \gamma - s \alpha_2$及$s=y_1y_2$代入函数得：
$$
L(\alpha_2) = \gamma - s\alpha_2 + \alpha_2 - 
\frac{1}{2}(\gamma - s\alpha_2)^2K_{11} - \frac{1}{2}\alpha_2^2K_{22} - 
s(\gamma - s\alpha_2)\alpha_2K_{12}-
y_1(\gamma - s\alpha_2)v_1-y_2\alpha_2v_2 + Constant(常数项)\\
$$
如此一来要优化的函数中就只剩下$\alpha_2$一个变量了，该函数对$\alpha_2$的偏导数为0时，该函数取得极值，所以下面需要对该函数求偏导（这里注意$s^2=y_1y_1y_2y_2=1$）：
$$
\begin{align}
\frac{\partial L(\alpha_2)}{\partial \alpha_2} & = -s+1+\gamma sK_{11}-\alpha_2K_{11}-\alpha_2K_{22}-
\gamma sK_{12}+2\alpha_2K_{12}+y_2v_1-y_2v_2\\
& = -y_1y_2 + 1 + \gamma y_1y_2K_{11}-\alpha_2K_{11}-\alpha_2K_{22}-\gamma y_1y_2K_{12}+
2\alpha_2K_{12}+y_2v_1-y_2v_2\\
& = 0
\end{align}
$$
所以有：
$$
\alpha_2 = \frac{y_2[y_2-y_1+y_1\gamma(K_{11}-K_{12})+v_1-v_2]}{K_{11} + K_{22} - 2K_{12}}
$$
我们得到了$\alpha_2$的值，之后需要根据这个值更新$\alpha_2$，而根据之前的定义有：
$$
\begin{align}
\gamma & = \alpha_1+ s\alpha_2 \\
& = \alpha_1^{new}+ s\alpha_2^{new} \\
& = \alpha_1^{old}+ s\alpha_2^{old}\\
& =\alpha_1^{new}+ y_1y_2\alpha_2^{new}\\
& =\alpha_1^{old}+ y_1y_2\alpha_2^{old}
\end{align}
$$


如此一来，我们可以将之前的$\alpha^{old}$代入，然后将$v_i=f(x_i)-y_1\alpha_1K_{1i}-y_2\alpha_2K_{2i}-b$也代入，另外为了表示方便，这里令$\eta = K_{11} + K_{22} - 2K_{12}$表示学习速率，另外还要注意$K_{12}=K_{21}$且$y_1y_1=y_2y_2=1$，所以有：
$$
\begin{align}
\alpha_2^{new} & = \frac{y_2[y_2-y_1+y_1\gamma(K_{11}-K_{12})+v_1-v_2]}{K_{11} + K_{22} - 2K_{12}}\\
& = y_2[y_2-y_1+y_1(\alpha_1^{old}+ y_1y_2\alpha_2^{old})(K_{11}-K_{12})+
f(x_1)-y_1\alpha_1^{old}K_{11}-y_2\alpha_2^{old}K_{12}-b-
f(x_2)+y_1\alpha_1^{old}K_{12}+y_2\alpha_2^{old}K_{22}+b]\frac{1}{\eta}\\
& = y_2[y_2-y_1+y_1\alpha_1^{old}K_{11}-y_1\alpha_1^{old}K_{12}+
y_2\alpha_2^{old}K_{11}-y_2\alpha_2^{old}K_{12}+
f(x_1)-y_1\alpha_1^{old}K_{11}-y_2\alpha_2^{old}K_{12}-b-
f(x_2)+y_1\alpha_1^{old}K_{12}+y_2\alpha_2^{old}K_{22}+b]\frac{1}{\eta}\\
& = y_2[(f(x_1)-y_1)-(f(x_2)-y_2)+y_2\alpha_2^{old}K_{11}+y_2\alpha_2^{old}K_{22}-
2 y_2\alpha_2^{old}K_{12}]\frac{1}{\eta}\\
& = y_2[(f(x_1)-y_1)-(f(x_2)-y_2)+y_2\alpha_2^{old}\eta]\frac{1}{\eta}\\
& = \alpha_2^{old} +\frac{y_2[(f(x_1)-y_1)-(f(x_2)-y_2)]}{\eta}
\end{align}
$$
这里令$E_i=f(x_i)-y_i$表示误差项，$\eta = K_{11} + K_{22} - 2K_{12}$表示学习速率，那么最后可得到$\alpha_2$的更新公式为：
$$
\alpha_2^{new} = \alpha_2^{old} +\frac{y_2(E_1-E_2)}{\eta}
$$
在得到新的$\alpha_2^{new}$之后，我们还要使用之前推出的上下界对其进行裁剪，所以裁剪后的结果为：
$$
\alpha_2^{new,clipped}=
\begin{cases}
H, \quad \quad\alpha_2^{new} > H\\
\alpha_2^{new}, \quad L \leq \alpha_2^{new} \leq H\\
L, \quad \quad\alpha_2^{new} < L
\end{cases}
$$
我们得到了$\alpha_2$，接下来需要根据$\alpha_2$求得$\alpha_1$，已知$\gamma = \alpha_1 - s \alpha_2=\alpha_1 - y_1y_2 \alpha_2$，所以有：
$$
\gamma =\alpha_1^{old} - y_1y_2 \alpha_2^{old}=\alpha_1^{new} - y_1y_2 \alpha_2^{new,clipped}
$$
最终得到$\alpha_1^{new}$的值为：
$$
\alpha_1^{new} =\alpha_1^{old} + y_1y_2 (\alpha_2^{old} -\alpha_2^{new,clipped})
$$

### 更新$b$值

在更新了$\alpha$值之后，还需要重新计算并更新阈值$b$的值，因为阈值$b$关系到输出函数$f(x)$和误差项$E$的计算，当$0 \leq \alpha_1^{new} \leq C$时，对应的数据点是支持向量上的点，此时满足：
$$
y_1(w^Tx_1+b)=1
$$
两边同时乘以$y_1$，并将$w=\sum_{i=1}^{n}\alpha_iy_ix_i$代入有：
$$
\sum_{i=1}^{n}\alpha_iy_ix_i \cdot x_1 + b = y_1
$$
将核函数由线性核函数推广到一般形式后有：
$$
\sum_{i=1}^{n}\alpha_iy_iK_{i1} + b = y_1
$$
所以有：
$$
\begin{align}
b_1^{new} & = y_1 - \sum_{i=1}^{n}\alpha_iy_iK_{i1}\\
& = y_1 - \sum_{i=3}^{n}\alpha_iy_iK_{i1} - \alpha_1^{new}y_1K_{11} - \alpha_2^{new}y_2K_{12}
\end{align}
$$
其中前两项为：
$$
\begin{align}
y_1 - \sum_{i=3}^{n}\alpha_iy_iK_{i1} & = y_1 - \sum_{i=3}^{n}\alpha_iy_iK_{i1} + f(x_1) - f(x_1)\\
& = (y_1-f(x_1)) + \alpha_1^{old}y_1K_{11} + \alpha_2^{old}y_2K_{12}+b^{old}\\
& = -E_1 + \alpha_1^{old}y_1K_{11} + \alpha_2^{old}y_2K_{12}+b^{old}
\end{align}
$$
将其代入有：
$$
\begin{align}
b_1^{new} & = y_1 - \sum_{i=3}^{n}\alpha_iy_iK_{i1} - \alpha_1^{new}y_1K_{11} - \alpha_2^{new}y_2K_{12}\\
& = -E_1 + \alpha_1^{old}y_1K_{11} + \alpha_2^{old}y_2K_{12}+b^{old} - \alpha_1^{new}y_1K_{11} - 
\alpha_2^{new}y_2K_{12}\\
& = b^{old} - E_1 - y_1(\alpha_1^{new}-\alpha_1^{old})K_{11} - y_2(\alpha_2^{new}-\alpha_2^{old})K_{12}
\end{align}
$$
同理可得：
$$
b_2^{new} = b^{old} - E_2 - y_1(\alpha_1^{new}-\alpha_1^{old})K_{12} - y_2(\alpha_2^{new}-\alpha_2^{old})K_{22}
$$
从理论上说，当$b_1^{new}$和$b_2^{new}$都有效的时候，两个乘子$\alpha_1^{new}$和$\alpha_2^{new}$对应的点都在间隔超平面上，一定有：
$$
b^{new} = b_1^{new} = b_2^{new}
$$
当都不满足的时候，SMO算法选择$b_1^{new}$和$b_2^{new}$的平均值(中点)作为新的阈值$b^{new}$，所以最终阈值$b^{new}$的取值为：
$$
b^{new} = 
\begin{cases}
b_1^{new}, \quad 0<\alpha_1^{new}<C\\
b_2^{new}, \quad 0<\alpha_2^{new}<C\\
\frac{b_1^{new}+b_2^{new}}2{}, otherwise.
\end{cases}
$$

### 优化步骤总结

梳理SMO算法的具体步骤如下：

1.找出违反约束的乘子$\alpha_i$和$\alpha_j$(违反下面的条件则违反约束)：
$$
\begin{cases}
\alpha_i > 0, \quad y_if(x_i) > 1 \quad \Leftrightarrow \quad y_iE_i > 0\\
\alpha_i < C, \quad y_if(x_i) < 1 \quad \Leftrightarrow \quad y_iE_i < 0
\end{cases}
$$

$$
\begin{cases}
\alpha_j > 0, \quad y_jf(x_j) > 1 \quad \Leftrightarrow \quad y_jE_j > 0\\
\alpha_j < C, \quad y_jf(x_j) < 1 \quad \Leftrightarrow \quad y_jE_j < 0
\end{cases}
$$

（这里为避免重复计算，在具体实现时可以先计算误差，再检查是否违反约束）

2.计算误差$E_i$和$E_j$：
$$
E_i = f(x_i)-y_i=\sum_{j=1}^{n}\alpha_jy_jK_{ij}+b-y_i\\
E_j = f(x_j)-y_j=\sum_{i=1}^{n}\alpha_iy_iK_{ij}+b-y_j
$$
3.计算$\alpha_j$的上下界$L$和$H$：
$$
\begin{cases}
L=max(0, \alpha_j^{old} - \alpha_i^{old}), H=min(C, C+\alpha_j^{old} - \alpha_i^{old}), \quad y_i \neq y_j\\
L=max(0, \alpha_j^{old} + \alpha_i^{old} - C), H=min(C, \alpha_j^{old} + \alpha_i^{old}), \quad y_i = y_j
\end{cases}
$$
4.计算学习速率$\eta$：
$$
\eta = K_{ii} + K_{jj} - 2K_{ij}
$$
5.更新$\alpha_j$：
$$
\alpha_j^{new} = \alpha_j^{old} +\frac{y_j[E_i-E_j]}{\eta}
$$
6.根据上下界裁剪$\alpha_j$：
$$
\alpha_j^{new,clipped}=
\begin{cases}
H, \quad \quad\alpha_j^{new} > H\\
\alpha_j^{new}, \quad L \leq \alpha_j^{new} \leq H\\
L, \quad \quad\alpha_j^{new} < L
\end{cases}
$$
7.更新$\alpha_i$：
$$
\alpha_i^{new} =\alpha_i^{old} + y_iy_j (\alpha_j^{old} -\alpha_j^{new,clipped})
$$
8.更新$b_i$和$b_j$：
$$
b_i^{new} = b^{old} - E_i - y_i(\alpha_i^{new}-\alpha_i^{old})K_{ii} - y_j(\alpha_j^{new}-\alpha_j^{old})K_{ij}\\
b_j^{new} = b^{old} - E_j - y_i(\alpha_i^{new}-\alpha_i^{old})K_{ij} - y_j(\alpha_j^{new}-\alpha_j^{old})K_{jj}
$$
9.更新$b$：
$$
b^{new} = 
\begin{cases}
b_i^{new}, \quad 0<\alpha_i^{new}<C\\
b_j^{new}, \quad 0<\alpha_j^{new}<C\\
\frac{b_i^{new}+b_j^{new}}2{}, otherwise.
\end{cases}
$$
