# Support Vector Regressor

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
\quad (C-\alpha_i)\xi_i=0, \quad (C-\hat\alpha_i)\hat\xi_i=0, \quad \beta_i=C-\alpha_i, \quad \hat\beta_i=C-\hat\alpha_i$代入得到约束对应的目标函数得到：
$$
\max_{\alpha,\hat\alpha}L(\alpha,\hat\alpha)=-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}(\hat\alpha_i-\alpha_i)(\hat\alpha_j-\alpha_j)(\vec{x_i}^T·\vec{x_j})+\sum_{i=1}^{n}y_i(\hat\alpha_i-\alpha_i)-\epsilon\sum_{i=1}^{n}(\hat\alpha_i+\alpha_i)\\
\text{s.t.} \quad \sum_{i=1}^{n}(\hat\alpha_i-\alpha_i)=0, \quad 0 \leq \alpha_i \leq C, \quad 0 \leq \hat\alpha_i \leq C
$$


