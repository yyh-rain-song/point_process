# some simplify
## write $\mu$ and $A$ in matrix form
Since
$$
\mu_u=\sum_c\sum_{some\ i}p_{ii}^c/T_c
$$
In this equation, $T_c$ can be processed in advance. Thus, we only consider given $u$ and $i$, calculate $p_{ii}$
$$
\begin{aligned}
    p_{ii}&= \frac{u_u^{(m)}}{u_u^{(m)}+G(u,i)}
\end{aligned}
$$
Where:
$$
\begin{aligned}
G(u,i)&=\sum_{j=1}^{i-1} A_{u,u_j^c}^{(m)}g(t_i^c-t_j^c)
\end{aligned}
$$
For a given sample $c$, we define a variable called $G\_matrix$, which can be processed in advance:
$$
G\_matrix[i,u']=\sum_{j=1,u_j^c=u'}^{i-1}g(t_i^c-t_j^c)
$$
Thus, given $u$ and $i$, we can calculate $p_{ii}$ in one line:
```
# given i, u=u_i^c, sample
sum_func = np.dot(sample.G_matrix[i], A[u])
p_ii = mu[u]/(mu[u]+sum_func)
```
And, given $u, u'$, we want to calculate $C$ as a whole:
$$
\begin{aligned}
C_{u,u'}&=\sum_c\sum_{i:u_i^c=u}\sum_{j:u_j^c=u'}p_{ij}^c\\
&=\sum_c\sum_{i:u_i^c=u}\frac{p_{ii}^c}{\mu_u}\sum_{j:u_j^c=u'}A_{uu'}g(t_i^c-t_j^c)\\
&=\sum_c\sum_{i:u_i^c=u}\frac{p_{ii}^c}{\mu_u}A_{uu'}\sum_{j:u_j^c=u'}g(t_i^c-t_j^c)\\
&=\sum_c\sum_{i:u_i^c=u}\frac{p_{ii}^c}{\mu_u}A_{uu'}\cdot G\_matrix[i,u']
\end{aligned}
$$
Therefore, given $sample, u, u'$, we can calculate $C$ in one only one **for-loop**
```
# given sample, u, u' (u_2 as variable)
C = 0
dim_idx = np.array(sample.dim_list)
wanted_dim_idx = np.where(dim_idx==u)[0]
for i in wanted_dim_idx:
    sum_func = np.dot(sample.G_matrix[i], self.A[u])
    sum_pij = self.A[u,u_2]/(self.mu[u]+sum_func)*sample.G_matrix[i,u]
    C += sum_pij
```
