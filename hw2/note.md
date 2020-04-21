# some simplify
## calculate p
The $\sum_{k=1}^{i-1}a_{u_i^c u_j^c}^{(m)}g(t_i^c-t_j^c)$ part makes the calculation time-consuming. However, only $C$ needs all $p_{ij}$, and in our **for** loop to calc $C$, $i$ is increasing, and fix $i$, j is also increasing.  
Therefore, let $sum=\sum_{k=1}^{i-1}a_{u_i^c u_j^c}^{(m)}g(t_i^c-t_j^c)$
+ for every i
  + init: $sum=0$, $j=0$
  + every time j increases, add $a_{u_i^c u_j^c}^{(m)}g(t_i^c-t_j^c)$ to $sum$  

This is ***not*** true for $p_{ii}$, so be careful  
Note that it's only right when we increase $i$ or $j$ ***only one*** per loop. I made some small trick to ensure this.  
Modify p_func to return both result and new $sum$.