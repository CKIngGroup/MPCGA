# MPCGA + HDIC + MTrim

In order to improve the performance of CGA, we propose a three-step procedure. The first step involves the use of a recursive algorithm that utilizes the top few gradients of the loss function to identify potentially important features. This algorithm recursively searches for potentially relevant subpaths within the current model until the number of features reaches K. In each iteration, the selected features are allowed to branch out into their own paths and proceed to the next iteration. This process is repeated in successive iterations. We refer to this procedure as the multipath Chebyshev's Greedy Algorithm.

In the second step, assume that Ω includes up to L features in each step. In this case, Algorithm 2 may
generate at most $`M :=L^K`$  estimation models. Let $`\hat{J}_K^1,...,\hat{J}_K^M`$ represent the sets of selected features for all paths. We apply the HDIC criterion to eliminate redundant features and select the path with the smallest
HDIC value, resulting in refined sets for all paths. Let

$`
\hat{k}_l = \text{argmin}_{1\leq k \leq K} HDIC(\hat{J}_K^l), \text{for } l \text{ in } 1,...,M,
`$

then the we can obtain the refine sets, ̂$`\hat{J}_{\hat{k}_1}^1,...,\hat{J}_{\hat{k}_M}^M`$.

In the third step removes, we remove these sets to obtains a more concise set of paths, a process referred to as model trimming(MTrim). We implement MTrim basing on the objective function and number of features used in each path. Let

$`
J^* = \text{argmin}_{J\in \{\hat{J}_{\hat{k}_1}^1,...,\hat{J}_{\hat{k}_M}^M\}}, \ell_{\text{min}} = \ell_n(\hat{\beta}_{J^*})`$, and $`L_{\text{min}} = |J^*|`$ is the number of features used in the path with the smallest loss. We
consider ℓmin as baseline. If exist m such that

$`
\ell_n(\hat{\beta}_{\hat{J}_{\hat{k}_m}^m}) - \ell_{\text{min}} \leq c_2 * \max(1,L_{\text{min}}-|\hat{J}_{\hat{k}_m}^m|),\text{        (1)}
`$
 
where $`c_2`$ is a tuning parameter, then we remain ̂$`\hat{J}_{\hat{k}_m}^m`$ as one of the final models. The main concept
of MTrim is that we tolerate a larger difference between ℓmin and losses of paths with less features than
the best path, but a fix criterion for those paths with more features than $`J^*`$. In the end, the final set of
MPCGA+HDIC+MTrim is

$`
\{\hat{J}_{\hat{k}_m}; 1\leq m\leq M \text{ and }m \text{ satisfies (1)}\}
`$

# A simple schematic diagram

![image](https://github.com/CKIngGroup/MPCGA/assets/117146718/a37465b6-f750-4d0d-9ae3-6757a5699e0c)

# Algorithm

![image](https://github.com/CKIngGroup/MPCGA/assets/117146718/c590b9a7-5f55-41ca-be3a-f75629ef3902)


# Usage

MPCGA(X,y,Kn,max_set = 3,imp = 0.7,max_split =3)

# Arguments

- X : Input numpy/dataframe/list of n rows and p columns.
- y : Response of length n.
- Kn : The numbert of MPCGA iterations. Kn must be a positive integer between 1 and p.
- max_set : Maximum number of candidate variable for each iteration.
- imp : If a gradient corresponding to x_t is greater than imp*max_gradient then x_t would be consider into candidate set.
- max_split : The maximum iteration for splitting path.

