# MPCGA + HDIC + MTrim

In order to improve the performance of CGA, we propose a three-step procedure. The first step involves the use of a recursive algorithm that utilizes the top few gradients of the loss function to identify potentially important features. This algorithm recursively searches for potentially relevant subpaths within the current model until the number of features reaches K. In each iteration, the selected features are allowed to branch out into their own paths and proceed to the next iteration. This process is repeated in successive iterations. We refer to this procedure as the multipath Chebyshev's Greedy Algorithm.

In the second step, assume that Ω includes up to L features in each step. In this case, Algorithm 2 may
generate at most M := LK estimation models. Let ̂$`\hat{J}_K^1,...,\hat{J}_K^M`$ represent the sets of selected features for all
paths. We apply the HDIC criterion to eliminate redundant features and select the path with the smallest
HDIC value, resulting in refined sets for all paths. Let

then the we can obtain the refine sets, ̂ J1
̂
k1
, ..., ̂ JM
̂
kM
.
We discover that some paths among { ̂J1
̂
k1
, ..., ̂ JM
̂
kM
} exhibit poor performance. In the third step removes,
we remove these sets to obtains a more concise set of paths, a process referred to as model trimming(MTrim).
We implement MTrim basing on the objective function and number of features used in each path. Let
J∗ = arg min
J∈{ ̂ J1
̂
k1
,..., ̂ JM
̂
kM
}
ℓn( ̂ βJ ),
ℓmin = ℓn( ̂ βJ∗ ), and Lmin = |J∗| is the number of features used in the path with the smallest loss. We
consider ℓmin as baseline. If exist m such that
ℓn( ̂ β ̂ Jm
̂
km
) − ℓmin ≤ c2 ∗ max(1, Lmin − | ̂ Jm
̂
km
|), (2)
where c2 is a tuning parameter, then we remain ̂ Jm
̂
km
as one of the final models. The main concept
of MTrim is that we tolerate a larger difference between ℓmin and losses of paths with less features than
the best path, but a fix criterion for those paths with more features than J∗. In the end, the final set of
MPCGA+HDIC+MTrim is
{ ̂ Ĵ km ; 1 ≤ m ≤ M and m satisfies (2)}

![image](https://github.com/CKIngGroup/MPCGA/assets/117146718/a37465b6-f750-4d0d-9ae3-6757a5699e0c)
