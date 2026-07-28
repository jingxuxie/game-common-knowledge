# Proof ledger

This document records the complete theorem statements and proof dependencies used by the paper. The LaTeX supplement contains the polished versions.

## Model

A finite state `theta` is drawn from `mu`. Agent `i` observes `o_i(theta)`. Candidate public feature `j` is a deterministic map `phi_j(theta)` with cost `c_j`. The designer selects `S` before the state is drawn, satisfying `sum_{j in S} c_j <= B`. The realization `z_S(theta)` is publicly broadcast, and the observation protocol is common knowledge. Agent `i` chooses `a_i` from a policy depending only on `(o_i(theta), z_S(theta))`.

For common payoff `u(theta,a_1,...,a_n)`, define

```text
V(S) = max_pi E_mu[u(theta, pi_1(o_1,z_S), ..., pi_n(o_n,z_S))],
F(S) = V(S) - V(empty).
```

Stochastic sensors are included by augmenting the state with exogenous sensor noise.

## Lemma 1: deterministic policies suffice

**Statement.** For every fixed public feature set `S`, there is an optimal deterministic decentralized policy profile.

**Proof.** A behavioral policy assigns a simplex of action probabilities to each agent-information cell. Expected utility is multilinear in these finitely many simplices. Fix all cells except one. The objective is linear in that cell, so some extreme point is optimal. Replace that cell by the corresponding deterministic action and continue through all cells. The objective never decreases, and the resulting profile is deterministic. `QED`

## Lemma 2: monotonicity

**Statement.** If `S subset T`, then `V(S) <= V(T)`.

**Proof.** A policy feasible under `S` remains feasible under `T` by ignoring the additional coordinates of `z_T`. `QED`

## Proposition 3: parity destroys submodularity

Let `theta=(b_1,b_2)` be uniform, both agents have no private information, and both receive utility one exactly when they output `b_1 XOR b_2`. Feature `j` reveals `b_j`. Then

```text
V(empty)=V({1})=V({2})=1/2,   V({1,2})=1.
```

Thus the marginal value of feature 2 is zero at the empty set and one half after feature 1; `F` is not submodular.

## Theorem 4: arbitrary monotone access structures

**Statement.** Let `A` be any upward-closed family of subsets of `[m]`, with the empty set unauthorized. There is a two-agent, binary-action public-observation design instance such that

```text
V(S)=1       if S is in A,
V(S)=1/2     otherwise.
```

**Construction.** Let `E_1,...,E_L` be the minimal authorized sets. Draw a secret bit `X` uniformly. Independently for every hyperedge `E_l={j_1,...,j_r}`, draw `r-1` fair bits and assign XOR shares

```text
R_1,...,R_{r-1}, X XOR R_1 XOR ... XOR R_{r-1}
```

to the features in `E_l`. Feature `j` broadcasts every share assigned to `j` across all blocks. Both agents receive utility one iff both output `X`.

If `S` contains a minimal authorized edge, its shares reconstruct `X`. If it contains no minimal edge, every block has a missing fair share; partial views have identical distributions under `X=0` and `X=1`, including jointly across independent blocks. Thus unauthorized sets cannot beat chance. `QED`

## Theorem 5: equal marginal information, unequal coordination value

Let `X` be uniform and let all marginal channels be binary symmetric with crossover `epsilon in [0,1/2]`. One experiment sends the same channel output to all agents; the other sends conditionally independent copies. If the team succeeds only when all `n` agents output `X`, then

```text
V_public = 1-epsilon,
V_private = max{1/2, (1-epsilon)^n},
I(X;Z)=I(X;Y_i)=1-h_2(epsilon).
```

Deterministic sufficiency reduces every local binary rule to constant zero, constant one, identity, or complement. The public identity convention succeeds with `1-epsilon`. In the private case, any constant-containing profile is at most `1/2`, while all nonconstant profiles have success `(1-epsilon)^r epsilon^(n-r)`, maximized when every agent uses identity. `QED`

## Proposition 6: a mutual-information decoy

Draw independent `T~Bernoulli(p)` with `0<p<1/2` and `N~Bernoulli(1/2)`. Both agents must output `T`. The two candidate features reveal `N` and `T`, and the budget is one. Although

```text
I((T,N);N)=1 > h_2(p)=I((T,N);T),
```

the nuisance feature has zero normalized coordination gain while revealing `T` gives perfect coordination.

## Theorem 7: NP-hardness and approximation threshold

Reduce weighted Max `k`-Coverage. Public context `r` has normalized weight `w_r`, and a fair target bit is hidden. Feature `j` reveals the bit in contexts in `C_j` and reports an erasure otherwise. Then

```text
V(S)=1/2 + (1/2) sum_{r in union_{j in S} C_j} w_r.
```

Maximizing normalized gain is exactly weighted Max `k`-Coverage. NP-hardness and the `1-1/e` approximation threshold transfer. `QED`

## Theorem 8: exact joint feature-policy MILP

Use binary variables `x_j`, `y_{i,theta,a}`, and `q_{theta,a_vector}`. The feature-controlled nonanticipativity constraints are

```text
|y_{i,theta,a}-y_{i,theta',a}|
    <= sum_{j: phi_j(theta) != phi_j(theta')} x_j
```

whenever `o_i(theta)=o_i(theta')`. A feasible selected set and deterministic policy define a feasible integer solution. Conversely, when selected public and private observations agree in two states, the right-hand side is zero and all action indicators agree, so every integer solution induces a valid decentralized policy. Deterministic sufficiency completes exactness. `QED`

## Theorem 9: correlated public certificates are submodular

In public context `r`, let `C_r(omega)` be the random set of features whose realized outputs are decision-sufficient public certificates. If `S` intersects this set, the team can implement the context-optimal compatible action; otherwise it receives baseline success `q_r`. The joint law of `C_r(omega)` can be arbitrary.

**Statement.**

```text
F(S)=sum_r w_r(1-q_r) E_omega[1{S intersects C_r(omega)}]
```

is normalized, monotone, and submodular.

**Proof.** Fix `r` and `omega`. The indicator is a coverage function. The marginal of adding `j` is one exactly when `j` belongs to `C_r(omega)` and the current set has not already intersected it. Enlarging the current set can only change this marginal from one to zero, never the reverse. Therefore each outcome-wise function is submodular. Nonnegative weighted sums and expectations preserve submodularity. `QED`

**Algorithmic consequences.** Cardinality-budget marginal-value greedy attains `1-1/e`. For arbitrary positive costs, the monotone-submodular knapsack algorithm of Sviridenko (2004) attains `1-1/e`; this is not a claim about ordinary ratio-greedy.

## Proposition 10: marginal reliability does not identify redundancy

For any `p in (0,1)`, consider two features with singleton certificate probability `p`.

- Perfectly correlated system: both are available together with probability `p`, so pair gain is `p`.
- Independent system: each is available independently with probability `p`, so pair gain is `1-(1-p)^2`.

The singleton gains are identical, but the second feature has zero marginal gain after the first in the correlated system and positive marginal gain in the independent system. Hence per-feature accuracy or singleton value cannot identify redundancy. `QED`

## Corollary 11: independent separable revelation

If sensor `j` independently certifies context `r` with probability `p_{rj}`, then

```text
F(S)=sum_r w_r(1-q_r)[1-product_{j in S}(1-p_{rj})].
```

This is the product-form special case of Theorem 9.

## Theorem 12: marginal-value greedy can be arbitrarily bad

Mix a parity task of probability `1-delta` with a one-feature distractor task of probability `delta`, under budget two. Greedy first selects the distractor because both parity features have zero singleton gain; it can no longer complete the parity pair. Greedy and optimal gains are `delta/2` and `(1-delta)/2`, so their ratio `delta/(1-delta)` tends to zero. `QED`
