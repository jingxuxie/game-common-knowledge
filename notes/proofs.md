# Proof ledger

This document records the complete theorem statements and proof dependencies used by the paper.  The LaTeX supplement contains the polished versions.

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

**Authorized sets.** If `S` contains a minimal authorized edge, its shares XOR to `X`, hence value one is attainable.

**Unauthorized sets.** If `S` contains no minimal edge, then at least one share is absent from every block. In one XOR block, every strict subset of shares is uniform and independent of `X`; this follows by using a missing fair share to biject the completions for `X=0` and `X=1`. The blocks use independent randomness, so their collected partial views are jointly independent of `X`. The best team decision therefore succeeds with probability one half. `QED`

**Consequence.** Public-observation value can exhibit complementarity of any order and any finite monotone Boolean authorization pattern. This is stronger than a single non-submodular counterexample and explains why unrestricted instances admit no universal diminishing-returns argument.

## Theorem 5: equal marginal information, unequal coordination value

Let `X` be a uniform bit and `epsilon in [0,1/2]`.

- In the **public** experiment, every agent sees the same output `Z` of a binary symmetric channel with crossover `epsilon`.
- In the **private** experiment, agent `i` sees an independent output `Y_i` of the same channel, conditionally independent given `X`.

The team receives one iff all `n` agents output `X`.

**Statement.**

```text
V_public = 1-epsilon,
V_private = max{1/2, (1-epsilon)^n},
I(X;Z)=I(X;Y_i)=1-h_2(epsilon) for every i.
```

**Public proof.** All agents use the identity rule and succeed with probability `1-epsilon`. For a binary signal, the only deterministic rules are constant zero, constant one, identity, and complement, with success `1/2,1/2,1-epsilon,epsilon`; deterministic sufficiency proves optimality.

**Private proof.** Every deterministic local rule is one of the same four. If at least one agent is constant, simultaneous correctness occurs for at most one value of `X`, hence with probability at most one half. If no agent is constant, suppose `r` agents use identity and the rest use complement. Conditional on either state, simultaneous correctness has probability `(1-epsilon)^r epsilon^(n-r)`, maximized at `r=n`. The all-constant and all-identity profiles attain the two terms. `QED`

At `epsilon=0.1,n=10`, the values are `0.9` and `0.5`, a common-knowledge premium of `0.4` despite equal information for each individual agent.

## Proposition 6: a mutual-information decoy

Draw independent `T~Bernoulli(p)` with `0<p<1/2` and `N~Bernoulli(1/2)`. Both agents must output `T`; they have no private observations. The two candidate features reveal `N` and `T`, and the budget is one.

```text
I((T,N);N)=1 > h_2(p)=I((T,N);T).
```

Hence an unconditional mutual-information rule selects `N`. It leaves value at the no-feature baseline `1-p`, while publicizing `T` gives value one. The information rule's normalized coordination gain is zero.

## Theorem 7: NP-hardness and approximation threshold

Reduce weighted Max `k`-Coverage. Context `r` has normalized weight `w_r` and is observed by both agents. Independently draw a fair target bit `B`. Feature `j` outputs `B` in contexts belonging to set `C_j` and an erasure otherwise. Both agents must output `B`.

For selected set `S`, covered contexts reveal `B` perfectly and uncovered contexts retain value one half, so

```text
V(S)=1/2 + (1/2) sum_{r in union_{j in S} C_j} w_r.
```

Thus maximizing normalized gain is exactly weighted Max `k`-Coverage. Selection is NP-hard even for two agents, binary actions, deterministic features, unit costs, and common payoff. Feige's threshold transfers: unless `P=NP`, no polynomial algorithm guarantees `1-1/e+eta` for every fixed `eta>0` on normalized gain. `QED`

## Theorem 8: separable revelation is submodular

In context `r` with weight `w_r`, the team succeeds with baseline probability `q_r`. Sensor `j` independently reveals the correct target with probability `p_{rj}`; success is one if at least one selected sensor reveals. Then

```text
F(S)=sum_r w_r(1-q_r)[1-product_{j in S}(1-p_{rj})].
```

The marginal gain is

```text
Delta_j(S)=sum_r w_r(1-q_r) p_{rj} product_{l in S}(1-p_{rl}).
```

It is nonnegative and decreases as `S` grows. Therefore `F` is monotone submodular, and cardinality-budget greedy achieves `1-1/e` of optimal gain. `QED`

## Theorem 9: marginal-value greedy can be arbitrarily bad

With probability `1-delta`, the publicly known context is a parity task with two independent bits and two complementary features. With probability `delta`, it is a one-bit distractor task with a third feature. Budget is two.

The empty-set value is one half. Each parity feature has zero initial marginal gain, while the distractor has gain `delta/2`, so greedy selects it. One remaining feature cannot solve parity, leaving total gain `delta/2`. The optimal pair consists of the parity features and has gain `(1-delta)/2`. Hence

```text
greedy gain / optimal gain = delta/(1-delta) -> 0.
```

## Theorem 10: exact MILP

Use binary variables:

- `x_j`: feature `j` is selected;
- `y_{i,theta,a}`: agent `i` chooses action `a` in state `theta`;
- `q_{theta,a_vector}`: joint action `a_vector` is taken in state `theta`.

Constraints are:

```text
sum_j c_j x_j <= B,
sum_a y_{i,theta,a}=1,
sum_{a_vector} q_{theta,a_vector}=1,
q_{theta,a_vector} <= y_{i,theta,a_i},
|y_{i,theta,a}-y_{i,theta',a}|
    <= sum_{j: phi_j(theta) != phi_j(theta')} x_j
    whenever o_i(theta)=o_i(theta').
```

The objective is `sum_theta mu_theta sum_a u(theta,a) q_{theta,a}`.

**Soundness.** A feasible integer solution induces a valid selected set and deterministic policy: if two states are indistinguishable under the selected public and private signals, the right-hand side of the nonanticipativity constraint is zero, so all action indicators agree. The `q` constraints select the induced joint action.

**Completeness.** Any feasible feature set and deterministic decentralized profile define `x`, `y`, and `q` satisfying every constraint with identical objective. Deterministic sufficiency completes exactness. `QED`
