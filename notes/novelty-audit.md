# Novelty and positioning audit

This note records the closest literatures and the claim language that should be preserved in the submission. It is intentionally more conservative than the introduction.

## 1. Epistemic coordination and common knowledge

The electronic-mail and coordinated-attack literature establishes that deep finite mutual knowledge can differ sharply from common knowledge in coordination problems (Rubinstein 1989; Halpern and Moses 1990; Monderer and Samet 1989; Morris 2002). The paper should **not** claim to discover the value of common knowledge for coordination.

The new object is a design problem: from a finite, costly menu of state features, choose which realizations enter a reliable public channel before an aligned team acts. The equal-marginal-information theorem isolates the value difference between one shared realization and independent private copies inside this design model.

## 2. Team information structures

Classical team theory studies decentralized action under a fixed information structure (Radner 1962; Marschak and Radner 1972; Ho and Chu 1972; Witsenhausen 1971). Nayyar, Mahajan, and Teneketzis (2013) develop the common-information coordinator for dynamic teams.

The closest direct precursor is Summers, Li, and Kamgarpour (2017), who add information links in linear-quadratic-Gaussian teams and show that performance need not have a generic supermodularity property. We should not claim to be the first to optimize a team's information structure or the first to show that generic diminishing returns can fail.

The distinct package here is:

1. the design object is a costed menu of **state features made public**, rather than directed additions to an observation graph;
2. heterogeneous private observations remain, and each selected realization refines every agent's information partition simultaneously;
3. every finite monotone access structure is realized as a coordination-value threshold;
4. public and independent private channels are separated at equal per-agent mutual information;
5. hardness, an exact feature-policy MILP, an arbitrarily bad greedy family, and a positive submodular class are treated in one finite model.

## 3. Value of information and sensor selection

Blackwell comparison, informational substitutes, Gaussian-process sensor placement, and adaptive submodularity provide powerful conditions under which information acquisition can be ordered or optimized (Blackwell 1953; Chen and Waggoner 2016; Krause and Guestrin 2005; Krause, Singh, and Guestrin 2008; Golovin and Krause 2011).

Our mutual-information counterexample is not a claim that mutual information is generally inappropriate. It proves that a statistic of latent-state uncertainty has no universal guarantee for the **optimized value of decentralized joint action**. The operational emergency benchmark strengthens this point by using only features derived from payoff-relevant incident variables.

## 4. Cooperative AI and zero-shot coordination

MACKRL exploits common knowledge already present in a fixed partially observable task. Other-Play and later zero-shot-coordination formalisms study convention compatibility across unfamiliar partners. Noisy ZSC relaxes common knowledge of the underlying game, and Lauffer et al. identify strategically relevant information about collaborators.

BPOD is upstream of these problems: it assumes the game and prior are known and optimizes which exogenous state facts become public before policies are chosen. Avoid claiming that it solves partner uncertainty, unknown-game coordination, or sequential communication.

## 5. Information design and persuasion

Bayesian persuasion and Bayes correlated equilibrium optimize signals for strategic receivers, often under obedience constraints and possibly with externalities. BPOD has aligned agents, no persuasion or obedience constraint, and an interpretable fixed feature menu with sensing/interface costs. The shared theme is information-structure design; the solution concept and objective are different.

## 6. Secret sharing and access structures

Ito, Saito, and Nishizeki (1989) and the subsequent secret-sharing literature show that general monotone access structures can be realized. The XOR construction is therefore **not** a new secret-sharing scheme. The contribution is the reduction from access structures to a two-agent binary-action coordination-value function, which shows that arbitrary higher-order complementarity is present in BPOD.

## Claim language to retain

- “We introduce **Budgeted Public Observation Design**” as a named finite formulation.
- “Every finite monotone access structure is realizable as a coordination-value threshold.”
- “A public channel and independent private copies can have equal per-agent mutual information but different optimal team value.”
- “State mutual information and marginal-value greedy have no general approximation guarantee for BPOD.”
- “Separable revelation yields a monotone-submodular subclass with a tight classical greedy guarantee.”
- “The joint feature-policy MILP is exact for finite instances.”

## Claim language to avoid

- “The first work on information-structure design for teams.”
- “The first demonstration that common knowledge matters for coordination.”
- “Mutual information is always a bad public-feature objective.”
- “The access-structure construction is a new secret-sharing result.”
- “The exact MILP scales to large Dec-POMDPs.”

## Final archival search checklist

Before archival submission, repeat searches for:

- public sensor selection decentralized team;
- common observation design Dec-POMDP;
- shared interface design multi-agent coordination;
- state-feature publicization common-payoff game;
- secret sharing value of information team decision;
- optimal common-information acquisition.

A contemporaneous paper defining the same feature-menu problem should be cited and the formulation-level novelty narrowed. The access-structure theorem, equal-information separation, and exact finite optimization would remain the main differentiators.
