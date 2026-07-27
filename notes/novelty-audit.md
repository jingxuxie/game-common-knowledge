# Novelty and positioning audit

## Closest line 1: information-structure design in teams

Summers, Li, and Kamgarpour (IFAC-PapersOnLine 2017) choose additional **information links** in team problems and show that team performance lacks a general supermodularity property.  This is the closest precursor.

Our paper must not claim that it is the first to optimize a team's information structure or the first to observe failure of a generic diminishing-returns property.  The distinct package is:

1. the design object is a budgeted menu of **state features made public**, rather than directed additions to an observation graph;
2. existing heterogeneous private observations remain in the model, and selected realizations become common knowledge;
3. arbitrary monotone access structures are realized by an explicit XOR-share construction;
4. public and independent private channels are separated at equal per-agent mutual information;
5. complexity, exact joint feature/policy MILP, a positive submodular subclass, and a matching Max-Coverage approximation threshold are developed together.

## Closest line 2: informational substitutes and value of information

Chen and Waggoner (FOCS 2016) provide a broad theory of informational substitutes/complements, identifying submodularity of decision value as the key condition for efficient acquisition.  Our work specializes the value functional to decentralized common-payoff teams, where a public signal changes every agent's information cell simultaneously.  We contribute team-specific separations and algorithms rather than a competing general definition of informational substitutes.

Krause and Guestrin study entropy and decision-theoretic value-of-information selection in graphical models.  Our mutual-information decoy is not a claim that MI is universally inappropriate; it shows that state information can be misaligned with **joint decentralized action value**, even when every operational feature is a statistic of payoff-relevant state variables.

## Closest line 3: common information and common-knowledge MARL

The common-information approach of Nayyar, Mahajan, and Teneketzis reformulates decentralized control under a **fixed** partial-history-sharing structure.  MACKRL learns policies that exploit common knowledge already available to agents.  We instead optimize which state distinctions become common knowledge before decentralized execution.  The present paper is static; dynamic public-interface design is future work.

## Closest line 4: Bayesian persuasion and public signaling

Bayesian persuasion chooses a signaling scheme to influence strategic receivers subject to obedience.  Recent multi-agent persuasion work considers public/private/semi-private signals and externalities.  Our agents have aligned utility, there are no obedience constraints, and the designer chooses from a fixed feature menu under a sensing/interface budget.  The mathematical objects and motivation are therefore different, although both fields design information.

## Closest line 5: secret sharing

The access-structure proof uses a standard XOR secret-sharing idea.  We do not claim a new secret-sharing scheme.  The contribution is the reduction showing that public-observation coordination value can realize every monotone Boolean authorization pattern, which sharpens the structural limitations of generic feature-selection methods.

## Claim language to retain

- “We introduce **Budgeted Public Observation Design**” is defensible as a named finite-team formulation, but avoid claiming the broad first information-design problem for teams.
- “We show that any finite monotone access structure is realizable as a coordination-value threshold.”
- “We give an exact common-versus-private value separation at equal marginal mutual information.”
- “We characterize a separable-revelation subclass with submodular gain.”
- “We give an exact MILP for the finite problem.”

## Remaining search risks

Before archival submission, run a final scholar search for the exact phrases:

- public sensor selection decentralized team;
- common observation design Dec-POMDP;
- communication topology design common-payoff Bayesian game;
- secret sharing value of information team decision;
- public signal selection cooperative game;
- optimal common information acquisition.

A contemporaneous paper that independently defines the same feature-menu problem should be cited and the novelty statement narrowed, but the access-structure theorem and equal-MI separation remain differentiating results.
