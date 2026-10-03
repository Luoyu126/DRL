# Experimental mapping

For a graph instance `G` and binary solution representation `x`, the experiment uses a conditional reference model `p_theta(x | G)` and an exactly computable task objective.

## MIS

The reward is set cardinality subject to the independent-set constraint. Greedy decoding guarantees feasibility. Local refinement removes one selected node and greedily refills; a restart can use model scores, low-degree scores, random scores, a strongly perturbed model proposal, or an anchored proposal.

## TSP

The reward is negative tour length. The model predicts edge probabilities, which are decoded into a Hamiltonian tour. Local refinement is 2-opt. Restart arms use a perturbed model tour, random-start nearest neighbor, random permutation, noisy nearest neighbor, or angular construction, followed by the same 2-opt kernel.

## Fairness and limitations

Random and UCB use the same arms, objective-query budget and local-search kernel. UCB only changes arm allocation. Exact branch-and-bound MIS and Held–Karp TSP solutions are used solely for evaluation.

The model is a fixed-size MLP rather than a permutation-equivariant GNN. This isolates the exploration/distillation mechanism on CPU but is deliberately not a claim of competitive neural combinatorial optimization performance. TSP in particular exposes the need for structural graph inductive bias and constrained decoding.

