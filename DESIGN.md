# Design and invariants

- Treat target callbacks as unnormalized potential+gradient pairs. Validate finite scalars/vectors and copy positions before callback invocation.
- Momentum covariance is mass; position velocity is momentum divided by mass. Identity mass is the demo default, not an inferred inverse covariance.
- Leapfrog's standalone output has unflipped momentum; the final sign flip needed for an involutive Metropolis phase-space proposal leaves position and kinetic energy unchanged. Each transition independently refreshes momentum and discards it after deciding.
- Return raw signed energy errors, not a thresholded divergence label. Invalid numerical trajectories are explicitly rejected; initial callback errors fail. No undocumented step-size repair.
- Log acceptance avoids exponentiating huge energy differences. Open-interval uniforms resample the exact zero PRNG outcome.
- Keep repeated rejected states. Warmup is discard-only and cannot be described as adaptation.
- Cache only current potential for random-walk acceptance; HMC deliberately re-evaluates initial potential/gradient each trajectory. One combined callback counter records actual work, including callbacks that raise. No speed claims from operation counts.
- All demo chains have equal retained lengths and distinct fixed seeds. Pooled covariance/occupancy are descriptive. Do not count transitions between chains or mistake fixed start proportions for observed mixing.
- Classical R-hat: W is average within-chain sample variance, B is n times sample variance of chain means, variance estimate=(n−1)/n*W+B/n, R-hat=sqrt(variance estimate/W). No split/rank normalization, no cap at1, null/status on W=0. No ESS/MCSE claim.
- Correlated target rho=.95; separated mixture centers±6 on x with identity component covariance. Exact known moments enable comparisons without fitting a density or using an evaluation model.
- Settings/starts selected for a readable development illustration. Same draws do not mean equal target work; the random-walk arm is not a tuned best-effort baseline.
- The figure shows first200 retained points per Gaussian chain, not an independent sample. Both Gaussian scatter axes are[-4,4]. The separate mixture panel shows all1000 retained x states per chain, axes time0..999/x[-12,12], with outside counts and no cross-chain joins. Full JSON keeps all states/transitions.
- Test invariants and finite-difference checks constrain implementation errors on tested inputs, not proof of reversibility, volume preservation or target invariance under arbitrary finite precision/callbacks.

Source: independently derived Python implementation of the mechanics described by Neal: https://ar5iv.labs.arxiv.org/html/1206.1901 . No copied implementation, modern sampler diagnostics, paper benchmark or performance guarantee.

## Quadratic resonance: stable, accepted and barely moving

A second failure needs no separated modes. On the standard2D Gaussian, identity mass and epsilon=sqrt(2), **four leapfrog steps are an identity map in exact arithmetic**. Refreshing momentum cannot fix a trajectory that returns to its starting position. Floating-point trajectories only move by tiny rounding errors. Linear stability and unit determinant do not imply exploration.

```sh
python -m hamiltonlab quadratic examples/quadratic.json
```

`quadratic_map(precision,mass,step_size,steps,position)` derives and composes the per-coordinate2x2 map for `U(q)=sum(k_i*q_i²/2)`. It returns the full one-step/trajectory maps, determinant, strict linear-stability indicator, Hamiltonian quadratic defect and conditional Gaussian proposal moments. These moments are **before Metropolis acceptance**, not the accepted-chain kernel or a mixing bound. The identity-resonance example is deliberate, not a tuning recommendation.

```text
A = [[1-epsilon²*k/(2*m), epsilon/m],
     [-epsilon*k*(1-epsilon²*k/(4*m)), 1-epsilon²*k/(2*m)]]
(q_new,p_new) = A^L * (q,p)
if first row of A^L is [a,b]:
  E[q_new | q] = a*q; Var[q_new | q] = b²*m
  E[(q_new-q)²], with stationary independent input q,p = (a-1)²/k + b²*m
Hamiltonian defect = transpose(A^L)*diag(k,1/m)*A^L - diag(k,1/m)
```

At epsilon=sqrt(2), k=m=1, one step has diagonal0 and off-diagonals sqrt(2),−1/sqrt(2); its fourth power is identity. The computed four-step map has first row `[1,1.256074e-15]`, conditional proposal variance1.577722e-30 and determinant1. The corresponding one-step proposal variance is2, **not** the target variance1: ignoring Metropolis would change the target. A proposal-map moment is not a post-acceptance guarantee.

Four standard-Gaussian chains use starts(-2,-2),(2,2),(-1,-1),(1,1), seeds71..74 and200 retained transitions without discard, same epsilon=sqrt(2):

| Fixed steps | Retained acceptance range | Largest coordinate span, max over chains | Mean squared accepted jump, range across chains |
|---|---|---|---|
| 1 | .570 to .710 | 6.231696 | 2.231222 to4.027284 |
| 4 | 1.000 to1.000 | 2.81e-14 | 1.29e-30 to1.80e-30 |

Each span/jump is computed within a chain, never across starts. Every state/rejection remains in JSON. Equal transition counts use different work: one step takes2 callback calls per transition, four steps take5, plus each chain's initial callback. No speed claim, sampler ranking or general optimal trajectory-length claim. The demo asserts both meaningful one-step motion and high-acceptance four-step stagnation. Original Gaussian/random-walk/mixture fields and SVG are unchanged.

Nine new tests check map vs actual leapfrog at multiple lengths with unequal precision/mass, energy-defect reconstruction, inverse/negative time, independent seeded conditional moments, determinant, joint precision/mass scaling, one-step variance mismatch, full-period resonance, stable-but-stagnant chains, invalid/nonfinite map inputs and CLI. **29 tests total.** This algebra extends Neal's quadratic/leapfrog exposition; it is independently derived, with no copied code or paper performance claim. The map uses floating-point composition, not symbolic/interval arithmetic, exact-in-computer real arithmetic or a general target theorem. Precision/mass each lie in(0,1e4], step size has nonzero magnitude<=10, steps1..1000; nonfinite composed results fail. Only separable diagonal quadratics are covered. Arbitrary position-dependent Hessians, adaptation, stochastic gradients and accepted-kernel moments are not supplied.
