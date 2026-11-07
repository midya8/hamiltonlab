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

Nine new tests check map vs actual leapfrog at multiple lengths with unequal precision/mass, energy-defect reconstruction, inverse/negative time, independent seeded conditional moments, determinant, joint precision/mass scaling, one-step variance mismatch, full-period resonance, stable-but-stagnant chains, invalid/nonfinite map inputs and CLI. **47 tests total.** This algebra extends Neal's quadratic/leapfrog exposition; it is independently derived, with no copied code or paper performance claim. The map uses floating-point composition, not symbolic/interval arithmetic, exact-in-computer real arithmetic or a general target theorem. Precision/mass each lie in(0,1e4], step size has nonzero magnitude<=10, steps1..1000; nonfinite composed results fail. Only separable diagonal quadratics are covered. Arbitrary position-dependent Hessians, adaptation, stochastic gradients and accepted-kernel moments are not supplied.

## State-independent trajectory-length randomization

`sample(..., steps_range=[low,high])` optionally draws an inclusive discrete-uniform leapfrog length **before refreshing momentum on every transition**, independently of position, momentum, past acceptance and potential values. The range is fixed, not adapted. Each fixed-length Metropolis-corrected proposal is chosen with state-independent weight; this is not NUTS or a stopping rule based on the current trajectory. The sampler still does the ordinary energy accept/reject step and keeps repeated rejected states.

```sh
python -m hamiltonlab sample examples/random_length.json
```

The existing `sample` JSON accepts optional `steps_range` in addition to its required fields. `steps` remains required for schema compatibility but is ignored when a range is supplied. Output `steps` is then null, `steps_range` gives the actual range, each transition records `chosen_steps`, and `length_counts` covers **all** transitions, including discarded warmup and failed/rejected proposals. Actual callback counts include aborted trajectories. The maximum range endpoint drives the preflight work limit, rather than an optimistic average. Random-walk mode rejects this option. Bounds are integer1..1000, low<=high; singleton ranges are allowed. A supplied range consumes an extra random draw even when singleton, so it is not promised to reproduce a fixed-length seed stream. The default absent/null range preserves every previous fixed-setting output and random draw exactly.

For the same standard-Gaussian resonance starts, seed labels, epsilon=sqrt(2) and200 retained transitions, uniform lengths1..4 give:

| Chain seed | Counts L1 / L2 / L3 / L4 | Acceptance | Largest coordinate span | Mean squared accepted jump | Callback calls |
|---|---|---|---|---|---|
| 71 | 40 / 53 / 48 / 59 | .855 | 5.316326 | 4.188289 | 727 |
| 72 | 36 / 56 / 52 / 56 | .845 | 6.454790 | 2.961874 | 729 |
| 73 | 46 / 56 / 54 / 44 | .910 | 5.437958 | 6.141246 | 697 |
| 74 | 46 / 53 / 43 / 58 | .850 | 4.601358 | 3.302637 | 714 |

The four-step fixed arm remains unchanged: acceptance1, max span2.81e-14 and tiny squared jumps. The randomized arm moves on this deliberate resonance example, but these are **descriptive fixture comparisons** with different random streams and target work, not a controlled causal estimate or general improvement claim. Same seed labels do not mean paired momentum draws. Length2 may flip sign almost deterministically and length4 still nearly returns identity; the range includes both. Full JSON keeps the actual selections, acceptances and states rather than claiming every randomized proposal explores.

This option is not a solution to separated modes. A separate200-transition test starting at x=−6 in the original mixture, epsilon.15 and uniform lengths1..10, still makes zero mode switches. No mixing-time, convergence, calibrated-inference, acceptance-rate or efficiency guarantee follows from randomization. It does not adapt mass/step size or remove the need to inspect chains.

Nine added tests check unchanged default outputs, exact independently reproduced RNG consumption, state-independent length sequences from distant starts, inclusive endpoints/histograms, actual work counters, singleton range copying, retained rejection repeats, resonance motion, continued multimodal trapping, invalid ranges/work budgets and CLI. **47 tests total.** All prior demo fields and SVG remain unchanged. Source: independently implemented state-independent length variation discussed in [Neal, Section4, periodicity and trajectory length](https://ar5iv.labs.arxiv.org/html/1206.1901). No reference code copied or paper performance claim.

## Rank-normalized split and folded chain checks

The original classical diagnostic remains unchanged. `rank_split_rhat(chains, include_audit=False)` adds a different check: split every retained chain into two halves; pool draws per coordinate; average tied ranks; map ranks to normal scores with the Blom offset; calculate the between/within ratio on those scores. Repeat after folding draws around the pooled median, then report the maximum of both components. Both component values, within/between variances and statuses are returned, not only the maximum.

```sh
python -m hamiltonlab diagnose examples/chains.json
```

The `diagnose` JSON has exactly `chains`: a list of retained2D chains, not sampler result objects. It returns full intermediate rank/fold/normal-score audit arrays. API callers can omit these arrays. Use **retained samples only**; this function does not infer or remove warmup from a plain list.

```text
r = pooled one-based rank, average ranks for exactly tied finite values
z = NormalInverseCDF((r-3/8)/(S+1/4))
folded_draw = abs(draw - pooled_median)
Rhat_max = max(Rhat(split rank scores), Rhat(split folded rank scores))
```

To avoid hidden draw dropping, this implementation requires equal **even lengths>=4**,2..64 chains and at most100,000 total2D draws. Every retained draw is used and dropped_draws=0 is returned. Odd or unequal chains fail cleanly instead of silently trimming or padding. All values must be finite. Zero within-component variance returns null/status; if **either** component is undefined, the maximum remains null too, even if the other component is computable. Such a result is not a pass. Values below1 are not clipped, no threshold is enforced, and no ESS/MCSE or convergence certificate is supplied.

Existing retained-chain x-coordinate results, without resampling:

| Fixture | Rank split | Folded rank split | Maximum |
|---|---|---|---|
| Correlated Gaussian HMC | 1.007774 | 1.001669 | 1.007774 |
| Correlated Gaussian random walk | 1.742260 | 1.331213 | 1.742260 |
| Separated mixture HMC | 1.733936 | 1.021593 | 1.733936 |

The mixture maximum is smaller than its classical x R-hat6.944902, but still flags a different location distribution across chains; these are different statistics, not evidence of better mixing. Diagnostics do not change samples, settings, acceptance or the previous outputs.

Two deterministic diagnostic counterexamples are included with their complete supplied sequences. They are **not generated MCMC chains or a benchmark**:

- **Scale difference:** four100-draw chains repeat `[-.5,.5,-1.5,1.5]`, two at scale1 and two at scale10 on x. Classical x R-hat0.994987 and rank split0.989949 look low; folded rank split2.117248 exposes different scales. Its y coordinate is identical across chains and is not silently described as failing.
- **Half-chain shift:** four identical100-draw chains contain50 negative draws alternating−2/−1 followed by50 positive draws alternating1/2. Classical R-hat0.994987 does not see between-chain differences, but rank split2.115500 exposes the within-chain shift. Folded component0.989949 stays visible.

A monotone transform preserves the bulk rank component as long as ordering/ties are unchanged. Do not generalize this to the folded component under arbitrary nonlinear transforms: folding around a median changes distances. Exact-tie handling and finite normal quantiles are auditable; this is not a claim that numerical values match every statistical package's edge-case conventions. In particular, null for a degenerate folded component is deliberate and conservative, not a package-compatibility assertion.

Nine added tests cover tie ranks/Blom inversion, all ties/symmetric scores, independent closed-form variance ratio, explicit split/fold ordering, affine/chain-order invariance, nonlinear bulk-rank invariance, no mutation, degenerate components, scale/half-chain counterexamples, limits and CLI. **47 tests total.** All previous sampler/demo fields and the SVG remain unchanged. This is independently implemented from [Vehtari et al., Rank-normalization, folding, and localization, Sections4.1/4.2, equations14/15](https://arxiv.org/pdf/1903.08008). No reference implementation copied, no ESS implementation and no paper simulation/threshold guarantee claimed.
