# HamiltonLab

**A chain can accept almost every proposal and still miss half the target.** Dependency-free Hamiltonian Monte Carlo with explicit leapfrog energy errors, reversible integrator checks, log-domain Metropolis decisions and complete transition histories. Analytic synthetic targets let finite-chain behavior be compared with known moments, without pretending the moments prove convergence.

![Gaussian exploration and separated-mode failure](assets/chain-audit.svg)

## Run offline

```sh
python -m unittest discover -s tests -v
python -m hamiltonlab demo > /tmp/hamilton-demo.json
python examples/render_demo.py /tmp/hamilton-demo.json --check
python -m hamiltonlab sample examples/chain.json > /tmp/chain.json
```

Python3.10+, standard library only. No model/data downloads, account, GPU or paid API. `sample` JSON requires exactly `target`, `initial`, `method`, `draws`, `warmup`, `step_size`, `steps`, `mass`, `seed`. CLI targets are `correlated` and `mixture`; API accepts a callback returning `(potential, gradient)` for a2D position. Potential is negative log unnormalized density; its additive constant does not change decisions. Callback returns energy **and gradient even for random walk**, so reported callback counts are not a wall-time or gradient-efficiency comparison.

## What is implemented

With diagonal positive mass M, momenta are refreshed independently as `p~N(0,M)` and kinetic energy is `sum(p_i²/(2*m_i))`. Leapfrog uses half momentum/full position/half momentum steps, combines interior momentum updates and evaluates the gradient at the initial point and each of L position steps. A final momentum flip makes the phase-space proposal an involution; momentum is discarded, so the sampler only needs the returned position and unchanged kinetic energy. The standalone integrator returns the **unflipped** momentum so reversibility can be tested directly.

```text
p <- p - (epsilon/2)*grad U(q)
repeat L times:
  q <- q + epsilon*p/M
  p <- p - epsilon*grad U(q)   # half step on last iteration
DeltaH = U(q_new)+K(p_new) - U(q_old)-K(p_old)
accept iff log(uniform_open_0_1) < min(0, -DeltaH)
```

Rejected transitions repeat the old state, including after warmup. Dropping these repeats would bias the output. Every transition keeps before/proposal/state, acceptance, signed energy error, log uniform and warmup marker. `ValueError`/overflow from a trajectory is marked invalid and rejected, with null energy error if unavailable, not silently converted into acceptance. Invalid initial target evaluations fail the call. This numerical fallback is not a repair for a wrongly specified callback or discontinuous target.

Settings are fixed. `warmup` only discards early transitions: there is no step-size/mass adaptation, dual averaging, NUTS, tempering or optimizer. Random-walk Metropolis uses symmetric isotropic Gaussian proposals and log-domain acceptance on potential differences. It is a comparison setting, not an intentionally optimized baseline.

## Measured development fixture

Four chains per arm, seeds71..74,1,000 retained plus200 discarded transitions each. Correlated Gaussian covariance is `[[1,.95],[.95,1]]`, mean0. HMC uses epsilon.15,L10,identity mass; random walk uses proposal standard deviation.15. Same transition counts, **different work**. These settings were chosen for a readable development illustration, not an untouched benchmark or a sweep over optimal settings. Equal step parameters are not equal exploration budgets. No sampler ranking or speedup claim follows.

| Correlated Gaussian | HMC | Random walk | Analytic truth |
|---|---|---|---|
| Retained acceptance range | .979 to .987 | .765 to .800 | Not a target moment |
| Pooled mean x / y | .008864 / .018598 | .172653 / .195747 | 0 / 0 |
| Pooled sample variance x / y | 1.029658 / 1.016501 | 1.471826 / 1.499563 | 1 / 1 |
| Pooled sample covariance x,y | .973860 | 1.436712 | .95 |
| Classical unsplit x R-hat | 1.005600 | 1.391580 | Not a guarantee |
| Potential+gradient callbacks, all chains | 52,804 | 4,804 | Includes discarded transitions |

Finite-chain deviations are observations, not posterior error bars. Samples are dependent; pooled sample covariance uses denominator N−1 but is not claimed unbiased for these correlated draws. Four independently seeded starts help expose this fixture's behavior, not establish convergence.

### High acceptance, no mode exploration

The equally weighted mixture has means(-6,0)/(+6,0), identity component covariance, analytic mean0 and covariance diag(37,1). Four HMC chains start two in each component with the same epsilon/L as above. Retained acceptance ranges .997 to1.000, yet **each chain has zero x-sign/mode switches**. Classical x R-hat is6.944902. Pooled positive-x share is exactly.5, pooled mean x≈−.001131 and variance x≈36.420599, close-looking global summaries that conceal disconnected chains. Pooled occupancy was induced by the starts, not learned by mixing. Transitions are counted within chains only, never across concatenation boundaries.

The demo asserts both a reasonable correlated-Gaussian mean check and the separated-mode failure: high acceptance, zero within-chain mode switches and x R-hat>3. It is supposed to fail if either side disappears. It also checks a forward/flip/back leapfrog position residual5.55e-17 on one deterministic input. That finite check is not a proof for all callbacks or long trajectories.

## Diagnostics with their limits

`moments` reports means, N−1 sample covariance, positive-x share and x-sign transitions. `classic_rhat` implements the classical between/within variance ratio for equal-length chains. It is **unsplit and unranked**, has no ESS/MCSE or modern tail diagnostic, and values below1 are not clipped. Zero within-chain variance returns null plus an explicit undefined status, never a fake success. High acceptance, low R-hat, plausible pooled moments or a finite gradient/reversibility check alone cannot prove exploration or calibrated posterior inference.

## Bounds and tests

Fixed2D vectors; finite energies/gradients; diagonal mass in(0,1e4]; positive sampler step size<=10; integrator also accepts negative nonzero step size for reversal; leapfrog steps1..1000; draws2..10000; discarded transitions0..10000; seed0..2³²−1. Planned target work `(draws+warmup)*(L+1)` for HMC or `(draws+warmup)` for random walk must be<=1million, plus the single initial callback. Actual callback calls are counted, including aborted trajectories. Full traces are retained, so memory use grows with transition count. No thinning or hidden independent-sample resampling.

38 tests cover analytic target gradients vs central differences, closed one-step leapfrog, reversibility with nonunit masses, finite-difference4D Jacobian determinant≈1, energy error at decreasing step sizes on a fixed-time quadratic trajectory, log acceptance extremes/ties, deterministic seeds/no mutation, repeated rejection states, counted failed trajectories, random-walk work, seeded Gaussian reference moments, exact small R-hat arithmetic, undefined constant chains and CLI/bound errors. CI tests Python3.10..3.13 and regenerates the offline SVG byte-for-byte.

## Source and scope

Leapfrog and HMC mechanics are independently implemented from [Radford Neal, MCMC using Hamiltonian dynamics, Sections2..4](https://ar5iv.labs.arxiv.org/html/1206.1901), especially equations38..40 and the standard Metropolis energy test. No reference code copied. The classical R-hat variance formula is explicitly written in `diagnostics.py`; no modern rank-normalized diagnostic is claimed. Synthetic target gradients and moments are analytic derivations in this project. No paper benchmark, inference certification, differentiable training, fitted posterior or broad performance theorem is claimed.

MIT license.

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

Nine new tests check map vs actual leapfrog at multiple lengths with unequal precision/mass, energy-defect reconstruction, inverse/negative time, independent seeded conditional moments, determinant, joint precision/mass scaling, one-step variance mismatch, full-period resonance, stable-but-stagnant chains, invalid/nonfinite map inputs and CLI. **38 tests total.** This algebra extends Neal's quadratic/leapfrog exposition; it is independently derived, with no copied code or paper performance claim. The map uses floating-point composition, not symbolic/interval arithmetic, exact-in-computer real arithmetic or a general target theorem. Precision/mass each lie in(0,1e4], step size has nonzero magnitude<=10, steps1..1000; nonfinite composed results fail. Only separable diagonal quadratics are covered. Arbitrary position-dependent Hessians, adaptation, stochastic gradients and accepted-kernel moments are not supplied.

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

Nine added tests check unchanged default outputs, exact independently reproduced RNG consumption, state-independent length sequences from distant starts, inclusive endpoints/histograms, actual work counters, singleton range copying, retained rejection repeats, resonance motion, continued multimodal trapping, invalid ranges/work budgets and CLI. **38 tests total.** All prior demo fields and SVG remain unchanged. Source: independently implemented state-independent length variation discussed in [Neal, Section4, periodicity and trajectory length](https://ar5iv.labs.arxiv.org/html/1206.1901). No reference code copied or paper performance claim.
