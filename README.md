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

47 tests cover analytic target gradients vs central differences, closed one-step leapfrog, reversibility with nonunit masses, finite-difference4D Jacobian determinant≈1, energy error at decreasing step sizes on a fixed-time quadratic trajectory, log acceptance extremes/ties, deterministic seeds/no mutation, repeated rejection states, counted failed trajectories, random-walk work, seeded Gaussian reference moments, exact small R-hat arithmetic, undefined constant chains and CLI/bound errors. CI tests Python3.10..3.13 and regenerates the offline SVG byte-for-byte.

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

Nine new tests check map vs actual leapfrog at multiple lengths with unequal precision/mass, energy-defect reconstruction, inverse/negative time, independent seeded conditional moments, determinant, joint precision/mass scaling, one-step variance mismatch, full-period resonance, stable-but-stagnant chains, invalid/nonfinite map inputs and CLI. **47 tests at the rank-diagnostic increment.** This algebra extends Neal's quadratic/leapfrog exposition; it is independently derived, with no copied code or paper performance claim. The map uses floating-point composition, not symbolic/interval arithmetic, exact-in-computer real arithmetic or a general target theorem. Precision/mass each lie in(0,1e4], step size has nonzero magnitude<=10, steps1..1000; nonfinite composed results fail. Only separable diagonal quadratics are covered. Arbitrary position-dependent Hessians, adaptation, stochastic gradients and accepted-kernel moments are not supplied.

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

Nine added tests check unchanged default outputs, exact independently reproduced RNG consumption, state-independent length sequences from distant starts, inclusive endpoints/histograms, actual work counters, singleton range copying, retained rejection repeats, resonance motion, continued multimodal trapping, invalid ranges/work budgets and CLI. **47 tests at the rank-diagnostic increment.** All prior demo fields and SVG remain unchanged. Source: independently implemented state-independent length variation discussed in [Neal, Section4, periodicity and trajectory length](https://ar5iv.labs.arxiv.org/html/1206.1901). No reference code copied or paper performance claim.

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

Nine added tests cover tie ranks/Blom inversion, all ties/symmetric scores, independent closed-form variance ratio, explicit split/fold ordering, affine/chain-order invariance, nonlinear bulk-rank invariance, no mutation, degenerate components, scale/half-chain counterexamples, limits and CLI. **47 tests at the rank-diagnostic increment.** All previous sampler/demo fields and the SVG remain unchanged. This is independently implemented from [Vehtari et al., Rank-normalization, folding, and localization, Sections4.1/4.2, equations14/15](https://arxiv.org/pdf/1903.08008). No reference implementation copied, no ESS implementation and no paper simulation/threshold guarantee claimed.

## Analytic harmonic flow and mass convention

```sh
python -m hamiltonlab harmonic examples/harmonic.json
```

`harmonic_audit(precision, mass, step_size, steps, position=None)` compares the existing finite leapfrog map with the analytic **unflipped** flow at exactly the same signed duration `T=step_size*steps`. This is a reference for a separable Gaussian `U(q)=sum(k_i*q_i²/2)`, not another sampler or a solution for the mixture/correlated targets. It does not change any sampler draws.

```text
omega = sqrt(k/m), theta = omega*T
[q(T)] = [ cos(theta)       sin(theta)/sqrt(k*m) ] [q(0)]
[p(T)]   [-sqrt(k*m)*sin(theta)   cos(theta)     ] [p(0)]
x = sqrt(k)*q, y = p/sqrt(m)
[x(T),y(T)] = clockwise rotation(theta) * [x(0),y(0)]
```

For independent stationary inputs `q~N(0,1/k), p~N(0,m)`, the exact rotation preserves their joint standardized identity covariance. Its conditional position variance is `sin²(theta)/k`, its stationary lag-one position correlation is `cos(theta)` and its expected squared jump is `((cos(theta)-1)²+sin²(theta))/k`. These are analytic properties of a **hypothetical exact-flow fresh-momentum kernel**, not estimates of the accepted leapfrog chain. Output retains both phase-space maps, their standardized error matrix/Frobenius norm, joint covariance pushforwards, and the leapfrog position variance's relative error **before** Metropolis. That pre-correction covariance is not the accepted chain's invariant covariance.

Mass means **momentum covariance M**, with kinetic energy `p²/(2m)`, not inverse mass. For diagonal precision `[1,100]` and `T=pi/2`:

| Analytic reference | Identity mass `[1,1]` | Precision mass `[1,100]` |
|---|---|---|
| Frequencies | 1,10 | 1,1 |
| Phases | pi/2,5pi | pi/2,pi/2 |
| Stiff coordinate conditional variance | 3.749399e-33 (roundoff near zero) | .01 |
| Stiff coordinate lag-one correlation | -1 | 6.123234e-17 (roundoff near zero) |
| Stiff coordinate standardized numerical-map error | 1.000863 | .001926 |

The last row is measured for the same epsilon `pi/32`,L16; it describes these specified linear systems, not general posterior tuning or speed. Setting `m=k` equalizes the two frequencies under this convention. Setting `m=1/k` instead gives frequencies1,100. No mass estimation or adaptation is performed.

Even perfect integration does not fix a poor duration. At `T=2pi`, precision mass makes the exact map almost identity in floating point; x expected squared jump is5.999039e-32. At `T=pi`, exact x squared jump is4 but conditional variance is1.499760e-32: position only flips sign, preserving its magnitude. Thus **large squared jumps, zero integration error and invariance do not prove exploration**. The supplied full-/half-period examples retain all matrices, not only those summaries.

A fixed-duration `T=1.3` refinement fixture, precision `[1,4]`,identity mass, compares the second coordinate's standardized numerical-map Frobenius error:

| Steps | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| Error | .018919629 | .004712811 | .001177146 | .000294221 |

Doubling steps at fixed duration gives ratios close to4 here, consistent with second-order leapfrog error. This is a deterministic development check, not a universal numerical bound or a benchmark. Ten new tests cover closed dynamics, exact energy/covariance/determinant, composition/inverse time, mass convention, quarter/half/full phases, pre-correction variance, second-order refinement, error matrices, no mutation, limits and CLI. **57 tests at the harmonic-reference increment.** All previous demo fields and SVG remain unchanged; only `harmonic_reference` is added.

Inputs inherit quadratic-map bounds: finite2D position, precision/mass in `(0,1e4]`, absolute step size in `(0,10]`,integer steps1..1000. Absolute harmonic phase must also be<=1e6; nonfinite maps/moments fail instead of becoming misleading JSON. The formulas use ordinary floating point, including roundoff near resonances, and are not certified exact arithmetic. This independent extension derives the diagonal-mass solution from [Neal's Hamilton equations and one-dimensional harmonic example, equations8..13](https://ar5iv.labs.arxiv.org/html/1206.1901); no reference code copied, timing/efficiency claim, adaptive tuning or convergence certificate.

## Finite-window observables: a mean can cancel while magnitude stays stuck

```sh
python -m hamiltonlab window examples/window.json
```

`finite_window_audit(precision, mass, step_size, steps, draws)` evaluates the variance of an average of N consecutive positions and of their squares under the **hypothetical stationary exact-flow Gaussian kernel**. It does not analyze the recorded leapfrog chains. Stationarity includes a random initial position drawn from the Gaussian target; these formulas do not include transient bias from a fixed start.

Full independent Gaussian momentum refresh makes each coordinate an AR(1) in exact dynamics:

```text
q_next = rho*q + sqrt((1-rho²)/k)*z, z~N(0,1), rho=cos(sqrt(k/m)*T)
Cov(q_t,q_(t+h)) = rho^h/k
E(q²)=1/k, Var(q²)=2/k²
Cov(q_t²,q_(t+h)²) = 2*rho^(2h)/k²
F_N(r) = [N + 2*sum_(h=1..N-1) (N-h)*r^h] / N²
Var(mean q) = F_N(rho)/k
Var(mean q²) = 2*F_N(rho²)/k²
```

The squared-position covariance follows directly by conditioning the Gaussian update: `E(q_(t+h)²|q_t)=rho^(2h)*q_t²+(1-rho^(2h))/k` and `E(q_t^4)=3/k²`. This is an analytic derivation for that specified kernel, not a normality test or a Gaussian approximation to another chain.

To avoid cancellation near negative unit correlation, the reported variance uses a nonnegative representation: `a_j=1+rho+...+rho^(j-1)`, `Var(sum Z)=a_N²+(1-rho²)*sum_(j=1..N-1)a_j²` for standardized covariance `rho^|i-j|`. It applies to q²'s covariance too, without claiming that q² has Gaussian innovations. Both this factor and the separate signed lag-sum factor are returned with their roundoff difference. Complete lag correlations, weights, weighted terms, initial coefficient and innovation coefficient sum remain in JSON. No hidden clipping of a negative signed sum is used.

For x precision1, precision mass and N20:

| Exact-flow duration | Variance of mean q | Variance of mean q² | q variance / IID mean variance | q² variance / IID mean variance |
|---|---|---|---|---|
| pi/2 (quarter period) | .05 | .1 | 1 | 1 |
| pi (sign flip) | 0 | 2 | 0 | 20 |
| 2pi (identity) | 1 | 2 | 20 | 20 |
| acos(-.8) | .006775890 | .406179404 | .135517797 | 4.061794036 |

At the sign flip, an even-window position mean cancels exactly but q² never averages away its random initial magnitude. For N21, position-mean variance is1/441=.002267574 while q²-mean variance remains2. Negative correlation can help one observable and hurt another. **Zero variance of one average is not evidence of exploration.** These are finite-window variances, not asymptotic ESS, a CLT, an MCSE estimate from data or an inference guarantee. The ordinary infinite-sum autocorrelation shortcut is not used at the nonergodic endpoints rho=+/-1.

Draw count must be an integer1..10000; other inputs inherit the harmonic/quadratic limits. Nonfinite stationary squared-position variances fail cleanly. Nine added tests compare direct covariance double sums, exact quarter/full/half phases including even/odd windows, every lag/weight/coefficient, negative-correlation observables, a seeded stationary exact-kernel simulation, bounds, no mutation and CLI. **66 tests total.** Only `finite_window_reference` is added to the demo; all previous fields and SVG stay unchanged. The diagonal Gaussian dynamics derive from [Neal's harmonic example and momentum-covariance convention, equations8..13](https://ar5iv.labs.arxiv.org/html/1206.1901); the finite-window covariance derivation is provided here independently, not copied reference code or a paper experiment.
