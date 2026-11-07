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
