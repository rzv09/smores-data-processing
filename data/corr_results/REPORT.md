# DO feature correlation analysis

Sampling: `5min`; AR window = 12 steps; horizons (steps) = [1, 6, 12]; CV = leave-one-deployment-piece-out.

**Reading guide.** `mean_gain_pct` = average RMSE change vs the DO-only AR model when this feature is added (negative = helps; this is the main ranking). `r` / `r_diff` = pooled Pearson on levels / first differences. `peak_lag_steps > 0` = feature leads DO. `max_abs_corr_other_feat` high = redundant with another feature.

Caveats: only 4 pieces per site (~18 h each) and a single week; correlations on levels are inflated by the shared diurnal cycle (compare with `r_diff`); the two sites share no common feature except DO and hour-of-day, so cross-site transfer relies on DO itself.


## FL

| feature | mean_gain_pct | r | r_diff | mi | peak_lag_steps | peak_lag_r | sign_consistent | max_abs_corr_other_feat |
|---|---|---|---|---|---|---|---|---|
| Vx | -5.04 | 0.051 | -0.117 | 0.376 | 36 | 0.685 | False | 0.708 |
| hour_cos | -2.706 | 0.197 | 0.129 | 0.426 | -36 | 0.712 | False | 0.708 |
| hour_sin | -2.559 | -0.742 | -0.184 | 0.692 | 21 | -0.851 | True | 0.469 |
| speed_h | -1.76 | 0.309 | -0.0 | 0.289 | -30 | 0.413 | False | 1.0 |
| speed | -1.745 | 0.308 | 0.0 | 0.287 | -30 | 0.413 | False | 1.0 |
| P | -1.112 | -0.483 | -0.106 | 0.544 | 18 | -0.59 | True | 0.702 |
| dP | -0.43 | 0.212 | 0.008 | 0.229 | -33 | 0.551 | True | 0.537 |
| Vy | -0.05 | -0.459 | -0.106 | 0.352 | 7 | -0.491 | True | 0.702 |
| O2_spread | 3.124 | 0.105 | 0.084 | 0.182 | 36 | 0.225 | False | 0.247 |
| Vz | 3.576 | -0.113 | 0.029 | 0.173 | 36 | 0.381 | True | 0.69 |
