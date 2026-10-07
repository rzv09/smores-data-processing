# SMORES 2026: exploratory correlation analysis of dissolved oxygen

Pier array, FIU Biscayne Bay campus (25.9113, −80.1373). Analysis window: 2026-05-14 00:00 EDT to 2026-06-21 21:30 EDT (3,735 bins of 15 min).
Reproduce with `python eda/correlation_eda.py` (about 7 min). All settings are in the config block at the top of the script. Step 0 inspection: `python eda/step0_inspect.py`.

**How to read this.** **Finding** = measured directly in this dataset, with the table behind it named. **Hypothesis** = a possible explanation that this data cannot confirm.

All significance statements use 1-day moving-block bootstrap 95% CIs (1,000 replicates; 200 for lag CIs). p-values are corrected with an effective sample size following Bretherton et al. (1999), eq. 30: N_eff = N(1 − r₁ₐr₁ᵦ)/(1 + r₁ₐr₁ᵦ). On the raw 15-min series N_eff is only 60–190 of about 3,000 bins.

---

## 1. Data and setup (Step 0 summary)

| Item | What I found / did |
|---|---|
| Source | 13,476 per-chunk CSVs, `SENS_<prefix>_<chunk>_<Kind>.csv`. Chunk ids repeat across prefixes, so chunks are keyed by (prefix, chunk). Loader: [smores_raw.py](smores_raw.py). |
| Clock | Each chunk gets its own linear tick→RTC fit (residual RMS ≤ 0.33 s). 435 RTC-less chunks borrow a neighbouring chunk's fit after a continuity check. 11 chunks (0.5% of reads) can't be placed in time and are excluded. |
| Time zone | **The RTC records EDT.** Evidence: the deployment pressure step is at 20:45 RTC (logged deployment 21:30 EDT), water temperature peaks at 16:00 RTC, and midwater pO2 bottoms out at 04:00–07:00 RTC. The UTC index is RTC + 4 h. |
| Sampling | 52.0 s per sensor (5th–95th percentile 51.8–52.2 s). |
| Coverage | 81% of 15-min bins have data (72% before June 1, 89% after). Gaps are the same on every channel: missing chunk files, mostly May 16 – Jun 3 and Jun 14 – 22. |
| QC (codes 3/5/7 masked, rows kept) | ch13: code 7 on every read from May 15 on. ch7: 0.26% code 3, clustered June 11–14. All other channels: 0%. Two corrupt rows (unknown param or units code) are also masked. |
| Usable channels | **ch2, ch3, ch25 (midwater); ch7, ch26 (upper, 8.3 cm); ch8 (lower, 16.5 cm).** ch13 is dropped. Every other channel sits at a flat floor between −0.69 and +1.12 % sat for the whole deployment (daily SD ≤ 0.04 % sat; [common_00_dead_channel_check.csv](tables/common_00_dead_channel_check.csv)). Exceptions: ch10 is stuck at about 1.1% sat, ch21 has one 2-hour blip on Jun 1, and ch28 (ch26's lower partner) never lives. |
| %sat | Sensor param 21 = 100·pO2 / (0.20946·(760 − Pwv(T))), with Pwv from **Weiss & Price (1980), salinity 0**, to a ratio of 1.0000. The sensor's value is clipped at 0. I recompute %sat from unclipped pO2 with the same formula. |
| Depth labels | The file metadata has no depth information, so the deployment-map labels are used as given. |

Packages added to `.venv`: statsmodels, seaborn, pyarrow, ruptures.

![coverage](figures/common_00_coverage.png)

### Two method decisions that differ from the brief

1. **Deseasonalizing uses a local diurnal climatology.** For each hour of day I take the mean departure from that day's daily mean over a centred 7-day window, and subtract it. Subtracting one whole-record hour-of-day mean instead (the brief's method) injects a fake diurnal cycle into flat anoxic segments. ch7 at its floor before May 22 gets a deseasonalized daily SD of about 55 mmHg from a constant signal, and ch26 and ch8 get a spurious floor of about 2.8 mmHg. The whole-record version is kept as a sensitivity check ([sensitivity_deseason_and_split.csv](tables/sensitivity_deseason_and_split.csv)). Correlations change by at most 0.14 between the two methods, and none changes sign where it matters.
2. **The cross-correlation is run twice.** "deseason" is the as-specified version. "highpass" additionally subtracts a centred 24-h mean. The deseasonalized cross-correlation curves turned out to be nearly flat over ±24 h: shared multi-day level changes dominate them, so their peak lag is meaningless (§5). The high-pass version is the one that can resolve hours-scale leads.

---

## 2. pO2 or %sat as the primary signal?

Every analysis below was run on both signals. Figures exist for each (`figures/po2_*` and `figures/pctsat_*`). The sensor's clipped %sat was also run in statistics-only mode.

![conversion](figures/compare_01_conversion.png)

> **Note.** Over the deployment's 26–33 °C, %sat is pO2 times a factor between 0.651 and 0.660, a ±0.7% swing driven only by vapour pressure. So computed %sat is pO2 rescaled, and every correlation, ACF and lag is the same on both by construction.

![stats](figures/compare_02_stats.png)

> **Note.** Measured on every statistic in this report ([compare_02_stat_differences.csv](tables/compare_02_stat_differences.csv)), computed %sat differs from pO2 by at most 0.009 (Pearson), 0.018 (Spearman), 0.003 (ACF), 0.010 (correlation with temperature) and 0 h (lag). The sensor's clipped %sat is a different case: it changes Spearman by up to **0.51**. Clipping ties the negative anoxic floor of ch7, ch8 and ch26 at exactly 0, which distorts the ranks.
>
> **Recommendation: pO2 is the primary signal.** It is the direct measurement, needs no vapour-pressure or salinity assumption, and isn't clipped. It is also the modelling target already chosen in `SMORES_preprocessing_plan_v2.md` §4. Use %sat (computed from pO2) only for reporting thresholds, such as the 5% and 100% bands and supersaturation. Don't use mg/L: it carries solubility's ~2%/°C temperature dependence, and the sensor's S = 0 setting overstates it in bay water.

Everything below uses **pO2 (mmHg)**. The %sat versions of the figures are in `figures/pctsat_*`.

---

## 3. Overview, distributions and diurnal structure

![overview](figures/po2_01_overview.png)
![zoom](figures/po2_01b_overview_zoom.png)

**Findings**
- The three midwater channels are high and tightly coupled until about May 27. They then fall together, and **ch25 sits near 0 mmHg from about May 29 to June 2**. After that the series show the large daily swings the deployment report describes.
- **ch7 jumps from −0.57 to 149 mmHg across a 2-hour data gap on May 22** (21:28 UTC), with a simultaneous temperature step from 29.10 to 29.78 °C. Before that it is flat at its floor. Afterwards it behaves like a water-column sensor: it supersaturates to 270% sat, peaks at noon EDT, and moves in sync with ch3 (§5).
- **Hypothesis:** the ch7 optode was exposed (scour or disturbance) on May 22, so its "8.3 cm below seafloor" label may not hold after that date. The IMU shows no lander motion at any point (tilt drift < 0.1° over May 20 – Jun 6), so it isn't the frame moving.

![distributions](figures/po2_02_distributions.png)

Share of 15-min bins by saturation band, from [common_02_saturation_bands.csv](tables/common_02_saturation_bands.csv):

| channel | <5% sat | 5–100% | >100% | before Jun 1: >100% | from Jun 1: >100% |
|---|---|---|---|---|---|
| ch2 mid | 1.4% | 93.2% | 5.4% | 13.2% | 0.0% |
| ch3 mid | 1.6% | 89.3% | 9.1% | 21.3% | 0.6% |
| ch25 mid | 8.2% | 77.2% | 14.6% | 14.7% | 14.6% |
| ch7 upper | 24.2% | 58.4% | 17.3% | 1.8% | 28.3% |
| ch26 upper | 92.0% | 8.0% | 0% | 0% | 0% |
| ch8 lower | 92.3% | 7.7% | 0% | 0% | 0% |

![diurnal](figures/po2_03_diurnal.png)

**Finding:** diurnal strength (R² of the hour-of-day mean) rises sharply after June 1:

| channel | before Jun 1 | from Jun 1 |
|---|---|---|
| ch25 | 0.07 | 0.53 |
| ch7 | 0.02 | 0.76 |
| ch2 | 0.03 | 0.21 |
| ch3 | 0.04 | 0.12 |

Midwater peaks at 15:00–16:00 EDT and ch7 at 12:00 EDT. ch26 and ch8 have small afternoon pulses (R² ≤ 0.11). Source: [po2_03_diurnal_strength.csv](tables/po2_03_diurnal_strength.csv).

---

## 4. Correlation structure

![corr full](figures/po2_04_corr_full.png)

Full deployment, Pearson r with bootstrap 95% CI, from [po2_04_05_correlations.csv](tables/po2_04_05_correlations.csv):

| pair | (a) raw | (b) deseasonalized | (c) first difference |
|---|---|---|---|
| ch2–ch3 | 0.81 [0.71, 0.87] | 0.81 [0.72, 0.87] | 0.21 [0.11, 0.34] |
| ch2–ch25 | 0.61 [0.54, 0.70] | 0.65 [0.55, 0.74] | −0.01 [−0.04, 0.04] |
| ch3–ch25 | 0.49 [0.38, 0.62] | 0.54 [0.39, 0.65] | 0.00 [−0.04, 0.04] |
| ch3–ch7 | −0.19 [−0.33, 0.02] | −0.40 [−0.56, −0.13] | **0.19 [0.15, 0.26]** |
| ch2–ch7 | −0.20 [−0.34, 0.03] | −0.41 [−0.59, −0.11] | 0.04 [−0.01, 0.11] |
| ch25–ch26 | 0.21 [0.15, 0.28] | 0.13 [0.03, 0.23] | 0.01 [−0.05, 0.09] |
| any pair involving ch8 | ≤ 0.14 | ≤ 0.05 | ≤ 0.02 |

**How (a), (b) and (c) differ:**
- **(a) Raw** mixes three things: the shared daily light cycle, multi-day level changes, and fast fluctuations.
- **(b) Deseasonalized** removes the average daily shape, yet midwater correlations barely change (0.81 → 0.81, 0.61 → 0.65). So their coupling is *not* mainly the light cycle. It is shared multi-day level variation, such as the late-May decline and recovery. ch7's correlation with ch2 and ch3 becomes *more* negative (−0.4). That comes from opposite multi-day trends: ch7 is at its floor while midwater is high, then the reverse. It is not a mechanistic anti-coupling.
- **(c) First differences** keep only 15-min changes. Almost everything vanishes except **ch2–ch3 (0.21) and ch3–ch7 (0.19)**. These two pairs share fast variability that survives both (b) and (c), and they are the only pairs where that holds. Spearman's higher raw values for ch26 and ch8 (0.24–0.45) come from both series sitting at their floors at the same time.

**Bin size.** Raw and deseasonalized correlations change by at most 0.034 and 0.051 between 15 min and 5 min or 1 h, so those conclusions don't depend on bin size. Differenced correlations do (ch2–ch3: 0.12 at 5 min, 0.21 at 15 min, 0.41 at 1 h). That is expected: a difference at a different step measures a different timescale. Source: [po2_binsize_corr.csv](tables/po2_binsize_corr.csv).

![binsize](figures/po2_binsize_robustness.png)

### Regime split

![changepoint](figures/po2_05a_changepoint.png)

**Changepoint check (Binseg, l2 cost, on the log of the 24-h rolling SD of deseasonalized hourly pO2; [po2_05a_changepoints.csv](tables/po2_05a_changepoints.csv)).**
- The single best break across all six channels is **June 2 15:00 EDT**; for the midwater channels alone it is **June 2 20:00 EDT**.
- With two breaks allowed, both groups add **May 27 04:00 EDT**. ch25 alone breaks on May 27 04:00 EDT.
- %sat gives identical breakpoints.

So the data support a *transition window* of about May 27 – June 3, not a sharp break at June 1. June 1 sits inside it, and I kept it as the boundary you asked for. Correlations recomputed at the midwater break (June 3 00:00 UTC) change by ≤ 0.14 (largest: ch2–ch25 after the break, 0.25 → 0.12) and lead to the same conclusions ([sensitivity_deseason_and_split.csv](tables/sensitivity_deseason_and_split.csv)). Also, the logger stopped rolling over to new sessions at 18:21 EDT on June 2, within hours of the break. Since the IMU shows no motion then, the cause may be a site visit or a logger change. **(hypothesis; worth checking the field log)**

*Limitation:* the 7-day local climatology smears sharp steps by up to ±3.5 days. That is why ch7's single breakpoint lands on May 19 instead of its May 22 jump. Per-channel breakpoints near steps should be read with that tolerance.

![corr pre](figures/po2_05_corr_pre.png)
![corr post](figures/po2_05_corr_post.png)

**Finding: the coupling structure changes.**

| pair (deseasonalized r) | before Jun 1 | from Jun 1 |
|---|---|---|
| ch2–ch3 | 0.92 | 0.08 |
| ch2–ch25 | 0.86 | 0.25 |
| ch3–ch25 | 0.86 | 0.00 |
| ch3–ch8 | 0.10 | 0.35 |
| ch7–ch26 | 0.32 | 0.08 |

The midwater channels go from nearly identical to almost independent. The rolling correlation (§6) shows the drop happens in two steps, around May 29 and around June 6.

---

## 5. Lead/lag (cross-correlation, ±24 h)

![ccf deseason](figures/po2_06_ccf_curves_deseason.png)

**Deseasonalized (as specified).** The curves are almost flat over ±24 h. For ch2–ch3, r is about 0.73 at −24 h, 0.81 at 0 h and about 0.70 at +24 h. Eight of the 15 "peak lags" sit at ±21.5–24 h, and their bootstrap CIs span most of the window. That is the signature of slow, shared multi-day variation, not a lead time ([po2_06_ccf_peaks.csv](tables/po2_06_ccf_peaks.csv), mode = deseason).

![ccf highpass](figures/po2_06_ccf_curves_highpass.png)
![lag heatmap](figures/po2_06_ccf_peak_lag_highpass.png)

**High-pass (sub-daily anomalies only).** Only two pairs have a sharp peak whose bootstrap lag CI is tight:
- **ch2–ch3: 0 h, r = 0.40, lag CI [0, 0]**
- **ch3–ch7: 0 h, r = 0.34, lag CI [0, 0]**

Both are synchronous. This holds at 5-min and 1-h bins too, and in both regimes ([po2_binsize_ccf.csv](tables/po2_binsize_ccf.csv), [po2_06_ccf_peaks_by_regime.csv](tables/po2_06_ccf_peaks_by_regime.csv)).

**Do midwater channels lead the sediment channels? Not detectably.**
- **Into ch8 and ch26:** every midwater pair has a peak |r| of 0.08–0.20, barely outside the ±0.07–0.09 noise band, with lag CIs spanning roughly −20 to +20 h.
  - The one stable-looking value is ch25 → ch8 at +1.25 h (r = 0.15; +1.0 h at both 5 min and 1 h). Its bootstrap CI still runs from −24 to +17 h.
- **Into ch7:** the only robust link is ch3–ch7, and it is synchronous (0 h), so it gives no warning either.

Treat a ~1 h midwater lead on ch8 as a *hypothesis* to test with more data, not a finding.

---

## 6. Autocorrelation and rolling correlation

![acf](figures/po2_07_acf_pacf.png)

ACF at the forecast horizons, 15-min bins, from [po2_07_acf_table.csv](tables/po2_07_acf_table.csv):

| channel | raw 1 h | raw 6 h | raw 12 h | raw 24 h | deseason 1 h | deseason 6 h | deseason 12 h | deseason 24 h | e-folding time (raw) |
|---|---|---|---|---|---|---|---|---|---|
| ch2 | 0.73 | 0.61 | 0.59 | 0.62 | 0.73 | 0.69 | 0.66 | 0.61 | > 48 h |
| ch3 | 0.73 | 0.62 | 0.60 | 0.64 | 0.73 | 0.70 | 0.69 | 0.62 | > 48 h |
| ch25 | 0.72 | 0.27 | 0.05 | 0.61 | 0.63 | 0.55 | 0.48 | 0.45 | 4.75 h |
| ch7 | 0.80 | 0.17 | −0.07 | **0.76** | 0.74 | 0.59 | 0.59 | 0.57 | 4.25 h |
| ch26 | 0.58 | 0.03 | 0.25 | 0.40 | 0.51 | 0.16 | 0.27 | 0.19 | 1.75 h |
| ch8 | 0.70 | 0.16 | 0.13 | 0.16 | 0.69 | 0.25 | 0.16 | 0.03 | 2.75 h |

**Findings**
- ACF drops from about 0.85 at 15 min to about 0.73 at 1 h on every live channel. The PACF is dominated by lag 1 (0.79–0.93), with small contributions out to about 1 h.
- At 1-h bins the lag-1 h ACF rises to 0.61–0.89 (0.84–0.89 on the midwater channels and ch7) ([po2_binsize_acf.csv](tables/po2_binsize_acf.csv)). So a sizeable share of the 15-min variance is fast fluctuation: sensor noise or real sub-hour structure.
- ch2 and ch3 have long memory: the ACF stays around 0.6 out to 48 h, driven by slow level changes.
- ch25 and ch7 are strongly diurnal: their raw ACF returns to 0.61 and 0.76 at 24 h.
- ch26 and ch8 forget their state within a few hours.
- ch7's deseasonalized ACF stays high partly because of its May 22 step.
- **Hypothesis:** ch26's ACF bumps at 12 h and 36 h suggest a semidiurnal (tidal) component. Its correlation with the water-level anomaly is the highest of any channel (0.08–0.10 at about 1 h lag), but still very weak ([po2_09b_tide_ccf.csv](tables/po2_09b_tide_ccf.csv)).

![rolling](figures/po2_08_rolling_corr.png)

Rolling 3-day Pearson correlation on deseasonalized pO2. Pairs: ch7–ch8 (the paired column), ch25–ch7 (both supersaturate), ch2–ch3 and ch2–ch25 (the strongest pairs).

**Findings**
- **ch2–ch3** drops from 0.8–0.9 to about 0.4 on May 29, and to about 0 around June 6.
- **ch2–ch25** goes to about 0 during the May 29 – June 1 collapse, recovers briefly, then is about 0 from June 7.
- **ch7–ch8** and **ch25–ch7** wander between −0.5 and +0.5 with no stable relationship.

No relationship is stable across the deployment.

---

## 7. Temperature

![temperature](figures/common_09_temperature.png)
![temp corr](figures/po2_09_temperature_corr.png)

Source: [po2_09_temperature_corr.csv](tables/po2_09_temperature_corr.csv).

**Findings**
- **Deseasonalized** correlations are negative on most channels (ch3 −0.51 [−0.65, −0.34]; ch2 −0.29; ch26 −0.22). These are confounded: water warmed through the deployment while midwater DO fell.
- **High-pass** removes that trend, and the picture changes:
  - Before June 1, **midwater channels and ch7 are positively coupled with temperature** at sub-daily scales: ch25 0.39 [0.27, 0.52], ch7 0.28 [0.08, 0.46], ch2 0.27 [0.17, 0.32], ch3 0.16 [0.02, 0.31].
  - **ch26 and ch8 show no coupling** (−0.02 to −0.04, CIs include 0).
  - After June 1, all channels are within ±0.15.
- **The sign does not flip between midwater and sediment.** It is positive in the water column and null in the sediment.
- **Hypothesis:** warm, sunny afternoons drive both temperature and photosynthetic O2 in the water column, while buried sensors are decoupled from both.

### All measured features together

![feature corr](figures/po2_11_feature_corr.png)

There are 14 features: pO2 and optode temperature for the six live channels, plus water level (Bar100 pressure) and water temperature from the external Bar100. Table: [po2_11_feature_corr.csv](tables/po2_11_feature_corr.csv). The Bar100 has data in 74% of bins, against 81% for the optodes, because some auxiliary chunk files are missing.

**Findings**
- **Temperatures are nearly one feature.** All optode temperatures and the Bar100 correlate at 0.80–0.98, and at 0.89–0.98 after deseasonalizing. ch8 (deepest) is the least similar. For modelling, one temperature series carries almost all of this information; adding six is redundant.
- **The pO2–temperature block is negative after deseasonalizing**: −0.47 to −0.60 for ch3, and about −0.3 to −0.4 for ch2. This is the deployment-long trend discussed above (water warming while midwater DO fell). The high-pass analysis shows the sub-daily coupling is positive instead.
- **Water level vs pO2 is 0.11–0.27 after deseasonalizing** (ch3 0.27 [0.15, 0.36]). That is higher than the |r| < 0.1 found for the *tidal* anomaly in the tide table. So the association is with slow, multi-day water-level changes (spring–neap or wind setup), not the 12.4-h tide itself. **(hypothesis)**
- Water level and temperature are anti-correlated (−0.36 to −0.40), which also points to a slow shared driver rather than tides.

---

## 8. QC flags

![qc](figures/common_10_qc_heatmap.png)

**Findings**
- Failures don't cluster in time, except on two channels:
  - **ch13:** fails continuously from May 15, so it is dead, not intermittently fouled.
  - **ch7:** fails 1.4–2.1% of reads per day on June 11–14.
- No other channel ever fails. So the late-deployment camera fouling has no counterpart in the optode QC codes.

**Hypothesis:** biofouling on the optodes would show up as drift, not QC flags. Check the June 14–21 trends against the time-lapse images before trusting the last week.

---

## 9. Per-channel summary (pO2)

| | diurnal strength (R², before → after Jun 1) | how fast autocorrelation decays | co-varies with, after deseasonalizing (full period) | lead/lag |
|---|---|---|---|---|
| **ch2** midwater | weak → moderate (0.03 → 0.21); peak 15 EDT | slow; ACF about 0.6 out to 48 h | ch3 0.81, ch25 0.65; with ch7 −0.41 (trend only) | synchronous with ch3; no lead on sediment |
| **ch3** midwater | weak (0.04 → 0.12) | slow; same as ch2 | ch2 0.81, ch25 0.54; with ch8 0.35 after Jun 1 | synchronous with ch2 and ch7; the only channel with fast shared variability with ch7 |
| **ch25** midwater | weak → strong (0.07 → 0.53); peak 16 EDT, amplitude 118 mmHg after Jun 1 | diurnal; raw ACF 0.05 at 12 h, 0.61 at 24 h | ch2 0.65, ch3 0.54, ch26 0.13 | ch25 → ch8 +1.25 h, r = 0.15 (lag CI −24 to +17 h; not robust) |
| **ch7** upper | none (at floor until May 22) → very strong (0.76); peak 12 EDT, amplitude 260 mmHg | strongly diurnal; raw ACF 0.76 at 24 h | anti-correlated with ch2 and ch3 (trend); +0.19 with ch3 on differences | synchronous with ch3 (0 h) |
| **ch26** upper | weak (0.04 → 0.11); afternoon pulses | fast; e-folding 1.75 h; bump at 12 h | ch25 0.13 (weak) | none robust |
| **ch8** lower | weak (0.08 → 0.05) | fast; e-folding 2.75 h | none full-period; ch3 0.35 after Jun 1 | none robust |

---

## 10. Implications for forecasting

**Which horizons look predictable (hypotheses grounded in the ACF and diurnal findings)**
- **1 h:** every channel retains ACF 0.58–0.80 (0.61–0.89 at 1-h bins), so persistence is a strong baseline. A model has to beat it, which is hard at this horizon.
- **6–12 h:**
  - ch2 and ch3 hold ACF about 0.6, but that reflects slow level persistence, not predictable dynamics.
  - ch25 and ch7 depend on knowing the diurnal phase. Their raw ACF goes to about 0 at 12 h, while their diurnal R² after June 1 is 0.53 and 0.76, so a diurnal model should beat persistence clearly.
  - ch26 and ch8 have essentially no memory beyond about 3 h. Expect little skill.
- **24 h:** ch7 (0.76) and ch25 (0.61) are the most forecastable, from the diurnal cycle alone. ch8 and ch26 pulses look unpredictable at 24 h from their own history.

**Exogenous inputs**
- Clock and solar features (hour of day, sun elevation) carry most of the explainable variance for ch25 and ch7. **(finding)**
- No channel *leads* another by a detectable margin, so other channels add contemporaneous information but **no warning time**. ch3 is the best synchronous partner for ch2 and ch7. **(finding)**
- A ~1 h midwater → ch8 lead is a hypothesis worth testing as an input, not something to build on.
- The water-level anomaly correlates at |r| < 0.1 with every channel. **(finding)** This contradicts the 0.48–0.62 in plan §7, which came from a 13.8-hour sample where tide and diurnal cycle are confounded. **(hypothesis)**
- Temperature adds a modest sub-daily signal for midwater channels before June 1 only.

**Regime change**
- The correlation structure, means and diurnal amplitude all change across the late May – early June transition. **(finding)**
- So a model trained only on the first half is being tested out of distribution. I'd recommend either:
  1. a regime-aware split (train/validate inside each regime, report each separately), or
  2. blocked temporal cross-validation whose folds include post-transition data.

  Avoid a single "train on pre, test on post" split unless distribution shift is what you want to measure.
- Standardize per regime, or at least check scaler drift.

**Data quality to fix in preprocessing**
1. **Clock:** key chunks by (prefix, chunk). `src/data_processing/smores/ingest.discover_chunks` keys by chunk only and silently overwrites on this dump. Keep the borrowed-fit and unresolved-chunk bookkeeping.
2. **Time zone:** the RTC is EDT. Convert explicitly (+4 h to UTC) or every diurnal feature is shifted.
3. **Missing chunks:** 28% of bins before June 1 are missing (whole missing files). Ask Harvard whether those files exist.
4. **ch13:** exclude (dead from May 15).
5. **ch7:**
   - Treat the May 22 discontinuity as a sensor-state change, not as biology.
   - Before May 22 it is at its floor; decide whether that counts as "anoxic sediment" or "not yet installed or exposed".
   - Re-check its depth label for after May 22.
6. **Per-channel zero offsets:** the anoxic floor differs by channel. On live channels it is −1.0 (ch26), −0.57 (ch7) and −0.48 mmHg (ch8); across dead channels it spans −1.05 to −0.03, and ch10 is stuck at +1.72. Use per-channel anoxia thresholds, and don't clip.
7. **Signal:** model pO2. Don't use the sensor's clipped %sat or its S = 0 mg/L.
8. **Deseasonalizing:** use a local (rolling) diurnal climatology. A single whole-record climatology creates spurious variance on intermittent channels.
9. **Fouling:** the QC codes won't catch it. Validate June 14–21 against the time-lapse images.
10. **Midwater anoxia May 29 – Jun 2:** check it against field notes or the camera before treating it as real. If real, it is the most important event in the record for an early-warning model. **(hypothesis)**

---

*Produced with `eda/correlation_eda.py`. Figures are in `eda/figures/`, tables in `eda/tables/`, cached grids in `eda/cache/`. The raw data was not modified.*
