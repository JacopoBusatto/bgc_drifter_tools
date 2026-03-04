# Statistical Analysis Framework

## 1. Conceptual Model

Each MASTER dataset contains time series collected along a Lagrangian trajectory.  
Therefore, every variable \( X(t) \) implicitly depends on:

- time
- spatial position along trajectory
- dynamical regime

The total signal may be decomposed as:

\[
X(t) = X_{bg}(t) + X_{diurnal}(t) + X_{event}(t) + \epsilon(t)
\]

Where:

- \( X_{bg}(t) \): large-scale background variability (low-pass component along trajectory)
- \( X_{diurnal}(t) \): periodic diurnal signal
- \( X_{event}(t) \): transient dynamics (fronts, stirring, wind bursts)
- \( \epsilon(t) \): residual noise

The analysis pipeline is designed to progressively isolate these components and study their coupling.

---

# 2. Decomposition Module

## 2.1 Lagrangian Low-Pass Background

A Lagrangian low-pass filter (rolling mean or spline smoothing) estimates:

\[
X_{bg}(t) = LP_W[X(t)]
\]

Where \( W \) is a multi-day window (e.g., 48–96 hours).

Important:

This does **not** imply a purely temporal trend in an Eulerian sense.  
It represents a large-scale along-track variability that may reflect spatial gradients encountered by the drifting platform.

---

## 2.2 Diurnal Cycle Removal

If sampling permits, hourly climatology is computed:

\[
X_{diurnal}(h) = \mathbb{E}[X(t) \mid hour(t)=h]
\]

An anomaly is then defined as:

\[
X'(t) = X(t) - X_{diurnal}(t)
\]

This is particularly relevant for:

- temperature
- oxygen
- chlorophyll
- bio-optical variables

---

## 2.3 Residual Signal

The event-scale residual component is:

\[
X_{res}(t) = X(t) - X_{bg}(t) - X_{diurnal}(t)
\]

This component is typically used for coupling analysis and regime detection.

---

# 3. Pairwise Coupling Analysis

## 3.1 Pearson Correlation

\[
\rho_{XY} =
\frac{\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]}
{\sigma_X \sigma_Y}
\]

Computed on:

- raw values
- anomaly
- residual
- temporal derivatives

Because Lagrangian signals are autocorrelated, effective degrees of freedom are adjusted in significance testing.

---

## 3.2 Instantaneous Coupling Index

The local contribution to correlation is:

\[
c(t) = z_X(t) \cdot z_Y(t)
\]

Where:

\[
z_X(t) = \frac{X(t)-\mu_X}{\sigma_X}
\]

A rolling average of \( c(t) \) reveals temporally localized coupling strength.

This metric is useful for mapping where and when two variables co-vary along the trajectory.

---

## 3.3 Lagged Correlation

Lagged correlation identifies delayed responses:

\[
\rho(\tau) = corr(X(t), Y(t+\tau))
\]

This is particularly useful for:

- wind → oxygen response
- wind → chlorophyll response
- strain/vorticity → bio-optical response

Lag windows are typically set between ±24–72 hours depending on sampling frequency.

---

# 4. Mutual Information (Non-linear Dependence)

Pearson correlation detects linear relationships.

Mutual Information (MI) measures general statistical dependence:

\[
MI(X,Y) = \iint p(x,y)\log
\frac{p(x,y)}{p(x)p(y)} dxdy
\]

Properties:

- MI = 0 if and only if X and Y are independent
- Detects threshold and nonlinear relationships
- Requires permutation-based significance testing

MI is used as a screening tool to detect non-linear bio-physical coupling that may not be visible in correlation analysis.

---

# 5. Multivariate Structure

## 5.1 Principal Component Analysis (PCA)

Given standardized variables arranged in matrix \( \mathbf{X} \) (time × variables):

PCA finds orthogonal directions \( \mathbf{w}_k \) that maximize variance:

\[
\mathbf{w}_1 = \arg\max Var(\mathbf{X}\mathbf{w})
\]

Outputs:

- Loadings (variable weights for each component)
- Scores (time evolution of modes)
- Explained variance spectrum

Interpretation:

- Loadings identify which variables co-vary.
- Scores identify when specific dynamical modes dominate.

---

## 5.2 Canonical Correlation Analysis (CCA)

Given two variable blocks:

- \( \mathbf{X} \): physical variables
- \( \mathbf{Y} \): biological variables

CCA finds linear combinations:

\[
U = a^T \mathbf{X}
\]
\[
V = b^T \mathbf{Y}
\]

Maximizing:

\[
corr(U,V)
\]

Interpretation:

- Identifies strongest linear coupling between physical and biological variability.
- Provides a physically interpretable bio–physical coupling mode.

---

## 5.3 Partial Least Squares (PLS)

PLS maximizes covariance between predictors and responses while focusing on predictive skill.

Useful when:

- physical predictors are collinear
- objective is explanatory or predictive modeling
- regression robustness is required

Outputs include:

- latent components
- regression coefficients
- explained variance in biological variables

---

# 6. Regime Detection

## 6.1 Clustering

Clustering applied to physical features identifies dynamical regimes such as:

- low-wind calm conditions
- high-wind mixing conditions
- high-strain stirring conditions
- eddy-like retention regimes

Biological variables are then compared between regimes.

---

## 6.2 Change-Point Detection

Change-point algorithms detect time intervals where statistical properties shift.

Applied to:

- PC scores
- T–S properties
- strain or Okubo-Weiss

Allows segmentation into dynamically coherent periods.

---

# 7. Significance Testing

Due to autocorrelation in Lagrangian time series:

- Effective degrees of freedom are reduced.
- Block bootstrap or block permutation is recommended.
- Confidence intervals should be computed using resampling techniques.

---

# 8. Summary of Analysis Workflow

1. Preprocess and decompose signal.
2. Compute pairwise coupling (correlation, lag, instantaneous index).
3. Screen nonlinear relationships (Mutual Information).
4. Explore multivariate structure (PCA).
5. Quantify bio–physical coupling (CCA / PLS).
6. Detect dynamical regimes (clustering / change-point).
7. Assess statistical significance with resampling methods.

This framework provides a statistically rigorous and physically interpretable approach to multi-sensor Lagrangian analysis.




## Quality Control (QC) and Data Cleaning

MASTER datasets are Lagrangian time series sampled along a drifting platform trajectory.
As a consequence, raw signals may contain:

- irregular sampling (nominal 3-hourly, sometimes higher frequency)
- gaps (missing packets, sensor downtime)
- numerical spikes (e.g., curvature/vorticity/strain)
- physically impossible values due to sensor malfunction (e.g., salinity = 15 PSU)
- sensor “flatline” (stuck values)

Because downstream statistical analyses (correlation, PCA/CCA/PLS, clustering) are sensitive to outliers and missing data patterns, a dedicated QC step is required before any analysis.

### QC Philosophy

1. **No interpolation by default.**
   - The analysis operates on irregular time series using true timestamps and time-based rolling windows.
   - Interpolation can be enabled later for specific studies, but is not used in the standard pipeline.

2. **Hard physical bounds are enforced.**
   - Values outside physically plausible ranges are treated as non-physical sensor malfunctions.
   - These points are **masked (set to NaN)**, not clipped.

3. **Spike detection is performed on time-derivatives.**
   - Abrupt jumps over small \(\Delta t\) are detected via \(\left|\frac{dX}{dt}\right|\).
   - This is particularly useful for numerical kinematic features (curvature, vorticity, strain).
   - Detected spikes may be masked or only flagged depending on configuration.

4. **Sensor health diagnostics are reported.**
   - Flatline detection flags sensors that show near-zero variability for extended windows.
   - Coverage metrics and overlap requirements are computed for each variable pair.

5. **Everything is logged.**
   - QC never silently alters data: the pipeline produces a QC report summarizing all masks and flags.
   - Optionally, boolean flag columns (e.g., `qc_out_of_bounds_salinity_psu`) are appended to the dataset.

---

## Time Axis Checks (Irregular time supported)

The QC module computes \(\Delta t\) statistics:

- median \(\Delta t\)
- distribution of \(\Delta t\)
- number and length of gaps above a warning threshold (e.g., > 12 hours)

If multiple samples share the same timestamp or fall within the same effective time bin, a configurable strategy is applied (e.g., median aggregation).

---

## Hard Physical Bounds (Masking non-physical values)

For each variable \(X(t)\), user-defined bounds \([X_{min}, X_{max}]\) are applied.

If:

\[
X(t) < X_{min} \quad \text{or} \quad X(t) > X_{max}
\]

then:

- \(X(t)\) is set to NaN
- a QC flag is raised
- a warning summary is included in the QC report

This is the recommended approach for values that cannot be assigned a physical meaning (e.g., salinity in the teens for Mediterranean deployments).

---

## Derivative-Based Spike Detection

Spike detection uses the true sampling interval:

\[
\frac{dX}{dt}(t_i) \approx \frac{X(t_i) - X(t_{i-1})}{t_i - t_{i-1}}
\]

A spike is flagged if:

\[
\left|\frac{dX}{dt}\right| > T_X
\]

where \(T_X\) is a per-variable threshold defined in the YAML configuration.

Recommended usage:

- Apply spike detection primarily to kinematic/dynamical numerical features
  (curvature, vorticity, strain).
- Use conservative thresholds initially, and refine after inspecting QC reports.

---

## Flatline / Stuck Sensor Detection

A flatline condition is detected if rolling variability is near zero:

\[
\sigma_{roll}(X; W) < \epsilon_X
\]

for a window \(W\) (e.g., 24 hours) and variable-specific tolerance \(\epsilon_X\).

Flatline detection typically **flags** rather than masks data automatically, because a true constant signal is sometimes physically possible (though unlikely over long windows for most sensors).

---

## Overlap Requirements for Analyses

Pairwise and multivariate analyses require sufficient valid overlap between variables.

For a pair \((X,Y)\), analyses are computed only if:

- fraction of valid overlapping points \(\ge f_{min}\) (e.g., 0.7)
- number of overlapping points \(\ge N_{min}\) (e.g., 50)

This prevents misleading correlations driven by a small number of coincident samples.

---

## QC Report Outputs

For each platform, the QC module produces:

- `qc_summary.csv`:
  - time coverage, dt statistics, gaps
  - per-variable valid fraction
  - counts of out-of-bounds points, spike points, flatline flags

- `variable_stats.csv`:
  - min/median/max, quantiles
  - robust scale estimates (optional)

- `warnings.txt`:
  - human-readable warnings (e.g., “salinity_psu: 42 points out of bounds, masked”)

These outputs are designed to make preprocessing decisions transparent and reproducible.

