### CODE DESCRIPTION

#### master building

#### analysis
src/bgcd/analysis/

- preprocess.py
# QC Module – Function-by-Function Description  
File: `src/bgcd/analysis/preprocess.py`

---

## Public API

### `apply_qc(df, config, time_col="time_utc", group_col="platform_id", outdir=None)`

Main entry point for Quality Control.

**Purpose**
Applies all QC steps to a MASTER dataframe and optionally writes a QC report.

**Steps executed (in order)**

1. Convert time column to UTC datetime.
2. Time-axis validation and duplicate handling.
3. Hard physical bounds masking.
4. Derivative-based spike detection.
5. Flatline (stuck sensor) detection.
6. Build summary and variable statistics tables.
7. Optionally write QC reports to disk.

**Returns**
- `df_clean`: cleaned dataframe (copy of input, with NaNs + optional QC flags).
- `QCResult`: container with summary tables and warnings.

---

## Data Container

### `QCResult`

Dataclass storing QC outputs:

- `summary`: per-platform time and overlap summary table.
- `variable_stats`: per-variable QC counts and descriptive statistics.
- `warnings`: list of human-readable warning strings.
- `flag_columns`: names of QC flag columns added to dataframe.
- `outdir`: output directory path (if writing enabled).

---

## Internal Functions

---

### `_infer_single_platform_id(df, group_col)`

Infers the platform_id from dataframe.

- If exactly one unique id → returns that id.
- If multiple ids → returns `"__MULTI__"` to avoid silent misuse.
- If none → `"__UNKNOWN__"`.

Used only for report folder naming.

---

### `_check_time_axis_and_handle_duplicates(df, qc_cfg, time_col, group_col)`

**Purpose**
Validates time axis and handles duplicate timestamps.

**Operations**
- Drops rows with invalid timestamps.
- Detects duplicate `(platform_id, time)` rows.
- Aggregates duplicates using `_aggregate_duplicates`.
- Computes:
  - median Δt
  - percentiles of Δt
  - number of large gaps
  - maximum gap
- Emits warnings if:
  - median Δt deviates from expected value
  - gaps exceed threshold

**Returns**
- cleaned dataframe (no NaT, no duplicates)
- time_summary DataFrame
- warnings list

---

### `_aggregate_duplicates(df, group_col, time_col, strategy)`

Aggregates duplicate timestamps per platform.

**Strategy**
- Numeric columns → median / first / last
- Non-numeric columns → first

Used internally by time-axis check.

---

### `_apply_physical_bounds(df, qc_cfg, write_flags)`

**Purpose**
Masks physically implausible values.

**Logic**
For each variable defined in YAML bounds:
- If value < min or > max → set to NaN
- Optionally create boolean flag column:
  `qc_out_of_bounds_<var>`

**Returns**
- cleaned dataframe
- bounds statistics table
- warnings
- list of created flag columns

---

### `_detect_derivative_spikes(df, qc_cfg, time_col, group_col, write_flags)`

**Purpose**
Detects abrupt spikes based on time-derivative.

**Method**
For each variable with defined threshold:
- Compute discrete derivative:
  
  dx/dt ≈ (xᵢ − xᵢ₋₁) / (tᵢ − tᵢ₋₁)

- If |dx/dt| > threshold → spike
- Spike assigned to point i (later sample)

**Action**
- `mask_points` → set spike to NaN
- `flag_only` → keep value but flag

**Returns**
- cleaned dataframe
- spike statistics table
- warnings
- list of created flag columns

---

### `_detect_flatline(df, qc_cfg, time_col, group_col, write_flags)`

**Purpose**
Detects potential stuck sensors.

**Method**
For each selected variable:
- Compute rolling standard deviation over time window (e.g., 24h)
- If rolling std < epsilon → flag as flatline

**Action**
- Default: flag only
- Optional: mask values

**Returns**
- flatline statistics table
- warnings
- list of created flag columns
- dataframe (possibly masked)

---

### `_compute_overlap_overview(df, qc_cfg)`

Stores overlap requirements from config:

- `min_fraction_overlap`
- `min_points`

These are used later in coupling/multivariate modules.

Returns a single-row DataFrame (`platform_id="__ALL__"`).

---

### `_merge_summaries(parts)`

Merges multiple summary DataFrames on `platform_id`.

Used to combine:
- time summary
- overlap settings

---

### `_merge_variable_stats(parts, df_clean)`

Builds final per-variable QC table.

Combines:
- bounds stats
- spike stats
- flatline stats

Adds descriptive statistics from cleaned data:
- n_valid
- frac_valid
- 1st percentile
- median
- 99th percentile
- min
- max

Returns a single DataFrame.

---

### `_write_warnings(path, warnings)`

Writes warnings list to text file.

Each warning is one line.

---

# QC Workflow Summary

The QC module performs:

1. Time validation
2. Duplicate handling
3. Physical plausibility filtering
4. Numerical spike detection
5. Sensor health diagnostics
6. Report generation

The module:
- Never interpolates
- Never silently modifies data
- Always logs masking decisions
- Returns a clean copy of the dataset

Designed for:
- Irregular Lagrangian time series
- Multi-sensor oceanographic datasets
- Transparent and reproducible preprocessing



