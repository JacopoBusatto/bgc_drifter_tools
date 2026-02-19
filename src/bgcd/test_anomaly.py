import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- helpers (copiati dalla tua logica) ---
def compute_hourly_climatology_mean(t, y, min_samples_per_hour=3):
    tt = pd.to_datetime(t, errors="coerce")
    tmp = pd.DataFrame({"t": tt, "y": y}).dropna()
    mean_by_hour = tmp["y"].groupby(tmp["t"].dt.hour).mean().reindex(np.arange(24))
    cnt_by_hour = tmp["y"].groupby(tmp["t"].dt.hour).size().reindex(np.arange(24)).fillna(0).astype(int)
    mean_by_hour[cnt_by_hour < min_samples_per_hour] = np.nan
    return mean_by_hour, cnt_by_hour

def hourly_anomaly(t, y, min_samples_per_hour=3):
    clim, cnt = compute_hourly_climatology_mean(t, y, min_samples_per_hour=min_samples_per_hour)
    tt = pd.to_datetime(t, errors="coerce")
    return y - tt.dt.hour.map(clim), clim, cnt

def smooth_time(y, t, window="24H"):
    s = pd.Series(np.asarray(y), index=pd.DatetimeIndex(t)).sort_index()
    return s.rolling(window=window, min_periods=1).mean().reindex(pd.DatetimeIndex(t)).to_numpy()

# --- synthetic time (3-hour sampling + jitter) ---
rng = np.random.default_rng(0)
t = pd.date_range("2025-09-01", periods=24*14//3, freq="3h")  # 14 days, every 3h
t = t + pd.to_timedelta(rng.integers(-20, 21, size=len(t)), unit="min")  # jitter +/-20 min

# --- synthetic signal ---
hours = (t - t[0]).total_seconds() / 3600.0
trend = 0.02 * (hours / 24.0)                   # slow drift
diurnal = 0.4 * np.sin(2*np.pi*(hours % 24)/24) # 24h cycle
noise = 0.06 * rng.standard_normal(len(t))      # noise
y_raw = trend + diurnal + noise

# --- compare: anomaly computed on raw vs on smoothed ---
y_smooth = smooth_time(y_raw, t, "24h")

anom_raw, clim_raw, cnt_raw = hourly_anomaly(pd.Series(t), pd.Series(y_raw), min_samples_per_hour=3)
anom_smooth, clim_sm, cnt_sm = hourly_anomaly(pd.Series(t), pd.Series(y_smooth), min_samples_per_hour=3)

print("Counts per hour-of-day (raw):")
print(cnt_raw.to_string())
print("\nHourly climatology (raw):")
print(clim_raw.to_string())

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(t, y_smooth, label="signal (smooth 24H)")
ax2 = ax.twinx()
ax2.plot(t, anom_smooth, "--", label="anom from smoothed")
ax2.plot(t, smooth_time(anom_raw, t, "24h"), "--", label="anom raw then smooth 24H")

# one legend
h1,l1 = ax.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc="upper left", fontsize=9, frameon=True)

ax.grid(True, alpha=0.3)
ax.set_title("Synthetic test")
plt.tight_layout()
plt.show()
