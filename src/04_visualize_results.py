"""
Neuro-Symbolic Forecasting — Professional Visualisation
========================================================
Produces a four-panel analytics figure:

  Panel A  (top, wide)   — Time-series: Actual vs Predicted, ASP anomaly markers
  Panel B  (middle left) — Residual scatter (predicted vs residual), coloured by anomaly status
  Panel C  (middle right)— Hour-of-day mean absolute error profile with ±1 std band
  Panel D  (bottom)      — ASP anomaly breakdown horizontal bar chart

Output: notebooks/neuro_symbolic_plot.png  (300 dpi)
"""

import os

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Colour palette (consistent across all panels)
# ---------------------------------------------------------------------------
C_ACTUAL   = "#1A1A2E"   # very dark navy
C_PRED     = "#2196F3"   # material blue
C_ANOM     = "#E53935"   # material red
C_BAND     = "#90CAF9"   # light-blue confidence band
C_RESIDUAL = "#78909C"   # blue-grey
C_HOLIDAY  = "#FF8F00"   # amber — holiday background shading
BG_PANEL   = "#F8FAFD"   # near-white panel background
GRID_CLR   = "#CFD8DC"   # light grid lines

ANOMALY_MARKER_MAP = {
    "Critical AI Deviation":                       ("X",  C_ANOM,   90),
    "Ramp Rate Violation (Spike)":                 ("^",  "#FF6F00", 70),
    "Ramp Rate Violation (Drop)":                  ("v",  "#6A1B9A", 70),
    "Negative Generation Predicted":               ("o",  "#B71C1C", 80),
    "Predicted Below Historical Grid Baseline":    ("s",  "#1B5E20", 80),
}
DEFAULT_MARKER = ("D", "#616161", 70)


def _style_axes(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor(BG_PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_CLR)
    ax.spines["bottom"].set_color(GRID_CLR)
    if grid:
        ax.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.8)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=7, color="#1A1A2E")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#37474F")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#37474F")
    ax.tick_params(labelsize=8, colors="#37474F")


# ---------------------------------------------------------------------------
# Panel A — Time-series forecast (7-day window)
# ---------------------------------------------------------------------------
def _panel_timeseries(ax, df_window, anomalies_window, holiday_dates):
    # Shade public holiday bands
    prev_hday = None
    for ts in df_window["timestamp"]:
        d = ts.date()
        if d in holiday_dates and d != prev_hday:
            ax.axvspan(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1),
                       alpha=0.10, color=C_HOLIDAY, zorder=0)
            prev_hday = d

    ax.plot(df_window["timestamp"], df_window["power_generation"],
            color=C_ACTUAL, linewidth=1.4, label="Actual Generation", zorder=3)
    ax.plot(df_window["timestamp"], df_window["predicted_generation"],
            color=C_PRED, linewidth=1.2, linestyle="--", alpha=0.85,
            label="dCeNN-ELM Prediction", zorder=3)

    # ASP anomaly markers grouped by reason
    if not anomalies_window.empty:
        for reason, grp in anomalies_window.groupby("reason"):
            marker, color, sz = ANOMALY_MARKER_MAP.get(reason, DEFAULT_MARKER)
            ax.scatter(grp["timestamp"], grp["predicted_generation"],
                       marker=marker, color=color, s=sz, zorder=6,
                       linewidths=0.5, edgecolors="white",
                       label=f"ASP: {reason}")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Fill ±10 MW confidence band around prediction
    ax.fill_between(df_window["timestamp"],
                    df_window["predicted_generation"] - 10,
                    df_window["predicted_generation"] + 10,
                    alpha=0.12, color=C_BAND, zorder=2, label="±10 MW Band")

    _style_axes(ax,
                title="Panel A — Forecast vs Actual (Jan 2024, 7-day window)",
                xlabel="",
                ylabel="Power Generation (MW)")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=7.5,
              framealpha=0.92, edgecolor=GRID_CLR, ncol=2)

    # Annotation for worst anomaly
    if not anomalies_window.empty:
        worst = anomalies_window.iloc[0]
        ax.annotate(
            worst["reason"],
            xy=(worst["timestamp"], worst["predicted_generation"]),
            xytext=(15, 18), textcoords="offset points",
            fontsize=7, color=C_ANOM,
            arrowprops=dict(arrowstyle="->", color=C_ANOM, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_ANOM, lw=0.8),
        )


# ---------------------------------------------------------------------------
# Panel B — Residual scatter
# ---------------------------------------------------------------------------
def _panel_residuals(ax, df_full, anomaly_ts_set):
    residuals = df_full["power_generation"] - df_full["predicted_generation"]
    is_anom = df_full["timestamp"].isin(anomaly_ts_set)

    ax.scatter(df_full.loc[~is_anom, "predicted_generation"],
               residuals[~is_anom],
               s=6, alpha=0.35, color=C_RESIDUAL, linewidths=0, label="Normal")
    ax.scatter(df_full.loc[is_anom, "predicted_generation"],
               residuals[is_anom],
               s=25, alpha=0.85, color=C_ANOM, linewidths=0.4,
               edgecolors="white", zorder=5, label="ASP Anomaly")

    ax.axhline(0, color="#B0BEC5", linewidth=0.9, linestyle="-")

    # Loess-like rolling mean trend
    pred_sorted = df_full["predicted_generation"].sort_values()
    res_sorted  = residuals.reindex(pred_sorted.index)
    rolling     = res_sorted.rolling(window=200, center=True, min_periods=50).mean()
    ax.plot(pred_sorted, rolling, color="#FF6F00", linewidth=1.2,
            linestyle="-", alpha=0.75, label="Trend (rolling mean)")

    _style_axes(ax,
                title="Panel B — Residuals vs Predicted",
                xlabel="Predicted Generation (MW)",
                ylabel="Residual: Actual − Predicted (MW)")
    ax.legend(fontsize=7.5, framealpha=0.92, edgecolor=GRID_CLR)


# ---------------------------------------------------------------------------
# Panel C — Hourly MAE profile
# ---------------------------------------------------------------------------
def _panel_hourly_mae(ax, df_full):
    df_full = df_full.copy()
    df_full["abs_err"] = (df_full["power_generation"] - df_full["predicted_generation"]).abs()
    df_full["hour"]    = pd.to_datetime(df_full["timestamp"]).dt.hour

    hourly = df_full.groupby("hour")["abs_err"].agg(["mean", "std"]).reset_index()

    ax.fill_between(hourly["hour"],
                    hourly["mean"] - hourly["std"],
                    hourly["mean"] + hourly["std"],
                    alpha=0.20, color=C_PRED, label="±1 std")
    ax.plot(hourly["hour"], hourly["mean"],
            color=C_PRED, linewidth=1.6, marker="o", markersize=4,
            markerfacecolor="white", markeredgewidth=1.2,
            label="Mean MAE")

    # Highlight worst hour
    worst_h = hourly.loc[hourly["mean"].idxmax()]
    ax.axvline(worst_h["hour"], color=C_ANOM, linewidth=0.8,
               linestyle=":", alpha=0.7)
    ax.text(worst_h["hour"] + 0.3, worst_h["mean"],
            f'Worst: hour {int(worst_h["hour"])}\n({worst_h["mean"]:.1f} MW)',
            fontsize=7.5, color=C_ANOM, va="bottom")

    # Shade evening hours
    ax.axvspan(18, 23, alpha=0.07, color=C_HOLIDAY, label="Evening (18–23h)")

    ax.set_xticks(range(0, 24, 2))
    _style_axes(ax,
                title="Panel C — Hourly MAE Profile (full test set)",
                xlabel="Hour of Day",
                ylabel="Mean Absolute Error (MW)")
    ax.legend(fontsize=7.5, framealpha=0.92, edgecolor=GRID_CLR)


# ---------------------------------------------------------------------------
# Panel D — ASP anomaly breakdown bar
# ---------------------------------------------------------------------------
def _panel_anomaly_breakdown(ax, df_anomalies):
    if df_anomalies is None or df_anomalies.empty:
        ax.text(0.5, 0.5, "No anomalies detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#607D8B")
        _style_axes(ax, title="Panel D — ASP Anomaly Breakdown", grid=False)
        return

    counts = df_anomalies["reason"].value_counts().sort_values()
    colors = [ANOMALY_MARKER_MAP.get(r, (None, "#607D8B", None))[1] for r in counts.index]

    bars = ax.barh(counts.index, counts.values, color=colors,
                   height=0.55, edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8.5, fontweight="bold",
                color="#37474F")

    ax.set_xlim(0, counts.max() * 1.18)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _style_axes(ax,
                title="Panel D — ASP Anomaly Breakdown (full Jan 2024 window)",
                xlabel="Count",
                ylabel="",
                grid=False)
    ax.tick_params(axis="y", labelsize=8)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def plot_neuro_symbolic_results(preds_path, anomalies_path, output_image_path):
    print("Loading data for visualization...")

    if not os.path.exists(preds_path):
        print("Predictions file not found.")
        return

    df_preds = pd.read_csv(preds_path)
    df_preds["timestamp"] = pd.to_datetime(df_preds["timestamp"])

    has_anomalies = os.path.exists(anomalies_path)
    df_anomalies = None
    if has_anomalies:
        df_anomalies = pd.read_csv(anomalies_path)
        df_anomalies["timestamp"] = pd.to_datetime(df_anomalies["timestamp"])

    # Detect holiday dates from predictions column if present, else empty set
    holiday_dates = set()
    if "is_holiday" in df_preds.columns:
        holiday_dates = set(df_preds.loc[df_preds["is_holiday"] == 1, "timestamp"].dt.date)

    # 7-day window (first 672 rows of Jan 2024)
    df_window = df_preds.head(672).copy()
    anomalies_window = (pd.DataFrame() if df_anomalies is None
                        else df_anomalies[df_anomalies["timestamp"].isin(df_window["timestamp"])].copy())
    anomaly_ts_set = set() if df_anomalies is None else set(df_anomalies["timestamp"])

    # ---------------------------------------------------------------------------
    # Figure layout: 4 panels via GridSpec
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 13), facecolor="white")
    fig.suptitle(
        "dCeNN-ELM + ASP Neuro-Symbolic Energy Forecasting\n"
        "Austrian Power Grid — January 2024 Analysis",
        fontsize=13, fontweight="bold", color="#1A1A2E", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.4, 1.6, 1.2],
        hspace=0.48,
        wspace=0.32,
        left=0.07, right=0.97, top=0.93, bottom=0.05,
    )

    ax_ts    = fig.add_subplot(gs[0, :])       # Panel A — full width top
    ax_res   = fig.add_subplot(gs[1, 0])       # Panel B — bottom left
    ax_hour  = fig.add_subplot(gs[1, 1])       # Panel C — bottom right
    ax_bar   = fig.add_subplot(gs[2, :])       # Panel D — full width bottom

    _panel_timeseries(ax_ts,   df_window,    anomalies_window, holiday_dates)
    _panel_residuals(ax_res,   df_preds,     anomaly_ts_set)
    _panel_hourly_mae(ax_hour, df_preds)
    _panel_anomaly_breakdown(ax_bar, df_anomalies)

    # Footer metadata
    n_anom = 0 if df_anomalies is None else len(df_anomalies)
    rmse   = np.sqrt(((df_preds["power_generation"] - df_preds["predicted_generation"]) ** 2).mean())
    mae    = (df_preds["power_generation"] - df_preds["predicted_generation"]).abs().mean()
    fig.text(
        0.5, 0.005,
        f"Test Set RMSE: {rmse:.2f} MW  |  MAE: {mae:.2f} MW  |  "
        f"Total ASP Anomalies: {n_anom}  |  Holiday timesteps shaded amber",
        ha="center", fontsize=8, color="#607D8B", style="italic",
    )

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Visualization saved to {output_image_path}")
    plt.close()


if __name__ == "__main__":
    PREDS_FILE    = "data/predictions_2024.csv"
    ANOMALIES_FILE = "data/flagged_anomalies.csv"
    OUTPUT_IMAGE  = "notebooks/neuro_symbolic_plot.png"

    plot_neuro_symbolic_results(PREDS_FILE, ANOMALIES_FILE, OUTPUT_IMAGE)