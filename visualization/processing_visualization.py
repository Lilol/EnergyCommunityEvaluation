import logging
from os import makedirs
from os.path import join

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Categorical, date_range

from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import CombinedMetricEnum, PhysicalMetric
from parameteric_evaluation.physical import SharedEnergy
from utility.configuration import config
from utility.enum_definitions import convert_enum_to_value

logger = logging.getLogger(__name__)


def _setup_theme(context: str = "talk") -> None:
    sns.set_theme(style="whitegrid", context=context)
    sns.set_palette("colorblind")


def _fmt_value(value: float) -> str:
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def plot_shared_energy(input_da, n_fam, bess_size):
    _setup_theme(context="notebook")
    palette = sns.color_palette("viridis", n_colors=4)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(18, 12), constrained_layout=True)

    # Daily base
    daily = input_da.groupby("time.dayofyear").sum()
    daily, _ = SharedEnergy.calculate(daily)
    daily = daily.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})

    agg_levels = ["15min", "hour", "month", "season"]
    summary_rows = []
    df = DataFrame(columns=agg_levels + ["day"], index=range(365))
    df.loc[:, "day"] = daily.data
    for i, (agg_level, ax) in enumerate(zip(agg_levels, axes.flat)):
        logger.info(f"Plotting shared energy for: {agg_level}")
        if agg_level == "hour":
            aggregated = input_da.resample(time='1h').mean()
        elif agg_level == "4 hours":
            aggregated = input_da.resample(time='4h').mean()
        elif agg_level == "15min":
            aggregated = input_da
        elif agg_level == "season":
            aggregated = input_da.resample(time="QE", closed="left", label="right").sum()
        elif agg_level == "month":
            aggregated = input_da.resample(time="ME", closed="left", label="right").sum()
        else:
            aggregated = input_da.resample(time="YE", closed="left", label="right").sum()

        aggregated, _ = SharedEnergy.calculate(aggregated)
        aggregated = aggregated.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})

        if agg_level in ["season", "year", "month"]:
            # Create a daily time index covering the same span as aggregated
            daily_time = date_range(start=aggregated.time.min().values, end=aggregated.time.max().values, freq="D")
            # Reindex and interpolate
            aggregated_daily = aggregated.reindex(time=daily_time)
            aggregated = aggregated_daily.interpolate_na(dim="time", method="zero")
        else:
            aggregated = aggregated.resample(time="1d").sum()

        df.loc[:, agg_level] = aggregated.data.flatten()[:len(df)]
        diff_vals = df.loc[:, agg_level].values - daily.data
        sorted_vals = np.sort(diff_vals)
        diff_sorted = np.diff(sorted_vals)

        mae = float(np.mean(np.abs(diff_vals)))
        rmse = float(np.sqrt(np.mean(diff_vals ** 2)))
        bias = float(np.mean(diff_vals))
        summary_rows.append({"agg": agg_level, "mae": mae, "rmse": rmse, "bias": bias})

        axt = ax.twinx()
        ax.plot(diff_vals, label=f"{agg_level} - daily", color=palette[i], linestyle="-", linewidth=2.8, alpha=0.95)
        axt.plot(diff_sorted, label=f"{agg_level} - sorted gap", color="#6e6e6e", linestyle="-", linewidth=2.0)
        cumsum = np.cumsum(diff_sorted).astype(float)
        x = np.arange(len(cumsum))
        y1 = np.zeros(len(diff_sorted))
        axt.plot(cumsum, label=f"{agg_level} - cumulative", color="#424242", linestyle="-", linewidth=2.2)
        axt.fill_between(x, y1=y1, y2=cumsum, color="#9e9e9e", alpha=0.18)

        # Labels & titles
        ax.axhline(0, color="#4f4f4f", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
        ax.set_ylabel("Difference vs daily (kWh)")
        axt.set_ylabel("Sorted and cumulative gap (kWh)")
        ax.set_xlabel("Day of year")
        ax.set_title(f"Shared energy gap - {agg_level}", fontsize=12, weight="semibold")
        ax.tick_params(axis="both", labelsize=9)
        axt.tick_params(axis="y", labelsize=9)
        stats_text = (
            f"MAE: {_fmt_value(mae)} kWh\n"
            f"RMSE: {_fmt_value(rmse)} kWh\n"
            f"Bias: {_fmt_value(bias)} kWh"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#d0d0d0"},
        )
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = axt.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True, fontsize=8)
        sns.despine(ax=ax, right=False)

    fig.suptitle(
        f"Shared Energy Gap Across Aggregation Levels\nFamilies={int(n_fam)} | Battery size={bess_size} kWh",
        fontsize=16,
        weight="bold",
    )
    if summary_rows:
        best = min(summary_rows, key=lambda row: row["rmse"])
        fig.text(
            0.5,
            0.01,
            f"Best approximation vs daily by RMSE: {best['agg']} (RMSE={_fmt_value(best['rmse'])} kWh)",
            ha="center",
            fontsize=10,
            color="#3f3f3f",
        )

    figures_path = config.get("path", "figures")
    makedirs(figures_path, exist_ok=True)
    plt.savefig(join(figures_path, "shared_energy_aggregations.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()


def plot_metrics(results, n_fam, bess_size):
    _setup_theme(context="talk")
    df = DataFrame(columns=["time_resolution", "metric", "value"])
    for label in results.metric:
        label = label.values.item()
        if not isinstance(label, CombinedMetricEnum):
            continue
        df.loc[len(df), :] = [label.first, label.second,
                              results.sel(metric=label, battery_size=bess_size, number_of_families=n_fam).values[0]]

    df = df.map(convert_enum_to_value)
    if df.empty:
        logger.warning("No metric data available for plotting.")
        return

    palette = {"Self sufficiency": "#1f77b4", "Self consumption": "#ff7f0e", "Self production": "#2ca02c",
               "Grid liability": "#d62728"}

    order = ["15min", "hour", "dayofyear", "month", "season", "year"]  # logical ordering
    df["time_resolution"] = Categorical(df["time_resolution"], categories=order, ordered=True)

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df,
        x="time_resolution",
        y="value",
        hue="metric",
        style="metric",
        palette=palette,
        hue_order=palette.keys(),
        markers=True,
        dashes=False,
        linewidth=2.2,
        markersize=8,
    )
    sns.scatterplot(
        data=df,
        x="time_resolution",
        y="value",
        hue="metric",
        style="metric",
        palette=palette,
        hue_order=palette.keys(),
        s=110,
        alpha=0.9,
        ax=ax,
        edgecolor="white",
        linewidth=0.8,
        legend=False,
    )

    # Highlight best time resolution per metric with a star marker.
    best_rows = df.loc[df.groupby("metric")["value"].idxmax()]
    sns.scatterplot(
        data=best_rows,
        x="time_resolution",
        y="value",
        hue="metric",
        palette=palette,
        hue_order=palette.keys(),
        marker="*",
        s=260,
        edgecolor="#2f2f2f",
        linewidth=0.7,
        legend=False,
        ax=ax,
    )

    # Add reference lines
    for y in [0, 0.5, 1]:
        ax.axhline(y, ls="--", color="#808080", lw=0.8, alpha=0.7)

    # Value labels
    for _, row in df.iterrows():
        ax.text(
            row["time_resolution"],
            row["value"] + 0.02,
            f"{row['value']:.2f}",
            ha="center",
            fontsize=8,
            color="#2f2f2f",
        )

    ax.set_xlabel("Time resolution")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Parametric metrics - Families={n_fam}, Battery={bess_size} kWh", fontsize=13, weight="semibold")
    ax.set_ylim(min(-0.05, df["value"].min() - 0.05), max(1.05, df["value"].max() + 0.08))
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    ax.legend(title="Metric", frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    for _, row in best_rows.iterrows():
        ax.annotate(
            "best",
            (row["time_resolution"], row["value"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color="#303030",
        )
    sns.despine()
    plt.tight_layout()
    figures_path = config.get("path", "figures")
    makedirs(figures_path, exist_ok=True)
    plt.savefig(
        join(figures_path, f"metrics_families_{int(n_fam)}_bess_{bess_size}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
