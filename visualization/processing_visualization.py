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


def plot_shared_energy(input_da, n_fam, bess_size):
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 15))

    # Daily base
    daily = input_da.groupby("time.dayofyear").sum()
    daily, _ = SharedEnergy.calculate(daily)
    daily = daily.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})

    agg_levels = ["15min", "hour", "month", "season"]
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
        sorted_vals = sorted(diff_vals)
        diff_sorted = np.diff(sorted_vals).flat

        axt = ax.twinx()
        ax.plot(diff_vals[:], label=f"{agg_level}-daily", color=palette[i], linestyle="-", linewidth=3)
        axt.plot(diff_sorted, label=f"{agg_level}-daily sorted", color="grey", linestyle="-", linewidth=2.5)
        cumsum = np.cumsum(diff_sorted).astype(float)
        x = np.arange(len(cumsum))
        y1 = np.zeros(len(diff_sorted))
        axt.plot(cumsum, label=f"{agg_level}-daily cumulative", color="darkgrey", linestyle="-", linewidth=2.5)
        axt.fill_between(x, y1=y1, y2=cumsum, color="grey", alpha=0.3)

        # Labels & titles
        ax.set_ylabel('Difference between aggregates (kWh)')
        axt.set_ylabel('Sorted difference values (kWh)')
        ax.set_xlabel('Day of Year')
        ax.set_title(f"Shared energy for: {agg_level}-daily")
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = axt.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True)

    fig.suptitle(f"Shared Energy Gap\nFamilies = {int(n_fam)}, Battery Size = {bess_size} kWh", weight="bold")

    plt.tight_layout()
    figures_path = config.get("path", "figures")
    makedirs(figures_path, exist_ok=True)
    plt.savefig(join(figures_path, "shared_energy_aggregations.png"), dpi=300)
    plt.show()
    plt.close()


def plot_metrics(results, n_fam, bess_size):
    df = DataFrame(columns=["time_resolution", "metric", "value"])
    for label in results.metric:
        label = label.values.item()
        if not isinstance(label, CombinedMetricEnum):
            continue
        df.loc[len(df), :] = [label.first, label.second,
                              results.sel(metric=label, battery_size=bess_size, number_of_families=n_fam).values[0]]

    df = df.map(convert_enum_to_value)

    palette = {"Self sufficiency": "#1f77b4", "Self consumption": "#ff7f0e", "Self production": "#2ca02c",
               "Grid liability": "#d62728"}

    order = ["15min", "hour", "dayofyear", "month", "season", "year"]  # logical ordering
    df["time_resolution"] = Categorical(df["time_resolution"], categories=order, ordered=True)

    plt.figure(figsize=(8, 5))
    ax = sns.scatterplot(data=df, x="time_resolution", y="value", hue="metric", style="metric", palette=palette, s=40,
                         hue_order=palette.keys(), legend=False, edgecolor="darkgrey", linewidth=1)
    sns.scatterplot(data=df, x="time_resolution", y="value", hue="metric", style="metric", palette=palette, s=200,
                         alpha=0.7, hue_order=palette.keys(), ax=ax, edgecolor="darkgrey", linewidth=1)

    # Add reference lines
    for y in [0, 0.5, 1]:
        ax.axhline(y, ls="--", color="gray", lw=0.7)

    # Value labels
    for _, row in df.iterrows():
        ax.text(row["time_resolution"], row["value"] + 0.02, f"{row['value']:.2f}", ha='center', fontsize=8)

    plt.xlabel('Time resolution')
    plt.ylabel('Metric value')
    plt.title(f'Number of families={n_fam}, Battery capacity={bess_size} kWh')
    plt.legend(title="Metric type")
    plt.tight_layout()
    plt.show()
