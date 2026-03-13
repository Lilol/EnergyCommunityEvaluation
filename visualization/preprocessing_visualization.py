from os import makedirs
from os.path import join
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from data_storage.data_store import DataStore
from io_operation.input.definitions import DataKind
from operation.linear_equation_scaler import ScaleByLinearEquation
from operation.quadratic_optimization_scaler import ScaleByQuadraticOptimization
from operation.time_of_use_scaler import ScaleTimeOfUseProfile
from utility import configuration


def _apply_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
    })


def _safe_postfix(value):
    if value is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip())


def _save_show_close(fig, output_path):
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def plot_family_profiles(data_fam_year, **kwargs):
    _apply_plot_style()
    for m, ds in data_fam_year.groupby(DataKind.MUNICIPALITY.value):
        plot_monthly_consumption(
            ds.squeeze(dim="municipality"),
            f"Families profiles in municipality {m}",
            filename_postfix=f"families_{_safe_postfix(m)}",
        )


def plot_pv_profiles(data_plants_year, **kwargs):
    _apply_plot_style()
    for m, ds in data_plants_year.groupby(DataKind.MUNICIPALITY.value):
        plot_monthly_consumption(
            ds.squeeze(dim="municipality"),
            f"Production profiles in municipality {m}",
            filename_postfix=f"pv_{_safe_postfix(m)}",
        )


def plot_consumption_profiles(yearly_consumption_profiles, **kwargs):
    _apply_plot_style()
    # Consumption profiles
    data_store = DataStore()
    user_data = data_store["users"]
    if yearly_consumption_profiles is None:
        yearly_consumption_profiles = data_store["yearly_load_profiles_from_bills"]
    for m, ds in yearly_consumption_profiles.groupby(DataKind.MUNICIPALITY.value):
        for user_type, real in user_data.groupby(
                user_data.sel({DataKind.USER_DATA.value: DataKind.USER_TYPE}).squeeze()):
            data = ds.sel({DataKind.USER.value: user_data.loc[
                (user_data.sel({DataKind.USER_DATA.value: DataKind.USER_TYPE}) == user_type).squeeze(), m][
                DataKind.USER.value].values}).squeeze()

            # By month
            plot_monthly_consumption(data,
                                     f'Consumption profiles of {user_type.value.upper()} users\nMunicipality: {m}',
                                     filename_postfix=f"cons_{_safe_postfix(m)}_{user_type.value}")

            # Monthly consumption
            real = real.sortby(real.sel({DataKind.USER_DATA.value: DataKind.ANNUAL_ENERGY}).squeeze()).sel(
                {DataKind.USER_DATA.value: DataKind.ANNUAL_ENERGY}).squeeze().sortby(DataKind.USER.value)
            estimated = data.groupby(DataKind.USER.value).sum(dim=DataKind.TIME.value).sortby(DataKind.USER.value)
            y_real = np.arange(0, 2 * len(real), 2)
            y_est = np.arange(1, 1 + 2 * len(estimated), 2)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(y_real, real.values, label="Measured yearly energy", color="#4C78A8", alpha=0.9)
            ax.barh(y_est, estimated.values, label="Estimated yearly energy", color="#F58518", alpha=0.9)
            ax.set_yticks(y_real)
            ax.set_yticklabels(real[DataKind.USER.value].values)
            ax.set_xlabel("Energy (kWh)")
            ax.set_title(f"Measured vs estimated yearly consumption\n{user_type.value.upper()} users in {m}")
            ax.legend(loc="lower right", frameon=True)
            ax.grid(axis="x", alpha=0.25)
            for y, value in zip(y_real, real.values):
                ax.text(value, y, f" {value:.1f}", va="center", fontsize=8)
            for y, value in zip(y_est, estimated.values):
                ax.text(value, y, f" {value:.1f}", va="center", fontsize=8)

            abs_gap = np.abs(real.values - estimated.values)
            mean_abs_gap = float(np.mean(abs_gap)) if len(abs_gap) else 0.0
            ax.text(
                0.02,
                0.98,
                f"Mean absolute gap: {mean_abs_gap:.2f} kWh",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CFCFCF", "alpha": 0.9},
            )

            figures_path = configuration.config.get("path", "figures")
            makedirs(figures_path, exist_ok=True)
            filename = f"consumption_profiles_{_safe_postfix(m)}_{user_type.value}.png"
            _save_show_close(fig, join(figures_path, filename))


def to_postfix(string, separator='_'):
    safe = _safe_postfix(string)
    return "" if not safe else f"{separator}{safe}"


def plot_monthly_consumption(data, title, filename_postfix=None):
    _apply_plot_style()
    figures_path = configuration.config.get("path", "figures")
    makedirs(figures_path, exist_ok=True)

    # Mean by hour
    mean_hour = data.groupby(data.time.dt.hour).mean().T.to_pandas()
    if not hasattr(mean_hour, "columns"):
        mean_hour = mean_hour.to_frame(name="series")
    fig, ax = plt.subplots(figsize=(12, 5))
    mean_hour.plot(ax=ax, linewidth=1.8, alpha=0.85)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average power (kW)")
    ax.set_title(f"{title} - mean daily profile")
    ax.set_xlim(0, 23)
    if len(mean_hour.columns) <= 12:
        ax.legend(title="Series", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    else:
        ax.legend().remove()
    _save_show_close(fig, join(figures_path, f'mean_profiles{to_postfix(filename_postfix)}.png'))

    # Aggr. by month
    monthly_sum = data.groupby(data.time.dt.month).sum().T.to_pandas()
    if not hasattr(monthly_sum, "columns"):
        monthly_sum = monthly_sum.to_frame(name="series")
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_sum.plot(ax=ax, marker="o", linewidth=1.8, alpha=0.9)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly energy (kWh)")
    ax.set_title(f"{title} - monthly energy")
    ax.set_xticks(range(1, 13))
    ax.grid(axis="y", alpha=0.25)
    if len(monthly_sum.columns) <= 12:
        ax.legend(title="Series", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    else:
        ax.legend().remove()
    _save_show_close(fig, join(figures_path, f'monthly_energy{to_postfix(filename_postfix)}.png'))

    # Whole year
    year_series = data.sum(dim=DataKind.USER.value).to_pandas()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(year_series.index, year_series.values, color="#4C78A8", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"{title} - yearly timeline")
    p05 = float(np.percentile(year_series.values, 5))
    p95 = float(np.percentile(year_series.values, 95))
    ax.axhline(p95, color="#F58518", linestyle="--", linewidth=1.1, label=f"95th percentile ({p95:.2f})")
    ax.axhline(p05, color="#54A24B", linestyle="--", linewidth=1.1, label=f"5th percentile ({p05:.2f})")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    _save_show_close(fig, join(figures_path, f'whole_year{to_postfix(filename_postfix)}.png'))


def visualize_profile_scaling(place_to_visualize="office"):
    _apply_plot_style()
    # -------------------------------------
    # setup
    # energy consumption in one month divided in tariff time-slots (kWh)
    if place_to_visualize == "office":
        x = np.array([191.4, 73.8, 114.55])
    elif place_to_visualize == "sport centre":
        x = np.array([151.8, 130.3, 127.1])  # sport centre
    else:
        x = np.array([200, 100, 300])

    # number of days of each day-type in the month
    nd = np.array([22, 4, 5])
    # maximum demand allowed
    y_max = 1.25
    # options for methods that scale a reference profile
    #  assigned
    y_ref_db = {"office": np.array(
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.4, 0.65, 0.9, 0.95, 1, 1, 0.95, 0.9, 0.85, 0.65, 0.45, 0.4, 0.35,
         0.25, 0.25, 0.25, 0.25]), "sport centre": np.array(
        [0.3, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.45, 0.55, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.85, 1, 1, 1,
         0.95, 0.75, 0.55])}
    y_ref = np.repeat(y_ref_db[place_to_visualize][np.newaxis, :], 3, axis=0).flatten()
    # options for 'q_opt' method
    qopt_obj = 0
    qopt_obj_reg = 0
    # ------------------------------------
    # methods
    # method 'scale_gse'
    y_gse, _ = ScaleTimeOfUseProfile()(x, y_ref)
    # method 'scale_seteq'
    y_seteq, stat = ScaleByLinearEquation()(x, nd, y_ref)
    print(f"Scale, linear equation, {place_to_visualize}: ", stat)
    # method 'scale_qopt'
    y_qopt, stat = ScaleByQuadraticOptimization(x, nd, y_ref, y_max=y_max, obj=qopt_obj, obj_reg=qopt_obj_reg)
    print(f"Scale, quadratic optimization, {place_to_visualize}: ", stat)
    # ------------------------------------
    # plot
    # figure settings
    figsize = (16, 8)
    fontsize = 13
    f_styles = [dict(color="#F94144", alpha=0.08), dict(color="#F9C74F", alpha=0.08), dict(color="#90BE6D", alpha=0.08)]
    fig, ax = plt.subplots(figsize=figsize)
    number_of_time_steps = len(y_ref)
    time = np.arange(number_of_time_steps)
    # plot profiles
    ax.plot(time, y_gse, label='GSE TOU scaling', lw=2.2, ls='-')
    ax.plot(time, y_seteq, label='Linear equation scaling', lw=2.2, ls='-')
    ax.plot(time, y_qopt, label='Quadratic optimization scaling', lw=2.2, ls='-')
    # plot reference profile
    if place_to_visualize:
        ax.plot(time, y_ref, color='k', ls='--', lw=1.2, label='Reference profile')
    # plot line for 'y_max'
    if y_max:
        ax.axhline(y_max, color='#C1121F', ls='--', lw=1.4, label=f'Max power ({y_max:.2f} kW)')
    # plot division into tariff time-slots
    f_sw_pos = []
    f_sw_styles = []
    tariff_time_slots = list(configuration.config.getarray("tariff", "tariff_time_slots", int))
    switch_steps = list(configuration.config.getarray("tariff", "tariff_period_switch_time_steps", int))
    switch_steps = sorted(set([0, *switch_steps, number_of_time_steps]))
    period_ids = (tariff_time_slots * ((len(switch_steps) // max(1, len(tariff_time_slots))) + 1))[:len(switch_steps) - 1]
    for idx in range(1, len(switch_steps)):
        previous = period_ids[idx - 1]
        h = switch_steps[idx]
        start = switch_steps[idx - 1]
        f_sw_pos.append((start, h))
        f_sw_styles.append(f_styles[previous % len(f_styles)])
    for pos, style in zip(f_sw_pos, f_sw_styles):
        ax.axvspan(*pos, **style, )
    for h in range(0, number_of_time_steps, 12):
        ax.axvline(h, color='grey', ls='-', lw=0.7, alpha=0.5)
    tariff_handles = [Patch(facecolor=s["color"], alpha=s["alpha"], edgecolor="none", label=f"Tariff band {i + 1}")
                      for i, s in enumerate(f_styles)]
    ax.set_xlabel("Time (h)", fontsize=fontsize)
    ax.set_ylabel("Power (kW)", fontsize=fontsize)
    ax.set_xticks(np.arange(0, number_of_time_steps + 1, 12))
    ax.set_xlim([0, number_of_time_steps])
    ax.tick_params(labelsize=fontsize)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + tariff_handles, labels + [h.get_label() for h in tariff_handles],
              fontsize=10, bbox_to_anchor=(1.01, 0.5), loc="center left", frameon=True)
    ax.set_title(f"Profile scaling comparison - {place_to_visualize.title()}")
    ax.grid(axis='y', alpha=0.25)
    fig.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.78)
    figures_path = configuration.config.get("path", "figures")
    makedirs(figures_path, exist_ok=True)
    _save_show_close(fig, join(figures_path, f'test_methods_scaling{to_postfix(place_to_visualize)}.png'))
