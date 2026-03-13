from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from configparser import RawConfigParser
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import xarray as xr

from io_operation.input.definitions import DataKind

DEMO_MUNICIPALITY = "ColabTown"
DEMO_REPO_URL = "https://github.com/Lilol/EnergyCommunityEvaluation.git"
FAMILY_TEMPLATE_ID = "family_template"
PV_PLANTS = (
    {"id": "PV_COLAB_01", "size_kw": 8.0, "descrizione": "PV plant on the school roof", "indirizzo": "Via Demo 10"},
    {"id": "PV_COLAB_02", "size_kw": 5.5, "descrizione": "PV plant on the library roof", "indirizzo": "Via Demo 12"},
)
DEMO_USERS = (
    {"pod": "POD_COLAB_01", "descrizione": "Student House A", "indirizzo": "Via Demo 1", "tipo": "dom", "potenza": 6.0,
     "base_monthly_kwh": 320.0},
    {"pod": "POD_COLAB_02", "descrizione": "Student House B", "indirizzo": "Via Demo 2", "tipo": "dom", "potenza": 6.0,
     "base_monthly_kwh": 280.0},
)

REQUIRED_PREPROCESSING_TABLES = (
    "data_users_tou",
    "data_users_year",
    "data_families_tou",
    "data_families_year",
    "data_plants_tou",
    "data_plants_year",
)


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _seasonal_variation(month: int, peak_month: int = 1, amplitude: float = 0.16) -> float:
    return 1 + amplitude * np.cos(2 * np.pi * (month - peak_month) / 12)


def _daily_reference_shape(day_type: int, month_index: int) -> np.ndarray:
    base = np.array([
        0.42, 0.38, 0.35, 0.34, 0.35, 0.40, 0.58, 0.78, 0.72, 0.58, 0.50, 0.47,
        0.49, 0.52, 0.55, 0.58, 0.66, 0.82, 1.00, 1.08, 1.02, 0.84, 0.62, 0.50,
    ])
    if day_type == 1:
        base *= 0.92
        base[8:18] *= 0.95
    elif day_type == 2:
        base *= 0.88
        base[7:10] *= 1.08
        base[18:22] *= 0.95

    if month_index in (11, 0, 1):
        base[[6, 7, 18, 19, 20, 21]] *= 1.10
    elif month_index in (5, 6, 7):
        base[[12, 13, 14, 15, 16]] *= 1.08

    return np.round(base, 4)


def _build_tariff_table() -> pd.DataFrame:
    weekday = [3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    saturday = [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
    sunday = [3] * 24
    df = pd.DataFrame([weekday, saturday, sunday], index=[0, 1, 2], columns=list(range(24)))
    df.index.name = None
    return df


def _build_typical_load_profiles() -> pd.DataFrame:
    rows = []
    profile_adjustments = {
        "dom": np.ones(24),
        "bta": np.array([
            0.50, 0.45, 0.42, 0.40, 0.42, 0.48, 0.62, 0.82, 0.95, 1.05, 1.10, 1.12,
            1.12, 1.08, 1.05, 1.02, 0.98, 0.92, 0.82, 0.72, 0.62, 0.56, 0.52, 0.50,
        ]),
    }
    for month_index in range(12):
        for user_type, multiplier in profile_adjustments.items():
            row = {"type": user_type, "month": month_index}
            for day_type in range(3):
                profile = np.round(_daily_reference_shape(day_type, month_index) * multiplier, 4)
                for hour, value in enumerate(profile):
                    row[f"y_j{day_type}_i{hour}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _monthly_tou_breakdown(total_kwh: float, month: int) -> tuple[float, float, float]:
    f1_share = 0.30 if month in (6, 7, 8) else 0.34
    f2_share = 0.31
    f3_share = 1 - f1_share - f2_share
    f1 = round(total_kwh * f1_share, 2)
    f2 = round(total_kwh * f2_share, 2)
    f3 = round(total_kwh - f1 - f2, 2)
    return f1, f2, f3


def _build_bills(pod: str, base_monthly_kwh: float, annual_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        total = round(base_monthly_kwh * _seasonal_variation(month, amplitude=0.14) + annual_shift, 2)
        f1, f2, f3 = _monthly_tou_breakdown(total, month)
        rows.append({
            "pod": pod,
            "anno": 2019,
            "mese": month,
            "totale": total,
            "f0": np.nan,
            "f1": f1,
            "f2": f2,
            "f3": f3,
        })
    return pd.DataFrame(rows)


def _build_pvgis_profile(pv_size_kw: float) -> pd.DataFrame:
    timestamps = pd.date_range("2019-01-01 00:00", "2020-01-01 00:00", freq="1h", inclusive="left")
    time_parts = pd.Series(timestamps)
    hours = time_parts.dt.hour.to_numpy()
    day_of_year = time_parts.dt.dayofyear.to_numpy()

    hour_angle = np.pi * (hours - 6) / 12
    daylight = np.sin(np.clip(hour_angle, 0, np.pi))
    daylight = np.clip(daylight, 0, None)

    seasonal = np.sin(np.pi * (day_of_year - 1) / 365)
    seasonal = np.clip(seasonal, 0.10, None)

    weather = 0.92 + 0.05 * np.sin(day_of_year / 5) + 0.03 * np.cos(day_of_year / 17)
    power = np.round(pv_size_kw * daylight * seasonal * weather, 4)
    return pd.DataFrame({"power": power}, index=timestamps)


def _write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=';', index=index)


def _write_common_input_files(common_dir: Path) -> None:
    _write_csv(_build_tariff_table(), common_dir / "arera.csv", index=True)
    typical_profiles = _build_typical_load_profiles()
    typical_profiles.index.name = None
    _write_csv(typical_profiles, common_dir / "y_ref_gse.csv", index=True)


def _write_municipality_input_files(municipality_dir: Path) -> dict[str, float]:
    users = pd.DataFrame([{key: user[key] for key in ("pod", "descrizione", "indirizzo", "tipo", "potenza")} for user in DEMO_USERS])
    user_bills = pd.concat([
        _build_bills(user["pod"], user["base_monthly_kwh"], annual_shift=10 * index)
        for index, user in enumerate(DEMO_USERS)
    ], ignore_index=True)
    family_bills = _build_bills(FAMILY_TEMPLATE_ID, base_monthly_kwh=180.0)
    production_profiles = {plant["id"]: _build_pvgis_profile(plant["size_kw"]) for plant in PV_PLANTS}

    plant_data = pd.DataFrame([
        {
            "pod": plant["id"],
            "descrizione": plant["descrizione"],
            "indirizzo": plant["indirizzo"],
            "pv_size": plant["size_kw"],
            "produzione annua [kWh]": round(float(production_profiles[plant["id"]]["power"].sum()), 2),
            "rendita specifica [kWh/kWp]": round(float(production_profiles[plant["id"]]["power"].sum()) / plant["size_kw"], 2),
        }
        for plant in PV_PLANTS
    ])

    _write_csv(users, municipality_dir / "lista_pod.csv")
    _write_csv(user_bills, municipality_dir / "dati_bollette.csv")
    _write_csv(family_bills, municipality_dir / "bollette_domestici.csv")
    _write_csv(plant_data, municipality_dir / "lista_impianti.csv")

    pvgis_dir = municipality_dir / "PVGIS"
    pvgis_dir.mkdir(parents=True, exist_ok=True)
    for plant_id, profile in production_profiles.items():
        pvgis_to_write = profile.copy()
        pvgis_to_write.index.name = "timestamp"
        pvgis_to_write.index = pvgis_to_write.index.strftime("%d/%m/%Y %H:%M")
        pvgis_to_write.to_csv(pvgis_dir / f"{plant_id}.csv", sep=';')
    return {"annual_production_kwh": round(float(plant_data["produzione annua [kWh]"].sum()), 2)}


def _write_demo_parameter_files(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    cost_path = data_dir / "cost_of_equipment_demo.csv"
    emission_path = data_dir / "emission_factors_demo.csv"

    cost_path.write_text(dedent(
        """\
        equipment,cost_type,max_size,cost
        pv,capex,20,1500
        pv,capex,200,1200
        pv,capex,600,1100
        pv,capex,inf,1050
        bess,capex,,300
        user,capex,,100
        bess,opex,,100
        pv,opex,,0
        """
    ), encoding="utf-8")

    emission_path.write_text(dedent(
        """\
        factor,value
        grid,0.263
        inj,0
        prod,0.05
        bess,175
        """
    ), encoding="utf-8")

    return {"cost_path": cost_path, "emission_path": emission_path}


def _write_demo_rec_structure(config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    rec_path = config_dir / "rec_structure_demo.json"
    rec_structure = {
        "school": {"generators": [PV_PLANTS[0]["id"]], "loads": [DEMO_USERS[0]["pod"]]},
        "municipal_office": {"generators": [PV_PLANTS[1]["id"]], "loads": [DEMO_USERS[1]["pod"]]},
        "families": {"generators": [], "loads": ["families_12"]},
    }
    rec_path.write_text(json.dumps(rec_structure, indent=2), encoding="utf-8")
    return rec_path


def _write_demo_config(workspace_root: Path, parameter_paths: dict[str, Path], rec_structure_path: Path) -> Path:
    config_dir = workspace_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "colab_demo_config.ini"

    paths = {
        "root": workspace_root,
        "input": workspace_root / "input",
        "output": workspace_root / "output",
        "figures": workspace_root / "figures",
        "reference_profile_source": workspace_root / "input" / "DatabaseGSE",
        "common": workspace_root / "input" / "Common",
        "rec_data": workspace_root / "input" / "DatiComuni",
        "log": workspace_root / "logs",
    }
    path_values = {key: value.as_posix() for key, value in paths.items()}
    cost_path = parameter_paths["cost_path"].as_posix()
    emission_path = parameter_paths["emission_path"].as_posix()

    config_text = dedent(
        f"""\
        [global]
        seed=42
        country=Italy

        [path]
        root={path_values['root']}
        input={path_values['input']}
        output={path_values['output']}
        figures={path_values['figures']}
        reference_profile_source={path_values['reference_profile_source']}
        common={path_values['common']}
        rec_data={path_values['rec_data']}
        log={path_values['log']}

        [rec]
        setup_file={rec_structure_path.as_posix()}
        location={DEMO_MUNICIPALITY}
        municipalities=all
        number_of_families=12
        n_families_to_check=6,12,18

        [production]
        estimator=PVGIS

        [profile]
        scaling_method=proportional

        [tariff]
        time_of_use_labels=energy1,energy2,energy3
        tariff_time_slots=1,2,3
        number_of_time_of_use_periods=3
        tariff_period_switch_time_steps=0,9,17,21

        [time]
        resolution=1h
        year=2019
        number_of_day_types=3
        day_types=0,1,2
        number_of_time_steps_per_day=24
        total_number_of_time_steps=72

        [output]
        file_format=csv

        [visualization]
        check_by_plotting=True

        [parametric_evaluation]
        read_from_cached=False
        to_evaluate=physical,economic,environmental,time_aggregation
        time_aggregation_metrics=self_consumption,self_sufficiency
        evaluation_parameters={{'battery_size': [0,4], 'number_of_families': [6,12,18]}}
        self_consumption_targets=0,0.25,0.5,0.75,1
        max_number_of_households=30
        min_number_of_households=0
        emission_factors_configuration_file={emission_path}
        cost_configuration_file={cost_path}
        economic_evaluation_number_of_years=20
        """
    )
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def create_demo_workspace(project_root: Path | str, workspace_root: Path | str) -> dict[str, str]:
    project_root = Path(project_root).resolve()
    workspace_root = Path(workspace_root).resolve()
    _reset_directory(workspace_root)

    common_dir = workspace_root / "input" / "Common"
    municipality_dir = workspace_root / "input" / "DatiComuni" / DEMO_MUNICIPALITY
    (workspace_root / "input" / "DatabaseGSE").mkdir(parents=True, exist_ok=True)
    (workspace_root / "output").mkdir(parents=True, exist_ok=True)
    (workspace_root / "figures").mkdir(parents=True, exist_ok=True)
    (workspace_root / "logs").mkdir(parents=True, exist_ok=True)

    _write_common_input_files(common_dir)
    production_info = _write_municipality_input_files(municipality_dir)
    parameter_paths = _write_demo_parameter_files(workspace_root / "data")
    rec_structure_path = _write_demo_rec_structure(workspace_root / "config")
    config_path = _write_demo_config(workspace_root, parameter_paths, rec_structure_path)

    return {
        "project_root": project_root.as_posix(),
        "workspace_root": workspace_root.as_posix(),
        "config_path": config_path.as_posix(),
        "municipality": DEMO_MUNICIPALITY,
        "annual_production_kwh": f"{production_info['annual_production_kwh']:.2f}",
    }


def _run_project_script(project_root: Path, script_name: str, config_path: Path) -> None:
    subprocess.run(
        [sys.executable, script_name, "--config", str(config_path)],
        cwd=project_root,
        check=True,
    )


def _load_demo_config(config_path: Path) -> RawConfigParser:
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Demo configuration file not found: '{config_path.as_posix()}'")
    cfg = RawConfigParser()
    with open(config_path, encoding="utf-8") as stream:
        cfg.read_file(stream)
    return cfg


def _normalize_output_layout(workspace_root: Path | str, municipality: str = DEMO_MUNICIPALITY) -> None:
    workspace_root = Path(workspace_root).resolve()
    nested_output = workspace_root / "output" / DataKind.MUNICIPALITY.value / municipality
    flat_output = workspace_root / "output" / municipality
    if not nested_output.exists():
        return

    flat_output.mkdir(parents=True, exist_ok=True)
    for file_path in nested_output.glob("*"):
        if file_path.is_file():
            shutil.copy2(file_path, flat_output / file_path.name)


def _ensure_plants_year_file(workspace_root: Path | str, municipality: str = DEMO_MUNICIPALITY) -> None:
    workspace_root = Path(workspace_root).resolve()
    output_dir = workspace_root / "output" / municipality
    target_file = output_dir / "data_plants_year.csv"
    if target_file.exists():
        return

    input_dir = workspace_root / "input" / "DatiComuni" / municipality / "PVGIS"
    plant_files = sorted(input_dir.glob("PV_*.csv"))
    if not plant_files:
        return

    profiles: dict[str, pd.Series] = {}
    for plant_file in plant_files:
        profile = pd.read_csv(plant_file, sep=';', index_col=0, parse_dates=True)
        if profile.empty:
            continue
        profiles[plant_file.stem] = profile.iloc[:, 0].astype(float)

    if not profiles:
        return

    # Match existing yearly CSV layout: users as rows, timestamps as columns.
    plants_year = pd.DataFrame(profiles).T
    plants_year.index.name = DataKind.USER.value
    output_dir.mkdir(parents=True, exist_ok=True)
    plants_year.to_csv(target_file, sep=';', float_format='%.4f')


def run_preprocessing(project_root: Path | str, config_path: Path | str) -> None:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    _run_project_script(project_root, "preprocessing.py", config_path)

    cfg = _load_demo_config(config_path)
    root = cfg.get("path", "root")
    municipality = cfg.get("rec", "location")
    _normalize_output_layout(root, municipality=municipality)
    _ensure_plants_year_file(root, municipality=municipality)


def run_parametric_evaluation(project_root: Path | str, config_path: Path | str) -> None:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()

    cfg = _load_demo_config(config_path)
    root = cfg.get("path", "root")
    municipality = cfg.get("rec", "location")
    _normalize_output_layout(root, municipality=municipality)
    _ensure_plants_year_file(root, municipality=municipality)
    _run_project_script(project_root, "run_parametric_evaluation.py", config_path)


def get_output_table_path(workspace_root: Path | str, name: str, municipality: str = DEMO_MUNICIPALITY) -> Path:
    workspace_root = Path(workspace_root).resolve()
    direct_path = workspace_root / "output" / municipality / f"{name}.csv"
    if direct_path.exists():
        return direct_path
    nested_path = workspace_root / "output" / DataKind.MUNICIPALITY.value / municipality / f"{name}.csv"
    return nested_path


def validate_preprocessing_outputs(
    workspace_root: Path | str,
    municipality: str = DEMO_MUNICIPALITY,
    required_tables: tuple[str, ...] = REQUIRED_PREPROCESSING_TABLES,
) -> dict[str, Path]:
    available: dict[str, Path] = {}
    missing: list[str] = []
    for name in required_tables:
        path = get_output_table_path(workspace_root, name=name, municipality=municipality)
        if path.exists():
            available[name] = path
        else:
            missing.append(name)

    if missing:
        expected_dir = Path(workspace_root).resolve() / "output" / municipality
        raise FileNotFoundError(
            "Missing preprocessing outputs before parametric evaluation: "
            f"{', '.join(missing)}. Expected under '{expected_dir.as_posix()}'."
        )
    return available


def run_demo_evaluation(project_root: Path | str, config_path: Path | str) -> list[Path]:
    config_path = Path(config_path).resolve()
    cfg = _load_demo_config(config_path)
    workspace_root = Path(cfg.get("path", "root")).resolve()
    municipality = cfg.get("rec", "location")

    validate_preprocessing_outputs(workspace_root, municipality=municipality)
    run_parametric_evaluation(project_root=project_root, config_path=config_path)
    return find_result_files(workspace_root=workspace_root, municipality=municipality)


def read_output_table(workspace_root: Path | str, name: str, municipality: str = DEMO_MUNICIPALITY) -> pd.DataFrame:
    output_path = get_output_table_path(workspace_root=workspace_root, name=name, municipality=municipality)
    return pd.read_csv(filepath_or_buffer=output_path, sep=';')


def open_output_dataarray(
    workspace_root: Path | str,
    name: str,
    municipality: str = DEMO_MUNICIPALITY,
    *,
    separated: bool = True,
) -> xr.DataArray:
    base_path = Path(workspace_root).resolve() / "output"
    file_path = base_path / municipality / f"{name}.nc" if separated else base_path / f"{name}.nc"
    if not file_path.exists() and separated:
        file_path = base_path / DataKind.MUNICIPALITY.value / municipality / f"{name}.nc"
    return xr.open_dataarray(file_path, engine="netcdf4")


def find_result_files(workspace_root: Path | str, municipality: str = DEMO_MUNICIPALITY) -> list[Path]:
    results_dir = Path(workspace_root).resolve() / "output" / municipality
    if not results_dir.exists():
        results_dir = Path(workspace_root).resolve() / "output" / DataKind.MUNICIPALITY.value / municipality
    return sorted(results_dir.glob("results_*.nc")) + sorted(results_dir.glob("results_*.csv.nc"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a Colab-friendly teaching dataset for EnergyCommunityEvaluation.")
    parser.add_argument(
        "--workspace-root",
        default=Path("colab_runtime").as_posix(),
        help="Folder where the demo input, output, config, and figures will be created.",
    )
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1].as_posix(),
        help="Path to the EnergyCommunityEvaluation project root.",
    )
    parser.add_argument("--run-preprocessing", action="store_true", help="Run preprocessing.py after generating the dataset.")
    parser.add_argument(
        "--run-parametric-evaluation",
        action="store_true",
        help="Run run_parametric_evaluation.py after generating the dataset.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    info = create_demo_workspace(project_root=project_root, workspace_root=workspace_root)
    print(json.dumps(info, indent=2))

    config_path = Path(info["config_path"])
    if args.run_preprocessing:
        run_preprocessing(project_root, config_path)
    if args.run_parametric_evaluation:
        result_files = [path.as_posix() for path in run_demo_evaluation(project_root, config_path)]
        print(json.dumps({"result_files": result_files}, indent=2))


if __name__ == "__main__":
    main()

