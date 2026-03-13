import logging
from dataclasses import dataclass
from pathlib import Path

from pandas import to_datetime, date_range

from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.data_store import DataStore
from data_storage.store_data import Store
from io_operation.input.definitions import DataKind
from io_operation.input.read import Read, ReadPvPlantData, ReadDataArray
from io_operation.output.write import WriteDataArray
from parameteric_evaluation.definitions import ParametricEvaluationType
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from transform.combine.combine import ArrayConcat
from transform.transform import TransformCoordinateIntoDimension, Aggregate, Apply, Rename, TransformPvPlantData
from utility import configuration
from utility.time_utils import n_periods_in_interval

logger = logging.getLogger(__name__)


class DatasetCreatorForParametricEvaluation(ParametricEvaluator):
    _key = ParametricEvaluationType.DATASET_CREATION
    # Setup and data loading
    _input_properties = {"input_root": configuration.config.get("path", "output")}
    _tou_columns = configuration.config.get("tariff", "time_of_use_labels")
    _dimensions_to_rename = {"coordinate": {"dim_1": DataKind.USER}, "to_replace_dimension": "dim_0",
                             "new_dimension": DataKind.CALCULATED.value}
    _coords_to_rename = {"dim_1": DataKind.TOU.value, "group": DataKind.MONTH.value}

    @classmethod
    def create_and_run_user_data_processing_pipeline(cls, user_type, input_filename):
        DataProcessingPipeline("read_and_store", workers=(
            Read(name=f"Read {user_type}", filename=input_filename, **cls._input_properties),
            TransformCoordinateIntoDimension(name=f"Transform {user_type}", **cls._dimensions_to_rename),
            # manage hourly data, sum all end users / plants
            Aggregate(name=f"Aggregate {user_type}", aggregate_on={"dim_1": DataKind.MONTH}),
            Apply(name=f"{user_type} TOU cols", operation=lambda x: x.sel({"dim_1": cls._tou_columns})),
            Rename(name=f"Rename {user_type}", coords=cls._coords_to_rename), Store(user_type))).execute()

    @classmethod
    def create_and_run_timeseries_processing_pipeline(cls, profile, input_filename):
        DataProcessingPipeline("read_and_store", workers=(
            Read(name=f"Read {profile}", filename=input_filename, **cls._input_properties),
            TransformCoordinateIntoDimension(name=f"Transform {profile}", **cls._dimensions_to_rename),
            # Get total production and consumption data
            # Here we manage monthly ToU values, we sum all end users/plants
            Apply(name=f"Sum users {profile}",
                  operation=lambda x: x.assign_coords(
                      dim_1=to_datetime(x.dim_1, format="mixed", dayfirst=True)
                  ).sum(DataKind.CALCULATED.value)),
            Store(profile))).execute()

    @classmethod
    def create_dataset_for_parametric_evaluation(cls):
        """ Get total consumption and production for all users separated months and time of use
        Create a single dataframe for both production and consumptions
        https://www.arera.it/dati-e-statistiche/dettaglio/analisi-dei-consumi-dei-clienti-domestici
        """
        if configuration.config.getboolean("parametric_evaluation", "read_from_cached"):
            logger.info(f"Reading data from cache...")
            DataProcessingPipeline("read_cached", workers=(
                ReadDataArray("read_energy_year", filename="energy_year"), Store("energy_year"),
                ReadDataArray("read_plant_data", filename="data_plants"), Store("data_plants"),
                ReadDataArray("read_tou_months", do_not_separate=True, filename="tou_months"),
                Store("tou_months"))).execute()

            if DataStore()["energy_year"].shape and DataStore()["data_plants"].shape and DataStore()[
                "tou_months"].shape:
                logger.info(
                    f"Yearly energy flows, PV plant data and TOU time slots successfully read from cache, continuing...")
                return

        logger.info("Creating dataset from scratch...")

        @dataclass
        class ParametricEvaluationUserType:
            user_type: str
            filename: str
            profile_type: str
            profile_filename: str
            power_type: DataKind

        def _candidate_paths(name: str) -> tuple[Path, Path]:
            output_root = Path(configuration.config.get("path", "output")).resolve()
            municipality = configuration.config.get("rec", "location")
            flat = output_root / municipality / f"{name}.csv"
            nested = output_root / DataKind.MUNICIPALITY.value / municipality / f"{name}.csv"
            return flat, nested

        user_types = (ParametricEvaluationUserType("pv_plants", "data_plants_tou", "pv_profiles", "data_plants_year",
                                                   DataKind.PRODUCTION),
                      ParametricEvaluationUserType("families", "data_families_tou", "family_profiles",
                                                   "data_families_year", DataKind.CONSUMPTION_OF_FAMILIES),
                      ParametricEvaluationUserType("users", "data_users_tou", "user_profiles", "data_users_year",
                                                   DataKind.CONSUMPTION_OF_USERS))

        required_input_names = sorted({
            *(u.filename for u in user_types),
            *(u.profile_filename for u in user_types),
        })
        missing_inputs = []
        for name in required_input_names:
            if not any(path.exists() for path in _candidate_paths(name)):
                missing_inputs.append(name)
        if missing_inputs:
            output_root = Path(configuration.config.get("path", "output")).resolve().as_posix()
            raise FileNotFoundError(
                "Missing preprocessing output tables required for parametric evaluation: "
                f"{', '.join(missing_inputs)}. Expected under '{output_root}/<municipality>/' "
                f"or '{output_root}/{DataKind.MUNICIPALITY.value}/<municipality>/'."
            )

        for user in user_types:
            cls.create_and_run_user_data_processing_pipeline(user.user_type, user.filename)
            cls.create_and_run_timeseries_processing_pipeline(user.profile_type, user.profile_filename)

        # Create a single dataframe for both production and consumption
        year = configuration.config.getint("time", "year")
        freq = configuration.config.get("time", "resolution")
        periods = n_periods_in_interval("1Y", freq)
        ut = [ut.user_type for ut in user_types]
        profile_types = [ut.profile_type for ut in user_types]
        DataProcessingPipeline("concatenate", workers=(
            ArrayConcat(dim=DataKind.USER.value, arrays_to_merge=ut, coords={DataKind.USER.value: ut}),
            Store("tou_months"), WriteDataArray("tou_months", do_not_separate=True),
            ArrayConcat(name="merge_profiles", dim=DataKind.USER.value, arrays_to_merge=profile_types,
                        coords={DataKind.USER.value: [u.power_type for u in user_types]}),
            Rename(name="rename_dimensions",
                   dims={"dim_1": DataKind.TIME.value, DataKind.USER.value: DataKind.CALCULATED.value}, ),
            Apply(name=f"assign_coords", operation=lambda x: x.assign_coords(
                {DataKind.TIME.value: date_range(start=f'{year}-01-01 00:00', freq=freq, periods=periods)})),
            Store("energy_year"), WriteDataArray("energy_year"))).execute()

        DataProcessingPipeline("pv_plants", workers=(
            ReadPvPlantData(), TransformPvPlantData(), Store("data_plants"), WriteDataArray("data_plants"))).execute()
