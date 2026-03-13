import logging

import xarray as xr
from pandas import merge

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from utility import configuration

logger = logging.getLogger(__name__)


class Combine(PipelineStage):
    stage = Stage.COMBINE
    _name = 'combine'

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        raise NotImplementedError("'execute()' is not implemented in Combine base class.")


class CalculateTypicalMonthlyConsumption(Combine):
    # Class to evaluate monthly consumption from hourly load profiles
    # evaluate the monthly consumption divided into tariff time-slots from the
    # hourly load profiles in the day-types
    _name = 'typical_monthly_consumption_calculator'

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        data_store = DataStore()
        time_of_use_time_slots = data_store["time_of_use_time_slots"]
        day_count = data_store["day_count"]

        return OmnesDataArray(xr.concat([xr.concat([dataset.isel({DataKind.DAY_TYPE.value: dt,
                                                                  DataKind.HOUR.value: time_of_use_time_slots.sel(
                                                                      {DataKind.DAY_TYPE.value: dt}) == tou}).sum(
            DataKind.HOUR.value) * day_count.sel({DataKind.DAY_TYPE.value: dt}).expand_dims(
            DataKind.TARIFF_TIME_SLOT.value).assign_coords(tariff_time_slot=[tou]).squeeze() for tou in
                                                    configuration.config.getarray("tariff", "tariff_time_slots", int)],
                                                   dim="tariff_time_slot") for dt in
                                         configuration.config.getarray("time", "day_types", int)],
                                        dim="day_type").squeeze(drop=True))


class AddYearlyConsumptionToBillData(Combine):
    _name = 'yearly_consumption_combiner'

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        data_bills = DataStore()["bills"]
        dd = xr.concat([ds.sel({DataKind.USER_DATA.value: [DataKind.MONO_TARIFF,
                                                           *configuration.config.getarray("tariff",
                                                                                          "time_of_use_labels", str),
                                                           DataKind.ANNUAL_ENERGY]}).sum(
            dim=DataKind.USER.value).assign_coords({DataKind.USER.value: u}) for u, ds in
                        data_bills.groupby(DataKind.USER.value)], dim=DataKind.USER.value,
                       join="outer").astype(float)
        return xr.concat([dd, dataset.T], dim=DataKind.USER_DATA.value, join="outer")


class Merge(Combine):
    _name = "data_merger"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    @staticmethod
    def check_labels(label, *data):
        if not all(label in dat for dat in data):
            raise ValueError(f"Not all data have a '{label}' label")

        # Extract 'label' columns and compare
        label_series = [df[label] for df in data]
        all_match = len(set.intersection(*map(set, label_series))) == len(label_series[0])

        return {'all_match': all_match,
                'matching_dataframes': [i + 1 for i, match in enumerate(all_match) for _ in range(match)]}

    def execute(self, to_merge: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError('"merge" method in DataMerger base class is not implemented.')


class MergeDataFrames(Merge):
    _name = "dataframe_merger"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, to_merge: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        merge_on = self.get_arg("merge_on", **kwargs)
        # Check if the same labels are present across all data supplied
        matches = self.check_labels(merge_on, to_merge)
        if not matches["all_match"]:
            logger.warning(f"Labels in '{merge_on}' does not match in all dataframes.")

        merged = to_merge[0]
        for merge_right in to_merge[1:]:
            merged = merge([merged, merge_right], on=merge_on)

        return merged


class ArrayConcat(Merge):
    _name = "concatenate_arrays"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray=None, *args, **kwargs) -> OmnesDataArray:
        dim = self.get_arg('dim', **kwargs)
        to_merge = self.get_arg('arrays_to_merge', **kwargs)
        coords = self.get_arg('coords', **kwargs)
        data_store = DataStore()
        return xr.concat([data_store[tm] for tm in to_merge], dim=dim).assign_coords(coords)
