import logging

import numpy as np
import xarray as xr
from pandas import date_range, DataFrame

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from utility import configuration
from utility.day_of_the_week import get_weekday_code

logger = logging.getLogger(__name__)


class Extract(PipelineStage):
    stage = Stage.EXTRACT
    _name = "data_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        raise NotImplementedError


class ExtractTypicalYear(Extract):
    _name = "typical_year_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        dds = xr.concat([xr.concat([dd.assign_coords({DataKind.TIME.value: dd.time.dt.hour}).rename(
            {DataKind.TIME.value: DataKind.HOUR.value}).expand_dims(DataKind.DAY_OF_MONTH.value).assign_coords(
            {DataKind.DAY_OF_MONTH.value: [day, ]}) for day, dd in df.groupby(df.time.dt.day)],
            dim=DataKind.DAY_OF_MONTH.value).expand_dims(DataKind.MONTH.value).assign_coords(
            {DataKind.MONTH.value: [month, ]}) for month, df in dataset.groupby(dataset.time.dt.month)],
            dim=DataKind.MONTH.value)

        day_types = DataStore()["day_types"]
        return OmnesDataArray(xr.concat([
            dds.where(day_types.where(day_types == i)).mean(DataKind.DAY_OF_MONTH.value, skipna=True).expand_dims(
                {DataKind.DAY_TYPE.value: [i, ]}) for i in configuration.config.getarray("time", "day_types", int)],
            dim=DataKind.DAY_TYPE.value)).transpose(DataKind.USER.value, DataKind.MONTH.value, DataKind.DAY_TYPE.value,
                                                    DataKind.HOUR.value, DataKind.MUNICIPALITY.value)


class ExtractTimeOfUseParameters(Extract):
    _name = "tariff_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    """
    ______

    # Total number and list of tariff time-slots
    # ARERA's day-types depending on subdivision into tariff time-slots
    # NOTE : f : 1 - tariff time-slot F1, central hours of work-days
    #            2 - tariff time-slot F2, evening of work-days, and saturdays
    #            3 - tariff times-lot F2, night, sundays and holidays
    """

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        tariff_time_slots = np.unique(dataset)
        configuration.config.set_and_check("tariff", "tariff_time_slots", tariff_time_slots,
                                           configuration.config.setarray, check=False)
        configuration.config.set_and_check("tariff", "number_of_time_of_use_periods", len(tariff_time_slots))

        # time-steps where there is a change of tariff time-slot
        h_switch_arera = np.where(dataset[:, :-1].values - dataset[:, 1:].values)
        h_switch_arera = (h_switch_arera[0], h_switch_arera[1] + 1)
        configuration.config.set_and_check("tariff", "tariff_period_switch_time_steps", h_switch_arera,
                                           configuration.config.setarray, check=False)

        # number of day-types (index j)
        # NOTE : j : 0 - work-days (monday-friday)
        #            1 - saturdays
        #            2 - sundays and holidays
        configuration.config.set_and_check("time", "number_of_day_types", dataset.shape[0])
        configuration.config.set_and_check("time", "day_types", list(range(dataset.shape[0])),
                                           configuration.config.setarray, check=False)

        # number of time-steps during each day
        configuration.config.set_and_check("time", "number_of_time_steps_per_day", dataset.shape[1])

        # total number of time-steps
        configuration.config.set_and_check("time", "total_number_of_time_steps", dataset.size)

        return dataset


class ExtractDayTypesInTimeframe(Extract):
    _name = "day_type_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        ref_year = configuration.config.get("time", "year")
        start = kwargs.pop("start", f"{ref_year}-01-01")
        end = kwargs.pop("end", f"{ref_year}-12-31")
        index = date_range(start=start, end=end, freq="d")
        ref_df = DataFrame(data=index.map(get_weekday_code), index=index, columns=[DataKind.DAY_TYPE, ])
        return xr.concat([
            OmnesDataArray(df.astype(int).set_index(df.index.day).rename(columns={DataKind.DAY_TYPE: month}),
                           dims=(DataKind.DAY_OF_MONTH.value, DataKind.MONTH.value)) for month, df in
            ref_df.groupby(ref_df.index.month)], dim=DataKind.MONTH.value, join="outer")


class ExtractDayCountInTimeframe(Extract):
    _name = "day_count_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        dataset = xr.concat([OmnesDataArray(unique_numbers[1], dims=DataKind.DAY_TYPE.value,
                                            coords={DataKind.DAY_TYPE.value: unique_numbers[0]}).expand_dims(
            {DataKind.MONTH.value: [i, ]}) for i, da in enumerate(dataset.T, 1) if
            (unique_numbers := np.unique(da, return_counts=True))], dim=DataKind.MONTH.value,
            join="outer").drop_sel({DataKind.DAY_TYPE.value: [np.nan]}, errors="ignore").fillna(0).astype(int)
        return dataset.assign_coords({DataKind.DAY_TYPE.value: dataset[DataKind.DAY_TYPE.value].astype(int)})


class ExtractTimeOfUseTimeSlotCountByDayType(Extract):
    _name = "time_of_use_time_slot_count_by_day_type_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        return xr.concat([OmnesDataArray(unique_numbers[1], dims=DataKind.TARIFF_TIME_SLOT.value,
                                         coords={DataKind.TARIFF_TIME_SLOT.value: unique_numbers[0]}).expand_dims(
            {DataKind.DAY_TYPE.value: [i, ]}) for i, a in enumerate(dataset.values) if
            (unique_numbers := np.unique(a, return_counts=True))], dim=DataKind.DAY_TYPE.value,
            join="outer").fillna(0)


class ExtractTimeOfUseTimeSlotCountByMonth(Extract):
    _name = "time_of_use_time_slot_count_by_month_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        day_type_count = DataStore()["day_count"]
        dataset = (dataset * day_type_count).sum("day_type")
        return dataset