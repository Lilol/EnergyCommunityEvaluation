import logging

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from utility import configuration
from utility.definitions import grouper

logger = logging.getLogger(__name__)


class Check(PipelineStage):
    stage = Stage.CHECK
    _name = "data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        raise NotImplementedError


class CheckAnnualSum(Check):
    _name = "annual_sum_checker"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        time_of_use_labels = configuration.config.getarray("tariff", "time_of_use_labels", str)
        groups = grouper(dataset, DataKind.USER.value, user_data=DataKind.MONTH)
        for (user, month), ds in dataset.groupby(groups):
            time_of_use_consumption = ds.sel({DataKind.USER_DATA.value: time_of_use_labels}).sum()
            annual_consumption = ds.sel({DataKind.USER_DATA.value: DataKind.ANNUAL_ENERGY}).sum()
            if time_of_use_consumption.values != annual_consumption.values:
                logger.warning(
                    f"The sum of time of use energy consumption must match the aggregated consumption for every user and"
                    f" month, but discrepancy found for annual consumption: {annual_consumption.values:.2f} <==> summed time of use "
                    f"consumption: {time_of_use_consumption.values:.2f} for user {user} and month {month}.")
        dataset.loc[..., DataKind.ANNUAL_ENERGY] = dataset.loc[..., time_of_use_labels].sum(DataKind.USER_DATA.value)
        return dataset
