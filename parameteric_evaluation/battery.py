import logging

import numpy as np
import xarray as xr

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import BatteryPowerFlows, OtherParameters

logger = logging.getLogger(__name__)


class Battery(Calculator):
    def __init__(self, size, t_hours=1):
        self._size = size
        self.p_max = self._size / t_hours

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        return Battery(kwargs.pop('size'), t_hours=kwargs.pop('t_min')).manage_bess(
            input_da), results_of_previous_calculations

    def manage_bess(self, dataset: OmnesDataArray) -> OmnesDataArray:
        logger.info(f"Calculating Battery operation for BESS size: {self._size}kWh and max power: {self.p_max}kW...")
        calc_dim = DataKind.CALCULATED.value
        time_dim = DataKind.TIME.value

        bess_coords = {**dataset.coords, calc_dim: [m for m in BatteryPowerFlows if m.value != "invalid"]}
        bess_power = OmnesDataArray(0.0, dims=dataset.dims, coords=bess_coords)
        dataset = xr.concat([dataset, bess_power], dim=calc_dim)

        if self._size == 0:
            return dataset

        inj = dataset.sel({calc_dim: OtherParameters.INJECTED_ENERGY})
        withdrawn = dataset.sel({calc_dim: OtherParameters.WITHDRAWN_ENERGY})

        charge = inj - withdrawn
        time_vals = inj[time_dim].values

        stored = xr.zeros_like(inj)
        bess_charge = xr.zeros_like(inj)
        new_inj = xr.zeros_like(inj)
        new_with = xr.zeros_like(inj)

        e = xr.zeros_like(charge.isel({time_dim: 0}))

        for t, time_val in enumerate(time_vals):
            power = charge.isel({time_dim: t})

            charge_max = xr.where(
                power < 0,
                xr.ufuncs.maximum(power, xr.ufuncs.maximum(-e, -self.p_max)),
                xr.ufuncs.minimum(power, xr.ufuncs.minimum(self._size - e, self.p_max))
            )

            e = e + charge_max

            stored.loc[{time_dim: time_val}] = e
            bess_charge.loc[{time_dim: time_val}] = charge_max
            new_inj.loc[{time_dim: time_val}] = inj.isel({time_dim: t}) - charge_max
            new_with.loc[{time_dim: time_val}] = withdrawn.isel({time_dim: t}) + charge_max

        dataset.loc[{calc_dim: BatteryPowerFlows.POWER_CHARGE}] = bess_charge
        dataset.loc[{calc_dim: BatteryPowerFlows.STORED_ENERGY}] = stored
        dataset.loc[{calc_dim: OtherParameters.INJECTED_ENERGY}] = new_inj
        dataset.loc[{calc_dim: OtherParameters.WITHDRAWN_ENERGY}] = new_with

        return dataset
