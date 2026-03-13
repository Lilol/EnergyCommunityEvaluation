import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import PhysicalMetric, OtherParameters, LoadMatchingMetric
from parameteric_evaluation.load_matching_evaluation import (
    SelfConsumption, SelfSufficiency, SelfProduction, GridLiability
)


class TestLoadMatchingEvaluation(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')

    def test_self_consumption_calculation(self):
        """Test SelfConsumption calculation"""
        shared = np.array([5.0] * 24)
        injected = np.array([10.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.SHARED_ENERGY, OtherParameters.INJECTED_ENERGY]
        }
        data = np.array([shared, injected])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, value = SelfConsumption.calculate(input_da)

        # Self consumption = shared / injected = (5*24) / (10*24) = 0.5
        self.assertAlmostEqual(value, 0.5, places=5)

    def test_self_sufficiency_calculation(self):
        """Test SelfSufficiency calculation"""
        shared = np.array([8.0] * 24)
        withdrawn = np.array([16.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.SHARED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]
        }
        data = np.array([shared, withdrawn])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, value = SelfSufficiency.calculate(input_da)

        # Self sufficiency = shared / withdrawn = (8*24) / (16*24) = 0.5
        self.assertAlmostEqual(value, 0.5, places=5)

    def test_grid_liability_calculation(self):
        """Test GridLiability calculation"""
        injected = np.array([12.0] * 24)
        withdrawn = np.array([10.0] * 24)
        total_consumption = np.array([10.0] * 24)
        shared = np.array([5.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [
                PhysicalMetric.SHARED_ENERGY,
                OtherParameters.INJECTED_ENERGY,
                OtherParameters.WITHDRAWN_ENERGY,
                PhysicalMetric.TOTAL_CONSUMPTION,
            ]
        }
        data = np.array([shared, injected, withdrawn, total_consumption])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, value = GridLiability.calculate(input_da)

        # Grid liability = (injected - withdrawn) / total_load = (12-10) / 10 = 0.2
        # value might be OmnesDataArray, extract scalar
        if hasattr(value, 'values'):
            value = float(value.values)
        self.assertAlmostEqual(value, 0.2, places=5)

    def test_grid_liability_negative_when_imports_dominate(self):
        """GL must be negative when imported energy exceeds exported energy."""
        injected = np.array([8.0] * 24)
        withdrawn = np.array([10.0] * 24)
        total_consumption = np.array([10.0] * 24)

        input_da = OmnesDataArray(
            data=np.array([injected, withdrawn, total_consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: [
                    OtherParameters.INJECTED_ENERGY,
                    OtherParameters.WITHDRAWN_ENERGY,
                    PhysicalMetric.TOTAL_CONSUMPTION,
                ],
            },
        )

        _, value = GridLiability.calculate(input_da)
        if hasattr(value, 'values'):
            value = float(value.values)
        self.assertLess(value, 0.0)
        self.assertAlmostEqual(value, -0.2, places=5)

    def test_grid_liability_positive_when_exports_dominate(self):
        """GL must be positive when exported energy exceeds imported energy."""
        injected = np.array([14.0] * 24)
        withdrawn = np.array([10.0] * 24)
        total_consumption = np.array([10.0] * 24)

        input_da = OmnesDataArray(
            data=np.array([injected, withdrawn, total_consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: [
                    OtherParameters.INJECTED_ENERGY,
                    OtherParameters.WITHDRAWN_ENERGY,
                    PhysicalMetric.TOTAL_CONSUMPTION,
                ],
            },
        )

        _, value = GridLiability.calculate(input_da)
        if hasattr(value, 'values'):
            value = float(value.values)
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, 0.4, places=5)

    def test_self_consumption_with_zero_injected(self):
        """Test SelfConsumption with zero injected energy"""
        shared = np.array([0.0] * 24)
        injected = np.array([0.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.SHARED_ENERGY, OtherParameters.INJECTED_ENERGY]
        }
        data = np.array([shared, injected])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        # The implementation doesn't raise an error for zero division, it returns 0 or NaN
        result, value = SelfConsumption.calculate(input_da)
        # Just verify it doesn't crash
        self.assertIsNotNone(value)

    def test_self_sufficiency_with_zero_withdrawn(self):
        """Test SelfSufficiency with zero withdrawn energy"""
        shared = np.array([0.0] * 24)
        withdrawn = np.array([0.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.SHARED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]
        }
        data = np.array([shared, withdrawn])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        # The implementation doesn't raise an error for zero division, it returns 0 or NaN
        result, value = SelfSufficiency.calculate(input_da)
        # Just verify it doesn't crash
        self.assertIsNotNone(value)

    def test_load_matching_calculator_keys(self):
        """Test LoadMatchingParameterCalculator keys"""
        self.assertEqual(SelfConsumption._key, LoadMatchingMetric.SELF_CONSUMPTION)
        self.assertEqual(SelfSufficiency._key, LoadMatchingMetric.SELF_SUFFICIENCY)
        self.assertEqual(SelfProduction._key, LoadMatchingMetric.SELF_PRODUCTION)
        self.assertEqual(GridLiability._key, LoadMatchingMetric.GRID_LIABILITY)

    def test_self_consumption_denominator(self):
        """Test SelfConsumption denominator"""
        self.assertEqual(SelfConsumption._denominator, OtherParameters.INJECTED_ENERGY)

    def test_self_sufficiency_denominator(self):
        """Test SelfSufficiency denominator"""
        self.assertEqual(SelfSufficiency._denominator, OtherParameters.WITHDRAWN_ENERGY)

    def test_grid_liability_nominator_denominator(self):
        """Test GridLiability nominator and denominator"""
        self.assertEqual(GridLiability._nominator, OtherParameters.INJECTED_ENERGY)
        self.assertEqual(GridLiability._denominator, OtherParameters.WITHDRAWN_ENERGY)

    def test_self_consumption_realistic_values(self):
        """Test SelfConsumption with realistic values"""
        # During day: high production, low consumption -> more injected
        # During night: no production, high consumption -> more withdrawn
        shared = np.array([3.0] * 12 + [0.0] * 12)
        injected = np.array([5.0] * 12 + [0.0] * 12)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.SHARED_ENERGY, OtherParameters.INJECTED_ENERGY]
        }
        data = np.array([shared, injected])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, value = SelfConsumption.calculate(input_da)

        # Self consumption = (3*12) / (5*12) = 0.6
        self.assertAlmostEqual(value, 0.6, places=5)


if __name__ == '__main__':
    unittest.main()
