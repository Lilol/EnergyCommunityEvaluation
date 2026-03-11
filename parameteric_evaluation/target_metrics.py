import logging
import sys
from typing import Iterable

import numpy as np
from pandas import DataFrame
from xarray import concat

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.output.write import Write2DData
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, LoadMatchingMetric
from parameteric_evaluation.other_calculators import WithdrawnEnergy, InjectedEnergy
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import PhysicalParameterCalculator, TotalConsumption, SharedEnergy
from utility import configuration
from visualization.visualize import plot_target_metrics_evaluation, plot_target_metrics_summary

# Handle override decorator for Python < 3.12
if sys.version_info >= (3, 12):
    from typing import override
else:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback: create a no-op decorator
        def override(func):
            return func

logger = logging.getLogger(__name__)

_metric_family_ratio = {
    LoadMatchingMetric.SELF_CONSUMPTION: "+",
    LoadMatchingMetric.SELF_SUFFICIENCY: "-",
    LoadMatchingMetric.SELF_PRODUCTION: "+",
}


class TargetMetricParameterCalculator(Calculator):
    _key = None
    _metric = LoadMatchingMetric.INVALID
    _param_calculator = None  # Will be set in subclasses
    _search_direction = _metric_family_ratio.get(_metric, " ")  # Default to positive if not specified

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        ...

    @classmethod
    def eval(cls, da: OmnesDataArray, n_fam) -> float:
        """Evaluate the metric for a given number of families."""
        if cls._param_calculator is None:
            cls._param_calculator = PhysicalParameterCalculator.create(cls._metric)
        da, _ = TotalConsumption.calculate(da, number_of_families=n_fam)
        da, _ = WithdrawnEnergy.calculate(da, update_existing=True)
        da, _ = InjectedEnergy.calculate(da, update_existing=True)
        da, _ = SharedEnergy.calculate(da)
        return cls._param_calculator.calculate(da, number_of_families=n_fam)[1].item()

    @staticmethod
    def find_closer(n_fam, step):
        """Return closer integer to n_fam, considering the step."""
        if n_fam % step == 0:
            return n_fam
        if n_fam % step >= step / 2:
            return (n_fam // step) * step + step
        else:
            return (n_fam // step) * step

    @classmethod
    def find_best_value(cls, da, n_fam_high, n_fam_low, step, current_value):
        # Stopping criterion (considering that n_fam is integer)
        if n_fam_high - n_fam_low <= step:
            logger.info("Procedure ended without exact match.")
            return (n_fam_high, cls.eval(da, n_fam_high)) if cls._search_direction == "+" else (n_fam_low,
                                                                                                       cls.eval(
                                                                                                           da,
                                                                                                           n_fam_low))

        # Bisection of the current space
        n_fam_mid = cls.find_closer((n_fam_low + n_fam_high) // 2, step)
        mid = cls.eval(da, n_fam_mid)

        # Evaluate and update
        if np.isclose(mid, current_value, 0.05):  # Check if close match is found
            logger.info(f"Found close match: {mid:.4f},{current_value}.")
            return n_fam_mid, mid

        # For "+": mid < target means we need MORE families (search upper half)
        # For "-": mid < target means we need FEWER families (search lower half)
        if cls._search_direction == "+":
            if mid < current_value:
                # Need more families -> search upper half
                return cls.find_best_value(da, n_fam_high, n_fam_mid, step, current_value)
            else:
                # Need fewer families -> search lower half
                return cls.find_best_value(da, n_fam_mid, n_fam_low, step, current_value)
        else:  # "-" direction
            if mid < current_value:
                # Need fewer families -> search lower half
                return cls.find_best_value(da, n_fam_mid, n_fam_low, step, current_value)
            else:
                # Need more families -> search upper half
                return cls.find_best_value(da, n_fam_high, n_fam_mid, step, current_value)

    @classmethod
    def call(cls, input_da: OmnesDataArray | None = None,
             results_of_previous_calculations: OmnesDataArray | None = None, parameters: dict | None = None,
             **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | float]:
        """
        Finds the optimal number of families to satisfy a given self-consumption
        ratio.

        Parameters:
        - target_value (float): Target metric ratio.
        - n_fam_max (int): Maximum number of families.
        - p_plants (numpy.ndarray): Array of power values from plants.
        - p_users (numpy.ndarray): Array of power values consumed by users.
        - p_fam (numpy.ndarray): Array of power values consumed by each family.
        - step (int): Step in the number of families.

        Returns:
        - tuple: Tuple containing the optimal number of families and the
            corresponding shared energy ratio.
        """

        # Goal: Find number of families that achieves target metric value (or as close as possible)
        n_fam_max = kwargs.get("maximum_number_of_families")
        n_fam_min = kwargs.get("minimum_number_of_families", 0)
        target_value = kwargs.get("target_value")
        step = kwargs.get("step_size", 5)

        # Evaluate boundary values to check if target is reachable
        value_at_min = cls.eval(input_da, n_fam_min)
        value_at_max = cls.eval(input_da, n_fam_max)

        # Check if target is within the reachable range
        min_val = min(value_at_min, value_at_max)
        max_val = max(value_at_min, value_at_max)

        if target_value < min_val or target_value > max_val:
            # Target is outside reachable range - return closest boundary
            if abs(value_at_min - target_value) < abs(value_at_max - target_value):
                logger.info(
                    f"Target ({target_value}) outside range [{min_val:.3f}, {max_val:.3f}], closest is min families ({n_fam_min}), value={value_at_min:.3f}!")
                return n_fam_min, value_at_min
            else:
                logger.info(
                    f"Target ({target_value}) outside range [{min_val:.3f}, {max_val:.3f}], closest is max families ({n_fam_max}), value={value_at_max:.3f}!")
                return n_fam_max, value_at_max

        # Target is within range - do bisection search to find closest match
        return cls.find_best_value(input_da, n_fam_max, n_fam_min, step, target_value)

    @classmethod
    def requirement_satisfied(cls, current_value, target_value):
        """Check if the requirement is satisfied. For all metrics, a target is satisfied when current >= target.
        The search direction (+ or -) only affects HOW we search for the solution, not WHEN it's satisfied."""
        if np.isclose(current_value, target_value, 0.05):
            return True
        return current_value >= target_value

    @classmethod
    def order_targets(cls, targets):
        """Order targets in ascending order (easy to hard) for both directions.
        This allows us to process achievable targets first and skip harder ones efficiently."""
        return sorted(targets)  # Always ascending - process easy to hard

    @classmethod
    def family_count_limit_reached(cls, n_fam, n_fam_max, n_fam_min):
        """Check if the family count limit is reached based on the search direction."""
        if cls._search_direction == "+":
            return n_fam >= n_fam_max
        elif cls._search_direction == "-":
            return n_fam <= n_fam_min
        else:
            logger.warning(
                f"Unknown search direction '{cls._search_direction}' for metric '{cls._metric.value}'. Assuming family count limit is not reached.")
            return False  # Default to False if search direction is unknown


# Dynamically create calculator subclasses
for metric in LoadMatchingMetric:
    if not metric.valid():
        continue

    if not configuration.config.has_option("parametric_evaluation",
                                           f"{metric.value.lower().replace(' ', '_')}_targets"):
        continue
    class_name = f"{metric.value.replace(' ', '')}TargetCalculator"


    def make_calculator(m):
        class _Calc(TargetMetricParameterCalculator):
            _key = f"{metric.value.replace(' ', '')}TargetCalculator"
            _metric = m
            _param_calculator = PhysicalParameterCalculator.create(m)
            _search_direction = _metric_family_ratio.get(m, " ")  # Default to positive if not specified

        _Calc.__name__ = class_name
        return _Calc


    globals()[class_name] = make_calculator(metric)


class TargetMetricEvaluator(ParametricEvaluator):
    _name = "Target metric evaluator"
    _key = ParametricEvaluationType.METRIC_TARGETS
    _max_number_of_households = configuration.config.getint("parametric_evaluation", "max_number_of_households")
    _min_number_of_households = configuration.config.getint("parametric_evaluation", "min_number_of_households", 0)

    @classmethod
    @override
    def invoke(cls, *args, **kwargs):
        logger.info(f"Invoke parametric evaluator '{cls._name}'...")
        dataset = kwargs.pop('dataset', args[0])
        results = kwargs.pop("results", args[1])
        dfs = []
        results_da = None
        for metric, calculator in cls._parameter_calculators.items():
            df, da = cls.evaluate_targets(dataset, calculator, **kwargs)
            dfs.append(df)
            # Combine OmnesDataArrays
            if results_da is None:
                results_da = da
            else:
                results_da = OmnesDataArray(concat([results_da, da], dim="metric", join="outer"))

        # Write combined results to CSV
        if results_da is not None:
            # Write2DData(filename="target_metrics_evaluation").execute(
            #     results_da, separate_to_directories_by=None
            # )
            # Visualize results
            plot_target_metrics_evaluation(results_da)
            plot_target_metrics_summary(results_da)

        logger.info(f"Parametric evaluation finished.")
        return dataset, results

    @staticmethod
    def get_eval_metrics(evaluation_type):
        return {f"{m.value.replace(' ', '')}TargetCalculator": TargetMetricParameterCalculator.get_subclass(
            f"{m.value.replace(' ', '')}TargetCalculator") for m in
            LoadMatchingMetric if m.valid() and configuration.config.has_option("parametric_evaluation",
                                                                                f"{m.value.lower().replace(' ', '_')}_targets")}

    @classmethod
    def get_targets(cls, metric):
        return configuration.config.getarray("parametric_evaluation",
                                             f"{metric.value.lower().replace(' ', '_')}_targets", float, fallback=[])

    @classmethod
    def evaluate_targets(cls, dataset, calculator, **kwargs):
        logger.info(f"Evaluating targets for metric '{calculator._metric.value}'...")
        targets = cls.get_targets(calculator._metric)

        results = DataFrame(np.nan, index=targets, columns=["metric_name", "number_of_families", "metric_realized"])
        results["metric_name"] = calculator._metric.value

        # Initialize OmnesDataArray for results
        results_da = OmnesDataArray(
            data=np.full((len(targets), 2), np.nan),
            dims=["target", "result_type"],
            coords={
                "target": targets,
                "result_type": ["number_of_families", "metric_realized"],
                "metric": calculator._metric.value
            },
            name=f"target_metrics_{calculator._metric.value.replace(' ', '_').lower()}"
        )

        # Evaluate number of families for each target
        for target in calculator.order_targets(targets):
            # Find number of families to achieve target (or closest match)
            nf, val = calculator.call(dataset, target_value=target,
                                      maximum_number_of_families=cls._max_number_of_households,
                                      minimum_number_of_families=cls._min_number_of_households,
                                      step_size=5)

            # Update results
            results.loc[target, ["number_of_families", "metric_realized"]] = nf, val
            results_da.loc[{"target": target}] = [nf, val]
        logger.info(f"\ntarget set; targets reached; number of families:\n{'\n'.join(f'{t:.2f}; '
                                                                                     f'{row.metric_realized:.2f}; '
                                                                                     f'{row.number_of_families}' for t, row in results.iterrows())}")
        # Expand dimensions to include metric name
        results_da = results_da.expand_dims({"metric": [calculator._metric.value]})

        return results, results_da
