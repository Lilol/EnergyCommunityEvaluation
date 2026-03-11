from enum import Enum

from utility.definitions import OrderedEnum


class Parameter(OrderedEnum):
    def to_abbrev_str(self):
        abbrev_dictionary = self._get_abbrev_mapping()
        return abbrev_dictionary.get(self, None)

    @classmethod
    def _get_abbrev_mapping(cls):
        raise NotImplementedError("Subclasses must implement _get_abbrev_mapping")

    @classmethod
    def get_all(cls):
        return cls.__members__.values()

    def valid(self):
        return self.value != "invalid"


class PhysicalMetric(Parameter):
    SHARED_ENERGY = "Shared energy"
    TOTAL_CONSUMPTION = "Total consumption"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.SHARED_ENERGY: "e_sh", cls.TOTAL_CONSUMPTION: "c_tot"}


class OtherParameters(Parameter):
    INJECTED_ENERGY = "Injected energy"
    WITHDRAWN_ENERGY = "Withdrawn energy"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.INJECTED_ENERGY: "e_inj", cls.WITHDRAWN_ENERGY: "e_with"}


class BatteryPowerFlows(Parameter):
    STORED_ENERGY = "Stored energy"
    POWER_CHARGE = "Charging power"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.STORED_ENERGY: "e_stor", cls.POWER_CHARGE: "p_charge"}


class EnvironmentalMetric(Parameter):
    BASELINE_EMISSIONS = "Baseline emissions"
    TOTAL_EMISSIONS = "Total emissions"
    ESR = "Emissions savings ratio"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.ESR: "esr", cls.TOTAL_EMISSIONS: "em_tot", cls.BASELINE_EMISSIONS: "e_base", }


class EconomicMetric(Parameter):
    CAPEX = "Capex"
    OPEX = "Opex"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.CAPEX: "capex", cls.OPEX: "opex"}


class LoadMatchingMetric(Parameter):
    SELF_CONSUMPTION = "Self consumption"
    SELF_SUFFICIENCY = "Self sufficiency"
    SELF_PRODUCTION = "Self production"
    GRID_LIABILITY = "Grid liability"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.SELF_CONSUMPTION: "sc", cls.SELF_SUFFICIENCY: "ss", }


class TimeAggregation(Parameter):
    THEORETICAL_LIMIT = "15min"
    HOUR = "hour"
    DAY = "dayofyear"
    MONTH = 'month'
    SEASON = 'season'
    YEAR = "year"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.HOUR: "hour", cls.MONTH: "month", cls.DAY: "month", cls.YEAR: "year",
                cls.THEORETICAL_LIMIT: "th_lim", }


class ParametricEvaluationType(OrderedEnum):
    DATASET_CREATION = "dataset_creation"
    METRIC_TARGETS = "metric_targets"
    TIME_AGGREGATION = "time_aggregation"
    PHYSICAL_METRICS = "physical"
    ECONOMIC_METRICS = "economic"
    ENVIRONMENTAL_METRICS = "environmental"
    LOAD_MATCHING_METRICS = "load_matching"
    ALL = "all"
    INVALID = "invalid"


def make_combined_enum(name, first_enum, second_enum, base_cls=Parameter):
    members = {}
    abbrev_map = {}

    for f in first_enum:
        if not f.valid():
            continue
        for s in second_enum:
            if not s.valid():
                continue
            enum_name = f"{f.name}_{s.name}"
            members[enum_name] = (f, s)
            abbrev_map[(f, s)] = f"{f.to_abbrev_str()}_{s.to_abbrev_str()}"

    # Custom metaclass to override value
    class _CombinedEnum(base_cls, Enum):
        def __new__(cls, first, second):
            obj = object.__new__(cls)
            obj._pair = (first, second)  # store tuple internally
            # The "real" value of the Enum will be a string
            obj._value_ = f"{second.value.title()} with '{first.value.title()}' resolution"
            return obj

        @property
        def first(self):
            return self._pair[0]

        @property
        def second(self):
            return self._pair[1]

        @classmethod
        def _get_abbrev_mapping(cls):
            return abbrev_map

        @classmethod
        def from_parts(cls, first, second):
            name = f"{first.name}_{second.name}"
            return cls[name]

    # Actually build the enum using the custom class
    return _CombinedEnum(name, members)

# --- Usage ---
CombinedMetricEnum = make_combined_enum(
    "CombinedMetricEnum",
    TimeAggregation,
    LoadMatchingMetric
)
