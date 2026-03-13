import argparse
import sys
from pathlib import Path


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parametric evaluation for a configured dataset.")
    parser.add_argument("--config", help="Path to configuration .ini file.")
    args, _ = parser.parse_known_args()
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: '{config_path.as_posix()}'")
        # Keep compatibility with utility.configuration, which reads sys.argv[2].
        sys.argv = [sys.argv[0], "--config", config_path.as_posix()]
        args.config = config_path.as_posix()
    return args


_CLI_ARGS = _parse_cli_args()

from operation import initialize_operation
from utility.init_logger import init_logger

initialize_operation()
from parameteric_evaluation import initialize_evaluators

initialize_evaluators()
from parameteric_evaluation.dataset_creation import DatasetCreatorForParametricEvaluation
from parameteric_evaluation.metric_evaluator import MetricEvaluator
from utility.configuration import config

init_logger()


def run_evaluation() -> None:
    if _CLI_ARGS.config:
        print(f"Using config: {_CLI_ARGS.config}")
    DatasetCreatorForParametricEvaluation.create_dataset_for_parametric_evaluation()
    MetricEvaluator.calculate_metrics(parameters=config.get("parametric_evaluation", "evaluation_parameters"))


if __name__ == '__main__':
    run_evaluation()
