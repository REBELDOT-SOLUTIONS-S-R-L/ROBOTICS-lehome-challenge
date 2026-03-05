"""
Dataset management tool for operations requiring Isaac Sim (SimulationApp).

This variant routes `record` to the direct-HDF5 recorder implementation.
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from .utils import common, setup_record_parser, setup_replay_parser
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for dataset operations requiring Isaac Sim."""
    isaac_args_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(isaac_args_parser)

    parser = argparse.ArgumentParser(
        description="LeHome dataset management tool (HDF5 recorder)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    setup_record_parser(subparsers, [isaac_args_parser])
    setup_replay_parser(subparsers, [isaac_args_parser])

    args = parser.parse_args()
    simulation_app = common.launch_app_from_args(args)

    try:
        import lehome.tasks.fold_cloth  # noqa: F401
        from .utils import dataset_record_hdf5, dataset_replay, dataset_replay_hdf5

        if args.command == "record":
            dataset_record_hdf5.record_dataset(args, simulation_app)
        elif args.command == "replay":
            dataset_path = Path(args.dataset_root)
            if dataset_path.suffix.lower() in {".hdf5", ".h5"}:
                dataset_replay_hdf5.replay(args)
            else:
                dataset_replay.replay(args)
    except Exception as e:
        logger.error(f"Error during dataset {args.command}: {e}", exc_info=True)
        raise
    finally:
        common.close_app(simulation_app)


if __name__ == "__main__":
    main()
