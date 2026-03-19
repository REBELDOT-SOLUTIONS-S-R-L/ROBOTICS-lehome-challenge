from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path


def _iter_parser_actions(parser: argparse.ArgumentParser):
    for action in parser._actions:
        yield action
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _iter_parser_actions(subparser)


def config_mapping_to_argv(
    config_data: dict[str, object], parser: argparse.ArgumentParser
) -> list[str]:
    """Convert a parser-compatible config mapping into argv tokens."""
    parser_actions = {
        action.dest: action
        for action in _iter_parser_actions(parser)
        if getattr(action, "dest", None) not in {None, "help"}
    }

    command = None
    argv: list[str] = []
    for raw_key, raw_value in config_data.items():
        dest = str(raw_key).lstrip("-").replace("-", "_")
        if dest in {"command", "subcommand"}:
            if raw_value is not None:
                command = str(raw_value)
            continue

        action = parser_actions.get(dest)
        if action is None or raw_value is None:
            continue

        option = next(
            (
                opt
                for opt in action.option_strings
                if opt.startswith("--") and not opt.startswith("--no-")
            ),
            action.option_strings[0],
        )

        if isinstance(raw_value, bool):
            if isinstance(action, argparse.BooleanOptionalAction):
                argv.append(option if raw_value else f"--no-{dest.replace('_', '-')}")
            elif action.nargs == 0 and raw_value:
                argv.append(option)
            elif action.nargs != 0:
                argv.extend([option, str(raw_value)])
            continue

        if isinstance(raw_value, list):
            argv.append(option)
            argv.extend(str(item) for item in raw_value)
            continue

        argv.extend([option, str(raw_value)])

    return ([command] if command else []) + argv


def load_config_argv(config_path: Path, parser: argparse.ArgumentParser) -> list[str]:
    """Load additional CLI arguments from a config file."""
    suffix = config_path.suffix.lower()
    if suffix in {".txt", ".args"}:
        return shlex.split(config_path.read_text(encoding="utf-8"))

    if suffix == ".json":
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                f"YAML config requested via {config_path}, but PyYAML is not installed."
            ) from exc
        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        with config_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            config_payload = {
                str(row["key"]): row["value"]
                for row in reader
                if row.get("key")
            }
    else:
        raise SystemExit(
            f"Unsupported config format for {config_path}. Use .json, .yaml, .yml, .csv, .txt, or .args."
        )

    if isinstance(config_payload, list):
        return [str(item) for item in config_payload]
    if not isinstance(config_payload, dict):
        raise SystemExit(
            f"Config file {config_path} must contain either a mapping of arguments or a raw argv list."
        )
    return config_mapping_to_argv(config_payload, parser)


def expand_cli_args_with_config(
    argv: list[str], parser: argparse.ArgumentParser
) -> list[str]:
    """Prepend config-file arguments so explicit CLI values can still override them."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    if not config_args.config:
        return argv
    config_path = Path(config_args.config).expanduser().resolve()
    return load_config_argv(config_path, parser) + argv
