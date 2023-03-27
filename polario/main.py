"""Main CLI interface
"""
import argparse
import json
import sys
from enum import Enum
from pathlib import Path

import polars as pl

from polario import __version__


class Command(Enum):
    SHOW = "show"
    SCHEMA = "schema"
    JSON_HEAD = "json_head"


def main() -> int:
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Polario: A Python package for polarimetric data analysis"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add_argument(
        "cmd",
        choices=[c.value for c in Command],
        help="command to run",
    )
    parser.add_argument(
        "paths", metavar="PATH", type=Path, nargs="+", help="input paths"
    )
    args = parser.parse_args()
    cmd = Command(args.cmd)
    for path in args.paths:
        if path.is_dir():
            raise ValueError(
                "Input path must be a file. File an issue if you want dataset support."
            )
        df = pl.read_parquet(path, use_pyarrow=True)
        if cmd == Command.SHOW:
            print(df)
        elif cmd == Command.SCHEMA:
            print(df.schema())
        elif cmd == Command.JSON_HEAD:
            json.dump(df.head().to_dicts(), sys.stdout, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
