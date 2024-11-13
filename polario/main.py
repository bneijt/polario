"""Main CLI interface"""

import argparse
import json
import sys
from enum import Enum
from functools import reduce
from pathlib import Path
from pprint import pprint

import polars as pl

from polario import __version__


class Command(Enum):
    SHOW = "show"
    SCHEMA = "schema"
    JSON_HEAD = "json_head"
    JSONL = "jsonl"
    CONCAT_CSV = "concat_csv"
    WRITE_CSV = "write_csv"


def main() -> int:
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="polario library commandline tool to inspect Parquet files"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "cmd",
        choices=[c.value for c in Command],
        help="command to run",
    )
    parser.add_argument(
        "paths",
        metavar="PATH",
        type=Path,
        nargs="+",
        help="input paths",
    )
    args = parser.parse_args()
    cmd = Command(args.cmd)

    if cmd == Command.CONCAT_CSV:
        df = reduce(
            lambda a, b: pl.concat([a, pl.read_csv(b, infer_schema_length=0)]),
            args.paths[1:],
            pl.read_csv(args.paths[0], infer_schema_length=0),
        )
        print(df)
        output_filename = Path(args.paths[0].stem + ".parquet")
        if output_filename.exists():
            raise ValueError(f"Output file {output_filename} already exists")
        print("Writing to", output_filename)
        df.write_parquet(output_filename)
        return 0
    paths: list[Path] = args.paths
    for path in paths:
        if path.is_dir():
            raise ValueError(
                "Input path must be a file. File an issue if you want dataset support."
            )
        df = pl.read_parquet(path, use_pyarrow=True)
        if cmd == Command.SHOW:
            print(df)
        elif cmd == Command.SCHEMA:
            pprint(df.schema)
        elif cmd == Command.JSON_HEAD:
            json.dump(df.head().to_dicts(), sys.stdout, indent=2)
        elif cmd == Command.JSONL:
            for row in df.to_dicts():
                json.dump(row, sys.stdout, separators=(",", ":"))
                sys.stdout.write("\n")
        elif cmd == Command.WRITE_CSV:
            output_path = Path(path.name).with_suffix(".csv")
            if output_path.exists():
                print(f"Output file {output_path} already exists")
                continue
            df.write_csv(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
