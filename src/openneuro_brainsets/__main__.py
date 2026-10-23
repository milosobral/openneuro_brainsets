"""Small helper CLI: ``python -m openneuro_brainsets {list,path}``.

Until torch_brain's ``brainsets`` CLI discovers plugins, ``path`` lets you run a
pipeline with ``brainsets prepare <path> --local --use-active-env``.
"""

import argparse

from .pipelines import NAMESPACE, get_pipeline_dir, list_brainsets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m openneuro_brainsets")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List the brainsets provided by this plugin.")
    path_parser = sub.add_parser(
        "path", help="Print the pipeline directory of a brainset."
    )
    path_parser.add_argument("brainset_id")
    args = parser.parse_args(argv)

    if args.command == "list":
        for brainset_id in list_brainsets():
            print(f"{NAMESPACE}:{brainset_id}")
    elif args.command == "path":
        print(get_pipeline_dir(args.brainset_id))


if __name__ == "__main__":
    main()
