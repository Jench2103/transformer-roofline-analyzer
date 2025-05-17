#!/usr/bin/env python3

import argparse
import json

from config_parsers.base import QueryConfig
from config_parsers.llama4 import Llama4ConfigParser

PARSER_REGISTRY = {
    "llama4": Llama4ConfigParser,
    # Extend here for other model types
}


def compute_roofline_metrics(model_conf: dict, args: argparse.Namespace) -> None:
    model_type: str = model_conf.get("model_type", "").lower()

    parser_cls = PARSER_REGISTRY.get(model_type)
    if parser_cls is None:
        raise NotImplementedError(f"No parser for model_type: {model_type}")

    if len(args.cached_tokens) != len(args.computed_tokens):
        raise ValueError(
            "`--cached-tokens` and `--computed-tokens` must have the same number of elements."
        )

    if (args.batch_size is not None) and (args.batch_size % len(args.cached_tokens) != 0):
        raise ValueError(
            "`--batch-size` must be a multiple of the elements in `--cached-tokens` and `--computed-tokens`."
        )

    cached_tokens: list[int] = args.cached_tokens * (
        int(args.batch_size / len(args.cached_tokens)) if args.batch_size is not None else 1
    )
    computed_tokens: list[int] = args.computed_tokens * (
        int(args.batch_size / len(args.computed_tokens)) if args.batch_size is not None else 1
    )

    query_config: QueryConfig = QueryConfig(cached_tokens, computed_tokens)

    parser = parser_cls(model_conf, query_config)
    parser.print_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to model config.json")
    parser.add_argument(
        "--cached-tokens",
        type=int,
        nargs="+",
        default=[0],
        required=False,
        help="List of token counts already present in the KV-cache for each query in the batch (e.g., cached context tokens used during attention).",
    )
    parser.add_argument(
        "--computed-tokens",
        type=int,
        nargs="+",
        default=[1],
        required=False,
        help="List of token counts to compute and store in the KV-cache for each query in the batch (e.g., prompt or generated tokens).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        help="The number of queries in a batch.",
    )

    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    compute_roofline_metrics(config, args)


if __name__ == "__main__":
    main()
