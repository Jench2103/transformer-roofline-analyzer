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

    query_config: QueryConfig = QueryConfig(
        args.cached_tokens, args.recomputed_tokens, args.computed_tokens
    )

    parser = parser_cls(model_conf, query_config)
    parser.print_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to model config.json")
    parser.add_argument("--cached-tokens", type=int, default=1024, required=False, help="")
    parser.add_argument("--recomputed-tokens", type=int, default=0, required=False, help="")
    parser.add_argument("--computed-tokens", type=int, default=1, required=False, help="")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    compute_roofline_metrics(config, args)


if __name__ == "__main__":
    main()
