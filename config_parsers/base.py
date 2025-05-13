from abc import ABC, abstractmethod
from enum import Enum
from typing import Final, Optional, cast

from prefixed import Float
from tabulate import tabulate


def torch_dtype_width(torch_type: str) -> int:
    return 2


class TransformerMode(Enum):
    Text = 1
    Vision = 2


class Number:
    def __init__(self, unit: str, formatter: str, value: Optional[int] = None):
        self.value: Optional[int] = value
        self.unit: str = unit
        self.formatter: str = formatter

    def __str__(self) -> str:
        if self.value is not None:
            return format(Float(self.value), self.formatter) + self.unit
        else:
            return ""


class QueryConfig:
    def __init__(self, cached_tokens: int, recomputed_tokens: int, computed_tokens: int):
        self._t_mode: TransformerMode = TransformerMode.Text
        self._n_cached_tokens: int = cached_tokens
        self._n_recomputed_tokens: int = recomputed_tokens
        self._n_computed_tokens: int = computed_tokens

    @property
    def t_mode(self) -> TransformerMode:
        return self._t_mode

    @property
    def n_cached_tokens(self) -> int:
        return self._n_cached_tokens

    @property
    def n_recomputed_tokens(self) -> int:
        return self._n_recomputed_tokens

    @property
    def n_computed_tokens(self) -> int:
        return self._n_computed_tokens


class BaseModelConfigParser(ABC):
    METRIC_COMPUTE: Final[str] = "Compute"
    METRIC_BW_IPT: Final[str] = "Bandwidth (Input)"
    METRIC_BW_WGT: Final[str] = "Bandwidth (Weight)"
    METRIC_BW_OPT: Final[str] = "Bandwidth (Output)"

    def __init__(self, model_config: dict, query_config: QueryConfig):
        self.model_conf: dict = model_config
        self.query_conf: QueryConfig = query_config
        self._hw_req_by_layers: Optional[dict[str, dict[str, Number]]] = None

    @abstractmethod
    def get_layer_list(self) -> list[str]:
        pass

    @abstractmethod
    def get_num_blocks(self) -> int:
        pass

    @property
    @abstractmethod
    def hw_req_by_layers(self) -> dict[str, dict[str, Number]]:
        pass

    def get_model_type(self) -> str:
        return self.model_conf.get("model_type", "unknown")

    def new_req_dict(self) -> dict[str, Number]:
        return {
            BaseModelConfigParser.METRIC_COMPUTE: Number("FLOPs", "!.2h"),
            BaseModelConfigParser.METRIC_BW_WGT: Number("B", "!.2k"),
            BaseModelConfigParser.METRIC_BW_IPT: Number("B", "!.2k"),
            BaseModelConfigParser.METRIC_BW_OPT: Number("B", "!.2k"),
        }

    def calc_total(self) -> dict[str, dict[str, Number]]:
        n_blocks: int = self.get_num_blocks()
        req_dict: dict[str, dict[str, Number]] = self.hw_req_by_layers.copy()
        total: dict[str, Number] = self.new_req_dict()

        # Initialize all items in `total`
        for key in total.keys():
            total[key].value = 0

        # Accumulate the total values of all layers in a block
        for item in req_dict.values():
            for key in total.keys():
                if item[key].value is not None:
                    total[key].value = cast(int, total[key].value) + cast(int, item[key].value)

        # Multipli each metric with the number of blocks
        for key in total.keys():
            total[key].value = cast(int, total[key].value) * n_blocks

        # Insert `total` into `req_dict`
        req_dict[f"Total ({n_blocks} Blocks)"] = total
        return req_dict

    def print_summary(self) -> None:
        # Create a list of dictionaries representing each node's metrics.
        # Each dictionary has a "Node" key and keys from the `metrics` dict.
        rows: list[dict] = [
            {"Node": node, **metrics} for node, metrics in self.calc_total().items()
        ]

        # Convert all numerical values in the rows to strings for uniform formatting
        rows = [{k: str(v) if isinstance(v, Number) else v for k, v in row.items()} for row in rows]

        # Insert a blank row (with empty Node and metrics) before the last row
        rows = rows[0:-1] + [{"Node": "", **self.new_req_dict()}] + [rows[-1]]

        # Set column alignment: "Node" column is left-aligned; metric columns are right-aligned
        colalign = ["left"] + ["right"] * (len(self.new_req_dict()))

        # Print the table using tabulate with GitHub-style formatting
        print(tabulate(rows, headers="keys", tablefmt="github", colalign=colalign))
