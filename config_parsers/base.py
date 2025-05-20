from abc import ABC, abstractmethod
from enum import Enum
from typing import Final, Optional, cast

from prefixed import Float
from tabulate import tabulate


def torch_dtype_width(torch_type: str) -> int:
    """
    Reference: https://docs.pytorch.org/docs/stable/tensors.html#data-types
    """

    match torch_type:
        # Integer
        case "uint8" | "int8" | "quint8" | "qint8":
            return 1
        case "uint16" | "int16" | "short":
            return 2
        case "uint32" | "int32" | "int" | "qint32":
            return 4
        case "uint64" | "int64" | "long":
            return 8

        # Floating point
        case "float8_e4m3fn" | "float8_e5m2":
            return 1
        case "float16" | "half" | "bfloat16":
            return 2
        case "float32" | "float":
            return 4
        case "float64" | "double":
            return 8

        case _:
            raise ValueError(f"Unsupported torch data type: `{torch_type}`.")


def act_flops(act: str) -> int:
    match act:
        # https://github.com/vllm-project/vllm/blob/84ab4feb7e994ee6c692957e6d80a528af072e49/csrc/activation_kernels.cu#L35
        case "silu":
            return 4

        case _:
            raise ValueError(f"Unsupported activation function: `{act}`.")


class TransformerMode(Enum):
    Text = 1
    Vision = 2


class Number:
    def __init__(self, unit: str, formatter: str, value: Optional[float] = None):
        self.value: Optional[float] = value
        self.unit: str = unit
        self.formatter: str = formatter

    def __str__(self) -> str:
        if self.value is not None:
            return format(Float(self.value), self.formatter) + self.unit
        else:
            return ""


class QueryConfig:
    def __init__(self, cached_tokens: list[int], input_tokens: list[int]):
        self._t_mode: TransformerMode = TransformerMode.Text
        self._n_cached_tokens: list[int] = cached_tokens
        self._n_input_tokens: list[int] = input_tokens

    @property
    def t_mode(self) -> TransformerMode:
        return self._t_mode

    @property
    def n_cached_tokens(self) -> list[int]:
        return self._n_cached_tokens

    @property
    def n_input_tokens(self) -> list[int]:
        return self._n_input_tokens


class BaseModelConfigParser(ABC):
    METRIC_COMPUTE: Final[str] = "Compute"
    METRIC_BW_IPT: Final[str] = "Bandwidth (Input)"
    METRIC_BW_WGT: Final[str] = "Bandwidth (Weight)"
    METRIC_BW_OPT: Final[str] = "Bandwidth (Output)"
    METRIC_OI: Final[str] = "Operational Intensity"

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

    @abstractmethod
    def get_layer_num_blocks(self, layer: str) -> int:
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

    def set_op_proj_req(
        self,
        req: dict[str, Number],
        dim_m: int,
        dim_n: int,
        dim_k: int,
        torch_dtype: str,
    ) -> None:
        metric_compute: int = (
            int(cast(float, req[BaseModelConfigParser.METRIC_COMPUTE].value))
            if req[BaseModelConfigParser.METRIC_COMPUTE].value is not None
            else 0
        )
        metric_bw_wgt: int = (
            int(cast(float, req[BaseModelConfigParser.METRIC_BW_WGT].value))
            if req[BaseModelConfigParser.METRIC_BW_WGT].value is not None
            else 0
        )
        metric_bw_ipt: int = (
            int(cast(float, req[BaseModelConfigParser.METRIC_BW_IPT].value))
            if req[BaseModelConfigParser.METRIC_BW_IPT].value is not None
            else 0
        )
        metric_bw_opt: int = (
            int(cast(float, req[BaseModelConfigParser.METRIC_BW_OPT].value))
            if req[BaseModelConfigParser.METRIC_BW_OPT].value is not None
            else 0
        )

        metric_compute += dim_m * dim_n * (dim_k * 2 - 1)
        metric_bw_wgt += (dim_k * dim_n) * torch_dtype_width(torch_dtype)
        metric_bw_ipt += (dim_m * dim_k) * torch_dtype_width(torch_dtype)
        metric_bw_opt += (dim_m * dim_n) * torch_dtype_width(torch_dtype)

        req[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        req[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        req[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        req[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt

    def set_op_sum_req(
        self, layer_entry: dict[str, Number], num_elem: int, num_tensors: int, torch_dtype: str
    ) -> None:
        metric_compute: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value))
            if layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value is not None
            else 0
        )
        metric_bw_wgt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value is not None
            else 0
        )
        metric_bw_ipt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value is not None
            else 0
        )
        metric_bw_opt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value is not None
            else 0
        )

        metric_compute += num_elem * (num_tensors - 1)
        metric_bw_ipt += num_elem * torch_dtype_width(torch_dtype) * num_tensors
        metric_bw_opt += num_elem * torch_dtype_width(torch_dtype)

        layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt

    def set_op_rope_req(
        self, layer_entry: dict[str, Number], token_dims: int, n_tokens: int, torch_dtype: str
    ) -> None:
        """
        In summary, there are 3 FLOPs on average for each element in a token representation.

        References:
        - Llama reference implementation: https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L132
        - vLLM implementation: https://github.com/vllm-project/vllm/blob/dc1440cf9f8f6233a3c464e1a01daa12207f8680/csrc/pos_encoding_kernels.cu#L37
        """

        metric_compute: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value))
            if layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value is not None
            else 0
        )
        metric_bw_wgt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value is not None
            else 0
        )
        metric_bw_ipt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value is not None
            else 0
        )
        metric_bw_opt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value is not None
            else 0
        )

        metric_compute += token_dims * 3 * n_tokens
        metric_bw_ipt += token_dims * n_tokens * torch_dtype_width(torch_dtype)
        metric_bw_opt += token_dims * n_tokens * torch_dtype_width(torch_dtype)

        layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt

    def set_op_rmsnorm_req(
        self, layer_entry: dict[str, Number], hidden_size: int, n_tokens: int, torch_dtype: str
    ) -> None:
        """
        Example Implementation from Llama:

        ```python
        def rmsnorm(x, eps):
            def _norm(y):
                return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

            return _norm(x.float()).type_as(x)

        class RMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                return rmsnorm(x, self.eps) * self.weight
        ```

        FLOPs Breakdown (assumming hidden_size = d):
        - Square the input values: d
        - Sum the squared values: d - 1
        - Calculate average and add constant epsilon: 2
        - Square root: 1
        - Divide each element by RMS (normalization): d
        - Multiply by constant gamma: d

        References:
        - PyTorch Document: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
        - Llama Reference Implementation: https://github.com/meta-llama/llama-models/blob/f3d16d734f4de7d5bb7427705399e350da5e200f/models/llama4/model.py#L27
        """

        metric_compute: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value))
            if layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value is not None
            else 0
        )
        metric_bw_wgt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value is not None
            else 0
        )
        metric_bw_ipt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value is not None
            else 0
        )
        metric_bw_opt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value is not None
            else 0
        )

        metric_compute += (hidden_size * 4 + 2) * n_tokens
        metric_bw_wgt += (hidden_size + 1) * torch_dtype_width(torch_dtype)
        metric_bw_ipt += hidden_size * n_tokens * torch_dtype_width(torch_dtype)
        metric_bw_opt += hidden_size * n_tokens * torch_dtype_width(torch_dtype)

        layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt

    def set_op_actmul_req(
        self,
        layer_entry: dict[str, Number],
        intermediate_size: int,
        n_tokens: int,
        act_flops: int,
        torch_dtype: str,
    ) -> None:
        """
        Fuse the activation function (i.e., SiLU) and element-wise multiplication against the outputs of Gate/Up projection in Llama FFN:

        ```python
        class FeedForward(nn.Module):
            def forward(self, x):
                x = F.silu(F.linear(x, self.w1.weight)) * F.linear(x, self.w3.weight)
                out = F.linear(x, self.w2.weight)
                if self.do_reduce:
                    return reduce_from_model_parallel_region(out)
                return out
        ```

        Reference implementation in vLLM:

        ```cpp
        template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
                  bool act_first>
        __device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                                    const scalar_t& y) {
          return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
        }
        // Activation and gating kernel template.

        template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
                  bool act_first>
        __global__ void act_and_mul_kernel(
            scalar_t* __restrict__ out,          // [..., d]
            const scalar_t* __restrict__ input,  // [..., 2, d]
            const int d) {
          const int64_t token_idx = blockIdx.x;
          for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
            const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
            out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
          }
        }
        ```

        References:
        - Llama Reference Implementation: https://github.com/meta-llama/llama-models/blob/f3d16d734f4de7d5bb7427705399e350da5e200f/models/llama4/ffn.py#L47
        - vLLM Implementation: https://github.com/vllm-project/vllm/blob/84ab4feb7e994ee6c692957e6d80a528af072e49/csrc/activation_kernels.cu#L12
        """

        metric_compute: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value))
            if layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value is not None
            else 0
        )
        metric_bw_wgt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value is not None
            else 0
        )
        metric_bw_ipt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value is not None
            else 0
        )
        metric_bw_opt: int = (
            int(cast(float, layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value))
            if layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value is not None
            else 0
        )

        metric_compute += (act_flops + 1) * intermediate_size + n_tokens
        metric_bw_ipt += intermediate_size * n_tokens * 2 * torch_dtype_width(torch_dtype)
        metric_bw_opt += intermediate_size * n_tokens * torch_dtype_width(torch_dtype)

        layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt

    def calc_total(self) -> dict[str, dict[str, Number]]:
        n_blocks: int = self.get_num_blocks()
        req_dict: dict[str, dict[str, Number]] = self.hw_req_by_layers.copy()
        total: dict[str, Number] = self.new_req_dict()

        # Initialize all items in `total`
        for key in total.keys():
            total[key].value = 0

        # Accumulate the total values of all layers in a block
        for layer, item in req_dict.items():
            for key in total.keys():
                if item[key].value is not None:
                    total[key].value = cast(int, total[key].value) + cast(
                        int, item[key].value
                    ) * self.get_layer_num_blocks(layer)

        # Insert `total` into `req_dict`
        req_dict[f"Total ({n_blocks} Blocks)"] = total
        return req_dict

    def calc_roofline(self, req_dict: dict[str, dict[str, Number]]) -> dict[str, dict[str, Number]]:
        req_dict_cp = req_dict.copy()

        for metrics in req_dict_cp.values():
            metrics[BaseModelConfigParser.METRIC_OI] = Number("FLOPs/Bytes", "!.2h")

            if (
                metrics[BaseModelConfigParser.METRIC_COMPUTE].value is not None
                and metrics[BaseModelConfigParser.METRIC_BW_WGT].value is not None
                and metrics[BaseModelConfigParser.METRIC_BW_IPT].value is not None
            ):
                compute_req: int = cast(int, metrics[BaseModelConfigParser.METRIC_COMPUTE].value)
                bandwidth_req: int = cast(
                    int, metrics[BaseModelConfigParser.METRIC_BW_WGT].value
                ) + cast(int, metrics[BaseModelConfigParser.METRIC_BW_IPT].value)
                metrics[BaseModelConfigParser.METRIC_OI].value = compute_req / bandwidth_req

        return req_dict_cp

    def print_summary(self) -> None:
        # Create a list of dictionaries representing each node's metrics.
        # Each dictionary has a "Node" key and keys from the `metrics` dict.
        rows: list[dict] = [
            {"Node": node, **metrics}
            for node, metrics in self.calc_roofline(self.calc_total()).items()
        ]

        # Convert all numerical values in the rows to strings for uniform formatting
        rows = [{k: str(v) if isinstance(v, Number) else v for k, v in row.items()} for row in rows]

        # Insert a blank row (with empty Node and metrics) before the last row
        rows = rows[0:-1] + [{"Node": "", **self.new_req_dict()}] + [rows[-1]]

        # Set column alignment: "Node" column is left-aligned; metric columns are right-aligned
        colalign = ["left"] + ["right"] * (len(self.new_req_dict()) + 1)

        # Print the table using tabulate with GitHub-style formatting
        print(tabulate(rows, headers="keys", tablefmt="github", colalign=colalign))
