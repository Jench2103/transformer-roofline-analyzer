from abc import ABC, abstractmethod
from typing import Final, Optional

from tabulate import tabulate

from .utils import Number, QueryConfig, torch_dtype_width


class BaseModelConfigParser(ABC):
    """
    Abstract base class for parsing and analyzing transformer model configurations
    with respect to their hardware resource requirements.

    This class defines a framework for extracting layer-wise compute and memory
    bandwidth statistics (e.g., FLOPs, input/output/weight bandwidths) from a
    HuggingFace-style model configuration and query configuration. These statistics
    are used to estimate roofline performance metrics and operational intensity
    (FLOPs per byte), which are helpful for performance modeling and optimization
    on various hardware platforms.

    Subclasses must implement:
        - `get_layer_list`: Returns a list of layer names to include in the analysis.
        - `get_num_blocks`: Returns the total number of transformer blocks in the model.
        - `get_layer_num_blocks`: Returns how many transformer blocks a given layer appears in.
        - `hw_req_by_layers`: Returns a dictionary mapping each layer name to its corresponding hardware metrics.

    Important:
        The `hw_req_by_layers` property **must** also ensure that the internal
        `_hw_req_by_layers` cache is constructed and populated if it is `None`.
        This lazy initialization ensures metrics are computed only once and reused
        for further aggregation or summary reporting.

    Key Features:
        - Defines standard metrics for hardware evaluation: compute (FLOPs),
          bandwidth (input, weight, output), and operational intensity (OI).
        - Provides utility functions to calculate these metrics for common
          transformer operations like projection, summation, RoPE, RMSNorm, and
          activation-gating patterns (e.g., in LLaMA FFNs).
        - Aggregates layer-wise metrics to compute total model requirements.
        - Computes OI and formats results in a tabulated summary for easy inspection.

    Attributes:
        model_conf (dict): The HuggingFace-style model configuration.
        query_conf (QueryConfig): The query configuration defining model usage.
        _hw_req_by_layers (Optional[dict]): Cached dictionary mapping each layer
                                            to its hardware metrics.
    """

    METRIC_NUM_BLOCKS: Final[str] = "Block Count"
    METRIC_COMPUTE: Final[str] = "Compute"
    METRIC_BW_IPT: Final[str] = "Bandwidth (Input)"
    METRIC_BW_WGT: Final[str] = "Bandwidth (Weight)"
    METRIC_BW_OPT: Final[str] = "Bandwidth (Output)"
    METRIC_OI: Final[str] = "Operational Intensity"

    def __init__(self, model_config: dict, query_config: QueryConfig):
        self.model_conf: dict = model_config
        self.query_conf: QueryConfig = query_config
        self._hw_req_by_layers: Optional[dict[str, dict[str, Number]]] = None

    @classmethod
    def normalize_config(cls, config_dict: dict) -> dict:
        """
        Normalize config by setting default values if needed.

        Subclasses should override this method to handle architecture-specific
        defaults (e.g., setting torch_dtype in the appropriate location).

        Args:
            config_dict: Raw config dictionary from HuggingFace or local file.

        Returns:
            Normalized config dictionary with defaults applied.
        """
        return config_dict

    @abstractmethod
    def get_layer_list(self) -> list[str]:
        """
        Returns a list of all layer names to be displayed in the output table.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_blocks(self) -> int:
        """
        Returns the total number of transformer blocks present in the model.
        """
        raise NotImplementedError()

    def get_layer_num_blocks(self, layer: str) -> int:
        """
        Returns the number of transformer blocks that include the specified layer.

        Args:
            layer (str): The name of the layer to query.

        Returns:
            int: Number of transformer blocks containing the layer.
        """
        return self.get_num_blocks()

    @abstractmethod
    def get_kvcache_size(self) -> int:
        """
        Returns the estimated total size of the key-value (KV) cache used during inference
        for the entire model, based on the configured context length and batch size.

        The KV cache stores intermediate attention keys and values for each transformer layer,
        and its memory footprint grows proportionally with the number of layers, attention heads,
        hidden dimensions, batch size, and context length.

        Returns:
            int: Total size of the KV cache in bytes.
        """
        raise NotImplementedError()

    def get_extra_storage_req(self) -> list[tuple[str, Number]]:
        """
        Returns a list of additional storage requirements specific to the model configuration.

        This function computes extra memory needs beyond the standard model weights and KV-cache.
        These may include, for example, additional expert weights used in Mixture-of-Experts (MoE)
        transformer variants, depending on the model's mode (e.g., text or vision) and its internal
        architectural parameters.

        Each returned item is a tuple of the form (description, size), where:
            - `description` is a string describing the type of additional storage (e.g., "Additional Expert Weights")
            - `size` is a `Number` instance representing the size in bytes.

        Note:
            All `Number` instances in the returned list must be initialized with:
                - unit = "B"         # for bytes
                - formatter = "!.2k" # for human-readable formatting with two decimal places and unit suffixes (e.g., KiB, MiB)

        Returns:
            list[tuple[str, Number]]: A list of (description, Number) pairs representing extra storage needs.
        """
        return []

    @property
    @abstractmethod
    def hw_req_by_layers(self) -> dict[str, dict[str, Number]]:
        """
        Provides a mapping of each layer to its corresponding hardware resource metrics (e.g., compute, bandwidth).

        Returns:
            dict[str, dict[str, Number]]: A dictionary where keys are layer names and values are dictionaries of metric names and their corresponding values.
        """
        raise NotImplementedError()

    def get_model_type(self) -> str:
        """
        Retrieves the model type specified in the HuggingFace configuration.

        Returns:
            str: The model type as defined in the configuration, or "unknown" if not specified.
        """
        return self.model_conf.get("model_type", "unknown")

    def new_req_dict(self) -> dict[str, Number]:
        """
        Initializes a new dictionary for storing hardware metrics for a single layer.

        Returns:
            dict[str, Number]: A dictionary with keys for each metric initialized with
            default `Number` objects.
        """
        return {
            BaseModelConfigParser.METRIC_NUM_BLOCKS: Number("", ""),
            BaseModelConfigParser.METRIC_COMPUTE: Number("FLOPs", "!.2h"),
            BaseModelConfigParser.METRIC_BW_WGT: Number("B", "!.2k"),
            BaseModelConfigParser.METRIC_BW_IPT: Number("B", "!.2k"),
            BaseModelConfigParser.METRIC_BW_OPT: Number("B", "!.2k"),
        }

    def set_op_proj_req(
        self,
        layer_entry: dict[str, Number],
        dim_m: int,
        dim_n: int,
        dim_k: int,
        torch_dtype: str,
    ) -> None:
        metric_compute: int = layer_entry[BaseModelConfigParser.METRIC_COMPUTE].get_value_int()
        metric_bw_wgt: int = layer_entry[BaseModelConfigParser.METRIC_BW_WGT].get_value_int()
        metric_bw_ipt: int = layer_entry[BaseModelConfigParser.METRIC_BW_IPT].get_value_int()
        metric_bw_opt: int = layer_entry[BaseModelConfigParser.METRIC_BW_OPT].get_value_int()

        metric_compute += dim_m * dim_n * (dim_k * 2 - 1)
        metric_bw_wgt += (dim_k * dim_n) * torch_dtype_width(torch_dtype)
        metric_bw_ipt += (dim_m * dim_k) * torch_dtype_width(torch_dtype)
        metric_bw_opt += (dim_m * dim_n) * torch_dtype_width(torch_dtype)

        layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt

    def set_op_sum_req(
        self, layer_entry: dict[str, Number], num_elem: int, num_tensors: int, torch_dtype: str
    ) -> None:
        metric_compute: int = layer_entry[BaseModelConfigParser.METRIC_COMPUTE].get_value_int()
        metric_bw_wgt: int = layer_entry[BaseModelConfigParser.METRIC_BW_WGT].get_value_int()
        metric_bw_ipt: int = layer_entry[BaseModelConfigParser.METRIC_BW_IPT].get_value_int()
        metric_bw_opt: int = layer_entry[BaseModelConfigParser.METRIC_BW_OPT].get_value_int()

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

        metric_compute: int = layer_entry[BaseModelConfigParser.METRIC_COMPUTE].get_value_int()
        metric_bw_wgt: int = layer_entry[BaseModelConfigParser.METRIC_BW_WGT].get_value_int()
        metric_bw_ipt: int = layer_entry[BaseModelConfigParser.METRIC_BW_IPT].get_value_int()
        metric_bw_opt: int = layer_entry[BaseModelConfigParser.METRIC_BW_OPT].get_value_int()

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

        metric_compute: int = layer_entry[BaseModelConfigParser.METRIC_COMPUTE].get_value_int()
        metric_bw_wgt: int = layer_entry[BaseModelConfigParser.METRIC_BW_WGT].get_value_int()
        metric_bw_ipt: int = layer_entry[BaseModelConfigParser.METRIC_BW_IPT].get_value_int()
        metric_bw_opt: int = layer_entry[BaseModelConfigParser.METRIC_BW_OPT].get_value_int()

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

        metric_compute: int = layer_entry[BaseModelConfigParser.METRIC_COMPUTE].get_value_int()
        metric_bw_wgt: int = layer_entry[BaseModelConfigParser.METRIC_BW_WGT].get_value_int()
        metric_bw_ipt: int = layer_entry[BaseModelConfigParser.METRIC_BW_IPT].get_value_int()
        metric_bw_opt: int = layer_entry[BaseModelConfigParser.METRIC_BW_OPT].get_value_int()

        metric_compute += (act_flops + 1) * intermediate_size + n_tokens
        metric_bw_ipt += intermediate_size * n_tokens * 2 * torch_dtype_width(torch_dtype)
        metric_bw_opt += intermediate_size * n_tokens * torch_dtype_width(torch_dtype)

        layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt

    def set_op_sdpa_req(
        self,
        layer_entry: dict[str, Number],
        tensor_qo_dims: int,
        tensor_kv_dims: int,
        torch_dtype: str,
    ) -> None:
        """
        Compute Scaled Dot-Product Attention (SDPA) hardware requirements.

        This method calculates the compute and bandwidth requirements for the attention
        operation, accounting for the KV-cache when processing cached tokens.

        Args:
            layer_entry: The layer metrics dictionary to update.
            tensor_qo_dims: Dimensions for query/output tensors (typically hidden_size).
            tensor_kv_dims: Dimensions for key/value tensors (head_dim * num_kv_heads).
            torch_dtype: Data type for bandwidth calculations.

        FLOPs Breakdown:
        - GEMM P = QK^T: qo_seq_len * kv_seq_len * (tensor_qo_dims * 2 - 1)
        - GEMM O = SV:   qo_seq_len * tensor_kv_dims * (kv_seq_len * 2 - 1)

        References:
        - PyTorch SDPA: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        """
        batch_size: int = len(self.query_conf.n_cached_tokens)

        metric_compute: int = 0
        metric_bw_wgt: int = 0
        metric_bw_ipt: int = 0
        metric_bw_opt: int = 0

        for query_idx in range(batch_size):
            qo_seq_len: int = self.query_conf.n_input_tokens[query_idx]
            kv_seq_len: int = (
                self.query_conf.n_cached_tokens[query_idx]
                + self.query_conf.n_input_tokens[query_idx]
            )

            tensor_qo_size: int = qo_seq_len * tensor_qo_dims * torch_dtype_width(torch_dtype)
            tensor_kv_size: int = kv_seq_len * (tensor_kv_dims * 2) * torch_dtype_width(torch_dtype)

            metric_bw_ipt += tensor_qo_size + tensor_kv_size
            metric_bw_opt += tensor_qo_size

            # GEMM: P = QK^T
            metric_compute += qo_seq_len * kv_seq_len * (tensor_qo_dims * 2 - 1)

            # GEMM: O = SV
            metric_compute += qo_seq_len * tensor_kv_dims * (kv_seq_len * 2 - 1)

        layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt

    def calc_total(self) -> dict[str, dict[str, Number]]:
        """
        Computes the aggregate roofline metrics across the entire model, summing values from all layers and adjusting for the number of transformer blocks.

        Returns:
            dict[str, dict[str, Number]]: A dictionary of metrics per layer, including a summary
            entry labeled "Total (<n> Blocks)".
        """

        n_blocks: int = self.get_num_blocks()
        req_dict: dict[str, dict[str, Number]] = self.hw_req_by_layers.copy()
        total: dict[str, Number] = self.new_req_dict()

        # Remove the layers that do not present in any blocks
        not_present_layers: list[str] = [
            layer for layer in req_dict.keys() if self.get_layer_num_blocks(layer) == 0
        ]
        for layer in not_present_layers:
            req_dict.pop(layer)

        # Initialize all items in `total`
        for key in total.keys():
            total[key].value = 0

        # Accumulate the total values of all layers in a block
        for layer, item in req_dict.items():
            for key in total.keys():
                total[key].value = total[key].get_value_int() + (
                    item[key].get_value_int() * self.get_layer_num_blocks(layer)
                )

        # Insert `total` into `req_dict`
        req_dict[f"Total ({n_blocks} Blocks)"] = total
        return req_dict

    def calc_roofline(self, req_dict: dict[str, dict[str, Number]]) -> dict[str, dict[str, Number]]:
        """
        Computes the operational intensity (FLOPs per byte) for each layer using compute and bandwidth metrics.

        Args:
            req_dict (dict[str, dict[str, Number]]): Dictionary of hardware metrics per layer.

        Returns:
            dict[str, dict[str, Number]]: Updated dictionary with operational intensity included.
        """
        req_dict_cp = req_dict.copy()

        for metrics in req_dict_cp.values():
            metrics[BaseModelConfigParser.METRIC_OI] = Number("FLOPs/Bytes", "!.2h")

            if (
                metrics[BaseModelConfigParser.METRIC_COMPUTE].value is not None
                and metrics[BaseModelConfigParser.METRIC_BW_WGT].value is not None
                and metrics[BaseModelConfigParser.METRIC_BW_IPT].value is not None
                and metrics[BaseModelConfigParser.METRIC_BW_OPT].value is not None
            ):
                compute_req: int = metrics[BaseModelConfigParser.METRIC_COMPUTE].get_value_int()
                bandwidth_req: int = (
                    metrics[BaseModelConfigParser.METRIC_BW_WGT].get_value_int()
                    + metrics[BaseModelConfigParser.METRIC_BW_IPT].get_value_int()
                    + metrics[BaseModelConfigParser.METRIC_BW_OPT].get_value_int()
                )
                metrics[BaseModelConfigParser.METRIC_OI].value = compute_req / bandwidth_req

        return req_dict_cp

    def print_summary(self) -> None:
        """
        Displays a formatted summary table of the model's hardware metrics and computed roofline statistics for each layer and the overall model.
        """

        # Create a list of dictionaries representing each node's metrics.
        # Each dictionary has a "Node" key and keys from the `metrics` dict.
        rows: list[dict] = [
            {"Node": node, **metrics}
            for node, metrics in self.calc_roofline(self.calc_total()).items()
        ]

        # Calculate and show the number of transformer blocks that contain each layer
        for layer_row in rows:
            if layer_row["Node"] != "" and "Total" not in layer_row["Node"]:
                layer_row["Block Count"] = (
                    f"{self.get_layer_num_blocks(layer_row['Node'])} / {self.get_num_blocks()}"
                )
            else:
                layer_row["Block Count"] = "N/A"

        # Calculate storage capacity requirement
        wgt_size: Number = rows[-1][BaseModelConfigParser.METRIC_BW_WGT]
        kv_size: Number = Number("B", "!.2k", self.get_kvcache_size())

        # Convert all numerical values in the rows to strings for uniform formatting
        rows = [{k: str(v) if isinstance(v, Number) else v for k, v in row.items()} for row in rows]

        # Insert a blank row (with empty Node and metrics) before the last row
        rows = rows[0:-1] + [{"Node": "", **self.new_req_dict()}] + [rows[-1]]

        # Set column alignment: "Node" column is left-aligned; metric columns are right-aligned
        colalign = ["left"] + ["center"] + ["right"] * (len(self.new_req_dict()))

        # Print the table using tabulate with GitHub-style formatting
        print(tabulate(rows, headers="keys", tablefmt="github", colalign=colalign))
        print()

        # Print storage capacity requirement
        storage_req_list: list[tuple[str, Number]] = [
            ("Weights", wgt_size),
            ("KV-cache", kv_size),
        ] + self.get_extra_storage_req()
        print(
            "Minimum Storage Requirement: "
            + " + ".join(f"({k}) {v}" for k, v in storage_req_list)
            + f" = {sum(v for _, v in storage_req_list)}"
        )
