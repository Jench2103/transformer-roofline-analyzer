from core import (
    BaseModelConfigParser,
    Number,
    TransformerMode,
    act_flops,
    torch_dtype_width,
)


class Llama4ConfigParser(BaseModelConfigParser):
    """
    Parser for LLaMA-4 transformer architecture with Mixture of Experts (MoE) support.

    LLaMA-4 uses an interleaved MoE architecture where:
    - MoE layers appear every `interleave_moe_layer_step` blocks
    - Non-MoE (dense) layers fill the remaining blocks
    - Each MoE layer has both routed experts and a shared expert

    Key architectural differences from LLaMA-2/3:
    - Nested config structure: fields are under `text_config` or `vision_config`
    - MoE routing: `num_local_experts` total, `num_experts_per_tok` activated per token
    - Shared expert: Always activated alongside routed experts
    - Different intermediate sizes: `intermediate_size` for MoE, `intermediate_size_mlp` for dense

    Supported modes:
    - Text mode: Fully implemented
    - Vision mode: Not yet implemented
    """

    @classmethod
    def normalize_config(cls, config_dict: dict) -> dict:
        """Set default torch_dtype in text_config if not present."""
        if "text_config" in config_dict and isinstance(config_dict["text_config"], dict):
            if "torch_dtype" not in config_dict["text_config"]:
                config_dict["text_config"]["torch_dtype"] = "float16"
        return config_dict

    def get_layer_list(self) -> list[str]:
        match self.query_conf.t_mode:
            case TransformerMode.Text:
                return [
                    "Attn - RMSNorm",
                    "Attn - QKV_Proj",
                    "Attn - RoPE",
                    "Attn - SDPA",
                    "Attn - O_Proj",
                    "Attn - ResidualAdd",
                    "Ffn - RMSNorm",
                    "Ffn - Router",
                    "Ffn - RoutedExp_GateUp_Proj",
                    "Ffn - RoutedExp_ActMul",
                    "Ffn - RoutedExp_Down_Proj",
                    "Ffn - SharedExp_GateUp_Proj",
                    "Ffn - SharedExp_ActMul",
                    "Ffn - SharedExp_Down_Proj",
                    "Ffn - RoutedSharedExpAdd",
                    "Ffn - NonMoE_GateUp_Proj",
                    "Ffn - NonMoE_ActMul",
                    "Ffn - NonMoE_Down_Proj",
                    "Ffn - ResidualAdd",
                ]

            case TransformerMode.Vision:
                raise NotImplementedError

    def get_num_blocks(self) -> int:
        match self.query_conf.t_mode:
            case TransformerMode.Text:
                return self.model_conf["text_config"]["num_hidden_layers"]
            case TransformerMode.Vision:
                return self.model_conf["vision_config"]["num_hidden_layers"]

    def get_layer_num_blocks(self, layer: str) -> int:
        """
        Return the number of transformer blocks that contain the specified layer.

        LLaMA-4 uses interleaved MoE architecture:
        - MoE layers (RoutedExp, SharedExp, RoutedSharedExpAdd) appear in:
          `num_blocks // interleave_moe_layer_step` blocks
        - Non-MoE (dense) layers appear in the remaining blocks:
          `num_blocks - (num_blocks // interleave_moe_layer_step)` blocks
        - Attention layers and other common layers appear in all blocks

        Example with 48 blocks and interleave_moe_layer_step=4:
        - MoE layers: 48 // 4 = 12 blocks (every 4th block)
        - Non-MoE layers: 48 - 12 = 36 blocks (remaining blocks)
        - Attention layers: 48 blocks (all)

        Args:
            layer: Name of the layer to query.

        Returns:
            Number of blocks containing this layer.
        """
        match self.query_conf.t_mode:
            case TransformerMode.Text:
                if (
                    "Ffn - RoutedExp" in layer
                    or "Ffn - SharedExp" in layer
                    or "Ffn - RoutedShared" in layer
                ):
                    return (
                        self.get_num_blocks()
                        // self.model_conf["text_config"]["interleave_moe_layer_step"]
                    )
                elif "Ffn - NonMoE" in layer:
                    return self.get_num_blocks() - (
                        self.get_num_blocks()
                        // self.model_conf["text_config"]["interleave_moe_layer_step"]
                    )
                else:
                    return self.get_num_blocks()
            case TransformerMode.Vision:
                return self.get_num_blocks()

    def get_kvcache_size(self) -> int:
        kvcache_size_per_block: int = 0

        match self.query_conf.t_mode:
            case TransformerMode.Text:
                text_config: dict = self.model_conf["text_config"]
                batch_size: int = len(self.query_conf.n_cached_tokens)
                tensor_kv_dims: int = text_config["head_dim"] * text_config["num_key_value_heads"]
                torch_dtype: str = text_config["torch_dtype"]

                for query_idx in range(batch_size):
                    kv_seq_len: int = (
                        self.query_conf.n_cached_tokens[query_idx]
                        + self.query_conf.n_input_tokens[query_idx]
                    )
                    kvcache_size_per_block += (
                        kv_seq_len * (tensor_kv_dims * 2) * torch_dtype_width(torch_dtype)
                    )

            case TransformerMode.Vision:
                raise NotImplementedError

        return kvcache_size_per_block * self.get_num_blocks()

    def get_extra_storage_req(self) -> list[tuple[str, Number]]:
        """
        Return additional storage requirements beyond weight bandwidth in the metrics table.

        For LLaMA-4 MoE, this includes:

        1. Additional Expert Weights:
           - During inference, only `num_experts_per_tok` experts are used per token
           - The weight bandwidth metrics only account for these activated experts
           - However, all `num_local_experts` must be stored in memory
           - This reports the storage for the unused experts:
             `(num_local_experts - num_experts_per_tok) * expert_size * num_moe_blocks`

        2. Embedding Table:
           - Token embedding weights: `hidden_size * vocab_size * dtype_width`

        Returns:
            List of (description, size) tuples for additional storage.
        """
        req_list: list[tuple[str, Number]] = []

        match self.query_conf.t_mode:
            case TransformerMode.Text:
                text_config: dict = self.model_conf["text_config"]

                # Additional Experts
                # Each expert has 3 weight matrices: gate, up, down
                # Size per expert: hidden_size * intermediate_size * dtype_width * 3
                exp_size: int = (
                    text_config["hidden_size"]
                    * text_config["intermediate_size"]
                    * torch_dtype_width(text_config["torch_dtype"])
                    * 3
                )
                # Number of additional experts per MoE block that aren't activated
                extra_exp_cnt: int = (
                    text_config["num_local_experts"] - text_config["num_experts_per_tok"]
                ) * (self.get_num_blocks() // text_config["interleave_moe_layer_step"])
                req_list.append(
                    ("Additional Experts", Number("B", "!.2k", exp_size * extra_exp_cnt))
                )

                # Embedding Table
                emb_table_size: int = (
                    text_config["hidden_size"]
                    * text_config["vocab_size"]
                    * torch_dtype_width(text_config["torch_dtype"])
                )
                req_list.append(("Embedding Table", Number("B", "!.2k", emb_table_size)))

            case TransformerMode.Vision:
                raise NotImplementedError

        return req_list

    @property
    def hw_req_by_layers(self) -> dict[str, dict[str, Number]]:
        """
        Compute hardware requirements for each layer in LLaMA-4.

        MoE FFN structure (for MoE blocks):
        - Router: Projects input to expert scores (hidden_size -> num_local_experts)
        - RoutedExp: `num_experts_per_tok` activated experts, each processes all tokens
        - SharedExp: Always-active shared expert (1 expert processing all tokens)
        - RoutedSharedExpAdd: Sum outputs from routed and shared experts

        Non-MoE FFN structure (for dense blocks):
        - NonMoE: Standard dense FFN with `intermediate_size_mlp`

        The routed expert loop (line ~220) calls set_op_* methods `num_experts_per_tok`
        times to account for all activated experts per token.

        Returns:
            Dictionary mapping layer names to their hardware metrics.
        """
        if self._hw_req_by_layers is not None:
            return self._hw_req_by_layers.copy()

        req_dict: dict[str, dict[str, Number]] = {
            key: self.new_req_dict() for key in self.get_layer_list()
        }

        match self.query_conf.t_mode:
            case TransformerMode.Text:
                text_config: dict = self.model_conf["text_config"]

                self.set_op_rmsnorm_req(
                    layer_entry=req_dict["Attn - RMSNorm"],
                    hidden_size=text_config["hidden_size"],
                    n_tokens=sum(self.query_conf.n_input_tokens),
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_proj_req(
                    layer_entry=req_dict["Attn - QKV_Proj"],
                    dim_m=sum(self.query_conf.n_input_tokens),
                    dim_n=text_config["head_dim"]
                    * (text_config["num_attention_heads"] + text_config["num_key_value_heads"] * 2),
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_rope_req(
                    layer_entry=req_dict["Attn - RoPE"],
                    token_dims=text_config["head_dim"]
                    * (text_config["num_attention_heads"] + text_config["num_key_value_heads"]),
                    n_tokens=sum(self.query_conf.n_input_tokens),
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_sdpa_req(
                    layer_entry=req_dict["Attn - SDPA"],
                    tensor_qo_dims=text_config["hidden_size"],
                    tensor_kv_dims=text_config["head_dim"] * text_config["num_key_value_heads"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_proj_req(
                    layer_entry=req_dict["Attn - O_Proj"],
                    dim_m=sum(self.query_conf.n_input_tokens),
                    dim_n=text_config["hidden_size"],
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_sum_req(
                    layer_entry=req_dict["Attn - ResidualAdd"],
                    num_elem=sum(self.query_conf.n_input_tokens) * text_config["hidden_size"],
                    num_tensors=2,
                    torch_dtype=text_config["torch_dtype"],
                )

                self.set_op_rmsnorm_req(
                    layer_entry=req_dict["Ffn - RMSNorm"],
                    hidden_size=text_config["hidden_size"],
                    n_tokens=sum(self.query_conf.n_input_tokens),
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_proj_req(
                    layer_entry=req_dict["Ffn - Router"],
                    dim_m=sum(self.query_conf.n_input_tokens),
                    dim_n=text_config["num_local_experts"],
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )

                for _ in range(text_config["num_experts_per_tok"]):
                    self.set_op_proj_req(
                        layer_entry=req_dict["Ffn - RoutedExp_GateUp_Proj"],
                        dim_m=sum(self.query_conf.n_input_tokens),
                        dim_n=text_config["intermediate_size"] * 2,
                        dim_k=text_config["hidden_size"],
                        torch_dtype=text_config["torch_dtype"],
                    )
                    self.set_op_actmul_req(
                        layer_entry=req_dict["Ffn - RoutedExp_ActMul"],
                        intermediate_size=text_config["intermediate_size"],
                        n_tokens=sum(self.query_conf.n_input_tokens),
                        act_flops=act_flops(text_config["hidden_act"]),
                        torch_dtype=text_config["torch_dtype"],
                    )
                    self.set_op_proj_req(
                        layer_entry=req_dict["Ffn - RoutedExp_Down_Proj"],
                        dim_m=sum(self.query_conf.n_input_tokens),
                        dim_n=text_config["hidden_size"],
                        dim_k=text_config["intermediate_size"],
                        torch_dtype=text_config["torch_dtype"],
                    )

                self.set_op_proj_req(
                    layer_entry=req_dict["Ffn - SharedExp_GateUp_Proj"],
                    dim_m=sum(self.query_conf.n_input_tokens),
                    dim_n=text_config["intermediate_size"] * 2,
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_actmul_req(
                    layer_entry=req_dict["Ffn - SharedExp_ActMul"],
                    intermediate_size=text_config["intermediate_size"],
                    n_tokens=sum(self.query_conf.n_input_tokens),
                    act_flops=act_flops(text_config["hidden_act"]),
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_proj_req(
                    layer_entry=req_dict["Ffn - SharedExp_Down_Proj"],
                    dim_m=sum(self.query_conf.n_input_tokens),
                    dim_n=text_config["hidden_size"],
                    dim_k=text_config["intermediate_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_sum_req(
                    layer_entry=req_dict["Ffn - RoutedSharedExpAdd"],
                    num_elem=sum(self.query_conf.n_input_tokens) * text_config["hidden_size"],
                    num_tensors=2,
                    torch_dtype=text_config["torch_dtype"],
                )

                self.set_op_proj_req(
                    layer_entry=req_dict["Ffn - NonMoE_GateUp_Proj"],
                    dim_m=sum(self.query_conf.n_input_tokens),
                    dim_n=text_config["intermediate_size_mlp"] * 2,
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_actmul_req(
                    layer_entry=req_dict["Ffn - NonMoE_ActMul"],
                    intermediate_size=text_config["intermediate_size_mlp"],
                    n_tokens=sum(self.query_conf.n_input_tokens),
                    act_flops=act_flops(text_config["hidden_act"]),
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_op_proj_req(
                    layer_entry=req_dict["Ffn - NonMoE_Down_Proj"],
                    dim_m=sum(self.query_conf.n_input_tokens),
                    dim_n=text_config["hidden_size"],
                    dim_k=text_config["intermediate_size_mlp"],
                    torch_dtype=text_config["torch_dtype"],
                )

                self.set_op_sum_req(
                    layer_entry=req_dict["Ffn - ResidualAdd"],
                    num_elem=sum(self.query_conf.n_input_tokens) * text_config["hidden_size"],
                    num_tensors=2,
                    torch_dtype=text_config["torch_dtype"],
                )

            case TransformerMode.Vision:
                raise RuntimeError("Unsupported Mode")

        self._hw_req_by_layers = req_dict
        return self._hw_req_by_layers.copy()
