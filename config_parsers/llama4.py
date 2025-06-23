from .base import (
    BaseModelConfigParser,
    Number,
    TransformerMode,
    act_flops,
    torch_dtype_width,
)


class Llama4ConfigParser(BaseModelConfigParser):
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
        req_list: list[tuple[str, Number]] = []

        match self.query_conf.t_mode:
            case TransformerMode.Text:
                text_config: dict = self.model_conf["text_config"]

                # Additional Experts
                exp_size: int = (
                    text_config["hidden_size"]
                    * text_config["intermediate_size"]
                    * torch_dtype_width(text_config["torch_dtype"])
                    * 3
                )
                extra_exp_cnt: int = (text_config["num_local_experts"] - 1) * (
                    self.get_num_blocks() // text_config["interleave_moe_layer_step"]
                )
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
                self.set_text_sdpa_req(req=req_dict["Attn - SDPA"])
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

    def set_text_sdpa_req(self, req: dict[str, Number]) -> None:
        text_config: dict = self.model_conf["text_config"]
        batch_size: int = len(self.query_conf.n_cached_tokens)
        tensor_qo_dims: int = text_config["hidden_size"]
        tensor_kv_dims: int = text_config["head_dim"] * text_config["num_key_value_heads"]
        torch_dtype: str = text_config["torch_dtype"]

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

        req[BaseModelConfigParser.METRIC_COMPUTE].value = metric_compute
        req[BaseModelConfigParser.METRIC_BW_WGT].value = metric_bw_wgt
        req[BaseModelConfigParser.METRIC_BW_IPT].value = metric_bw_ipt
        req[BaseModelConfigParser.METRIC_BW_OPT].value = metric_bw_opt
