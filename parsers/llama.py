from core import (
    BaseModelConfigParser,
    Number,
    TransformerMode,
    act_flops,
    torch_dtype_width,
)


class LlamaConfigParser(BaseModelConfigParser):
    def __init__(self, model_config, query_config):
        super().__init__(model_config, query_config)

        if self.query_conf.t_mode == TransformerMode.Vision:
            raise NotImplementedError

    def get_layer_list(self) -> list[str]:
        return [
            "Attn - RMSNorm",
            "Attn - QKV_Proj",
            "Attn - RoPE",
            "Attn - SDPA",
            "Attn - O_Proj",
            "Attn - ResidualAdd",
            "Ffn - RMSNorm",
            "Ffn - GateUp_Proj",
            "Ffn - ActMul",
            "Ffn - Down_Proj",
            "Ffn - ResidualAdd",
        ]

    def get_num_blocks(self) -> int:
        return self.model_conf["num_hidden_layers"]

    def get_kvcache_size(self) -> int:
        kvcache_size_per_block: int = 0

        batch_size: int = len(self.query_conf.n_cached_tokens)
        tensor_kv_dims: int = (
            self.model_conf["hidden_size"]
            / self.model_conf["num_attention_heads"]
            * self.model_conf["num_key_value_heads"]
        )
        torch_dtype: str = self.model_conf["torch_dtype"]

        for query_idx in range(batch_size):
            kv_seq_len: int = (
                self.query_conf.n_cached_tokens[query_idx]
                + self.query_conf.n_input_tokens[query_idx]
            )
            kvcache_size_per_block += (
                kv_seq_len * (tensor_kv_dims * 2) * torch_dtype_width(torch_dtype)
            )

        return kvcache_size_per_block * self.get_num_blocks()

    def get_extra_storage_req(self) -> list[tuple[str, Number]]:
        req_list: list[tuple[str, Number]] = []

        # Embedding Table
        emb_table_size: int = (
            self.model_conf["hidden_size"]
            * self.model_conf["vocab_size"]
            * torch_dtype_width(self.model_conf["torch_dtype"])
        )
        req_list.append(("Embedding Table", Number("B", "!.2k", emb_table_size)))

        return req_list

    @property
    def hw_req_by_layers(self) -> dict[str, dict[str, Number]]:
        if self._hw_req_by_layers is not None:
            return self._hw_req_by_layers.copy()

        req_dict: dict[str, dict[str, Number]] = {
            key: self.new_req_dict() for key in self.get_layer_list()
        }
        head_dim: int = self.model_conf["hidden_size"] / self.model_conf["num_attention_heads"]

        self.set_op_rmsnorm_req(
            layer_entry=req_dict["Attn - RMSNorm"],
            hidden_size=self.model_conf["hidden_size"],
            n_tokens=sum(self.query_conf.n_input_tokens),
            torch_dtype=self.model_conf["torch_dtype"],
        )
        self.set_op_proj_req(
            layer_entry=req_dict["Attn - QKV_Proj"],
            dim_m=sum(self.query_conf.n_input_tokens),
            dim_n=head_dim
            * (self.model_conf["num_attention_heads"] + self.model_conf["num_key_value_heads"] * 2),
            dim_k=self.model_conf["hidden_size"],
            torch_dtype=self.model_conf["torch_dtype"],
        )
        self.set_op_rope_req(
            layer_entry=req_dict["Attn - RoPE"],
            token_dims=head_dim
            * (self.model_conf["num_attention_heads"] + self.model_conf["num_key_value_heads"]),
            n_tokens=sum(self.query_conf.n_input_tokens),
            torch_dtype=self.model_conf["torch_dtype"],
        )
        self.set_sdpa_req(req=req_dict["Attn - SDPA"])
        self.set_op_proj_req(
            layer_entry=req_dict["Attn - O_Proj"],
            dim_m=sum(self.query_conf.n_input_tokens),
            dim_n=self.model_conf["hidden_size"],
            dim_k=self.model_conf["hidden_size"],
            torch_dtype=self.model_conf["torch_dtype"],
        )
        self.set_op_sum_req(
            layer_entry=req_dict["Attn - ResidualAdd"],
            num_elem=sum(self.query_conf.n_input_tokens) * self.model_conf["hidden_size"],
            num_tensors=2,
            torch_dtype=self.model_conf["torch_dtype"],
        )

        self.set_op_rmsnorm_req(
            layer_entry=req_dict["Ffn - RMSNorm"],
            hidden_size=self.model_conf["hidden_size"],
            n_tokens=sum(self.query_conf.n_input_tokens),
            torch_dtype=self.model_conf["torch_dtype"],
        )
        self.set_op_proj_req(
            layer_entry=req_dict["Ffn - GateUp_Proj"],
            dim_m=sum(self.query_conf.n_input_tokens),
            dim_n=self.model_conf["intermediate_size"] * 2,
            dim_k=self.model_conf["hidden_size"],
            torch_dtype=self.model_conf["torch_dtype"],
        )
        self.set_op_actmul_req(
            layer_entry=req_dict["Ffn - ActMul"],
            intermediate_size=self.model_conf["intermediate_size"],
            n_tokens=sum(self.query_conf.n_input_tokens),
            act_flops=act_flops(self.model_conf["hidden_act"]),
            torch_dtype=self.model_conf["torch_dtype"],
        )
        self.set_op_proj_req(
            layer_entry=req_dict["Ffn - Down_Proj"],
            dim_m=sum(self.query_conf.n_input_tokens),
            dim_n=self.model_conf["hidden_size"],
            dim_k=self.model_conf["intermediate_size"],
            torch_dtype=self.model_conf["torch_dtype"],
        )
        self.set_op_sum_req(
            layer_entry=req_dict["Ffn - ResidualAdd"],
            num_elem=sum(self.query_conf.n_input_tokens) * self.model_conf["hidden_size"],
            num_tensors=2,
            torch_dtype=self.model_conf["torch_dtype"],
        )

        self._hw_req_by_layers = req_dict
        return self._hw_req_by_layers.copy()

    def set_sdpa_req(self, req: dict[str, Number]) -> None:
        batch_size: int = len(self.query_conf.n_cached_tokens)
        tensor_qo_dims: int = self.model_conf["hidden_size"]
        tensor_kv_dims: int = (
            self.model_conf["hidden_size"]
            / self.model_conf["num_attention_heads"]
            * self.model_conf["num_key_value_heads"]
        )
        torch_dtype: str = self.model_conf["torch_dtype"]

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
