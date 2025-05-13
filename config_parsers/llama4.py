from typing import cast

from .base import BaseModelConfigParser, Number, TransformerMode, torch_dtype_width


class Llama4ConfigParser(BaseModelConfigParser):
    def get_layer_list(self) -> list[str]:
        return [
            "Attn - Norm",
            "Attn - QKV_Proj",
            "Attn - RoPE",
            "Attn - SDPA",
            "Attn - O_Proj",
            "Attn - ResidualAdd",
            "Ffn - Norm",
            "Ffn - Router",
            "Ffn - RoutedExp_GateUp_Proj",
            "Ffn - RoutedExp_ActMul",
            "Ffn - RoutedExp_Down_Proj",
            "Ffn - SharedExp_GateUp_Proj",
            "Ffn - SharedExp_ActMul",
            "Ffn - SharedExp_Down_Proj",
            "Ffn - RoutedSharedExpAdd",
            "Ffn - ResidualAdd",
        ]

    def get_num_blocks(self) -> int:
        match self.query_conf.t_mode:
            case TransformerMode.Text:
                return self.model_conf["text_config"]["num_hidden_layers"]
            case TransformerMode.Vision:
                return self.model_conf["vision_config"]["num_hidden_layers"]

    @property
    def hw_req_by_layers(self) -> dict[str, dict[str, Number]]:
        if self._hw_req_by_layers is not None:
            return self._hw_req_by_layers

        req_dict: dict[str, dict[str, Number]] = {
            key: self.new_req_dict() for key in self.get_layer_list()
        }

        match self.query_conf.t_mode:
            case TransformerMode.Text:
                text_config: dict = self.model_conf["text_config"]

                self.set_proj_req(
                    req=req_dict["Attn - QKV_Proj"],
                    dim_m=self.query_conf.n_computed_tokens + self.query_conf.n_recomputed_tokens,
                    dim_n=text_config["head_dim"]
                    * (text_config["num_attention_heads"] + text_config["num_key_value_heads"] * 2),
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_text_sdpa_req(req=req_dict["Attn - SDPA"])
                self.set_proj_req(
                    req=req_dict["Attn - O_Proj"],
                    dim_m=self.query_conf.n_computed_tokens,
                    dim_n=text_config["hidden_size"],
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )

                self.set_proj_req(
                    req=req_dict["Ffn - Router"],
                    dim_m=self.query_conf.n_computed_tokens,
                    dim_n=text_config["num_local_experts"],
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_proj_req(
                    req=req_dict["Ffn - RoutedExp_GateUp_Proj"],
                    dim_m=self.query_conf.n_computed_tokens,
                    dim_n=text_config["intermediate_size"] * 2,
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_proj_req(
                    req=req_dict["Ffn - RoutedExp_Down_Proj"],
                    dim_m=self.query_conf.n_computed_tokens,
                    dim_n=text_config["hidden_size"],
                    dim_k=text_config["intermediate_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_proj_req(
                    req=req_dict["Ffn - SharedExp_GateUp_Proj"],
                    dim_m=self.query_conf.n_computed_tokens,
                    dim_n=text_config["intermediate_size"] * 2,
                    dim_k=text_config["hidden_size"],
                    torch_dtype=text_config["torch_dtype"],
                )
                self.set_proj_req(
                    req=req_dict["Ffn - SharedExp_Down_Proj"],
                    dim_m=self.query_conf.n_computed_tokens,
                    dim_n=text_config["hidden_size"],
                    dim_k=text_config["intermediate_size"],
                    torch_dtype=text_config["torch_dtype"],
                )

            case TransformerMode.Vision:
                raise RuntimeError("Unsupported Mode")

        self._hw_req_by_layers = req_dict
        return self._hw_req_by_layers

    def set_proj_req(
        self,
        req: dict[str, Number],
        dim_m: int,
        dim_n: int,
        dim_k: int,
        torch_dtype: str,
    ) -> None:
        req[BaseModelConfigParser.METRIC_COMPUTE].value = dim_m * dim_n * (dim_k * 2 - 1)
        req[BaseModelConfigParser.METRIC_BW_WGT].value = (dim_k * dim_n) * torch_dtype_width(
            torch_dtype
        )
        req[BaseModelConfigParser.METRIC_BW_IPT].value = (dim_m * dim_k) * torch_dtype_width(
            torch_dtype
        )
        req[BaseModelConfigParser.METRIC_BW_OPT].value = (dim_m * dim_n) * torch_dtype_width(
            torch_dtype
        )

    def set_text_sdpa_req(self, req: dict[str, Number]) -> None:
        text_config: dict = self.model_conf["text_config"]
        total_seq_len: int = (
            self.query_conf.n_cached_tokens
            + self.query_conf.n_recomputed_tokens
            + self.query_conf.n_computed_tokens
        )
        tensor_qo_dims: int = text_config["hidden_size"]
        tensor_kv_dims: int = text_config["head_dim"] * text_config["num_key_value_heads"]
        torch_dtype: str = text_config["torch_dtype"]

        req[BaseModelConfigParser.METRIC_COMPUTE].value = 0  # Initialization
        req[BaseModelConfigParser.METRIC_BW_WGT].value = 0 * torch_dtype_width(torch_dtype)
        req[BaseModelConfigParser.METRIC_BW_IPT].value = (
            total_seq_len * (tensor_qo_dims + tensor_kv_dims * 2) * torch_dtype_width(torch_dtype)
        )
        req[BaseModelConfigParser.METRIC_BW_OPT].value = (
            total_seq_len * tensor_qo_dims * torch_dtype_width(torch_dtype)
        )

        # GEMM: P = QK^T
        req[BaseModelConfigParser.METRIC_COMPUTE].value = cast(
            int, req[BaseModelConfigParser.METRIC_COMPUTE].value
        ) + (self.query_conf.n_computed_tokens * total_seq_len * tensor_qo_dims)

        # GEMM: O = SV
        req[BaseModelConfigParser.METRIC_COMPUTE].value = cast(
            int, req[BaseModelConfigParser.METRIC_COMPUTE].value
        ) + (self.query_conf.n_computed_tokens * tensor_kv_dims * total_seq_len)
