# 中文导读：π₀/π₀.₅ 模型尺寸与推理配置。
# 来源：ybpy/pistar，固定版本 d7b6ebd；原始算法与接口保持不变。
import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # 设置当前模型变体的默认参数。
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # π₀.₅ 与 π₀ 的核心配置存在以下差异：
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    pistar: bool = False
    # PiStar 推理时对有优势条件与丢弃条件的速度场进行引导融合的系数。
    adv_guidance_beta: float = 2.0
    # 此参数由 ModelTransformFactory 读取，而非模型前向函数直接使用。
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            base_len = 200 if self.pi05 else 48
            star_len = 3 if self.pistar else 0
            object.__setattr__(self, "max_token_len", base_len + star_len)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_kwargs = {
                "images": {
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                "image_masks": {
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                "state": jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                "tokenized_prompt": jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                "tokenized_prompt_mask": jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            }
            if self.pistar and self.adv_guidance_beta != 1.0:
                observation_kwargs["tokenized_prompt_uncond"] = jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32
                )
                observation_kwargs["tokenized_prompt_uncond_mask"] = jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], bool
                )

            observation_spec = _model.Observation(
                **observation_kwargs,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """根据当前模型配置返回冻结参数过滤器。"""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # 仅冻结 Gemma 参数时，保留 Action Expert 为可训练。
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # 任何 LoRA 方案都应保留 LoRA 参数为可训练。
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
