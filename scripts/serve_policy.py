# 中文导读：通过策略服务向 LIBERO 返回动作块。
# 代码来自 ybpy/pistar；保留算法与运行参数。
import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """本项目仅支持 LIBERO 仿真推理。"""

    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """从指定训练 Checkpoint 加载推理策略。"""

    # 训练配置名称，例如 pi05_libero。
    config: str
    # Checkpoint 目录，例如 checkpoints/pi05_libero/exp/10000。
    dir: str


@dataclasses.dataclass
class Default:
    """使用所选环境的默认公开 Checkpoint。"""


@dataclasses.dataclass
class Args:
    """策略服务端启动参数。"""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LIBERO

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # 策略服务监听端口。
    port: int = 8000
    # 是否记录策略行为以便调试。
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# LIBERO 仿真环境默认使用的官方 Checkpoint。
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """初始化对应 LIBERO 环境的默认策略。"""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """根据 CLI 参数建立策略实例。"""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # 可选记录推理过程中模型产生的动作。
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
