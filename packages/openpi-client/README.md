# 策略远程调用客户端

本模块负责 **LIBERO 仿真环境与策略推理服务之间的通信**，并不承担训练任务。

- `websocket_client_policy.py`：通过 WebSocket 向策略服务端发送观测，获取动作预测。
- `image_tools.py`：图像编码与尺寸变换。
- `action_chunk_broker.py`：动作块管理。
- `base_policy.py`：客户端策略协议基础。
- `msgpack_numpy.py`：序列化 NumPy 类型。

调试重点：检查服务端主机与端口、输入图像编码、返回的动作块长度，以及客户端断线重连时的错误信息。

典型链路：`examples/libero/main.py` → `openpi_client.websocket_client_policy` → `scripts/serve_policy.py` → VLA 推理。

上游还包含 `runtime/` 模块及客户端测试，目前尚未迁移；现有 LIBERO 客户端是否依赖这些模块需要通过导入和策略服务联调确认。

