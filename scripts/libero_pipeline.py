#!/usr/bin/env python3
"""使用同一套显式本地配置完成 LIBERO 归一化、SFT/RECAP 训练和策略服务。"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('norm', 'train', 'serve'))
    p.add_argument('--stage', choices=('sft', 'recap'), required=True)
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--repo-id', required=True, help='逻辑数据集标识，与归一化资产ID保持一致')
    p.add_argument('--init-checkpoint', type=Path, help='train模式的基础/SFT checkpoint目录，包含params/')
    p.add_argument('--checkpoint', type=Path, help='serve模式的完整训练step目录')
    p.add_argument('--exp-name', default='local_run')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-train-steps', type=int, default=5)
    p.add_argument('--warmup-steps', type=int, default=1)
    p.add_argument('--save-interval', type=int, default=5)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--fsdp-devices', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--discrete-state-input', action=argparse.BooleanOptionalAction, default=False,
                   help='显式控制π0.5状态token；训练与推理必须保持一致')
    p.add_argument('--dry-run', action='store_true', help='仅打印配置；不导入模型、读取数据或启动计算')
    args = p.parse_args(argv)
    if args.mode != 'serve' and args.dataset_root is None:
        p.error('norm/train模式必须提供 --dataset-root')
    if args.mode == 'train' and not args.init_checkpoint:
        p.error('train模式必须提供 --init-checkpoint；recap阶段应指向SFT checkpoint')
    if args.mode == 'serve' and not args.checkpoint:
        p.error('serve模式必须提供 --checkpoint')
    if min(args.batch_size, args.num_train_steps, args.save_interval, args.fsdp_devices) <= 0:
        p.error('batch-size/num-train-steps/save-interval/fsdp-devices必须为正数')
    if args.batch_size % args.fsdp_devices:
        p.error('batch-size必须能被fsdp-devices整除')
    if not 0 <= args.warmup_steps < args.num_train_steps:
        p.error('warmup-steps必须满足0 <= warmup < num-train-steps')
    if args.max_frames is not None and args.max_frames < args.batch_size:
        p.error('max-frames必须至少覆盖一个完整batch')
    if args.num_workers < 0 or (args.max_frames is not None and args.max_frames <= 0):
        p.error('num-workers必须非负，max-frames必须为正数')
    if not 1 <= args.port <= 65535:
        p.error('port必须在1..65535范围')
    for label, value in [('repo-id', args.repo_id), ('exp-name', args.exp_name)]:
        path = Path(value)
        if not value.strip() or path.is_absolute() or '..' in path.parts or (label == 'exp-name' and len(path.parts) != 1):
            p.error(f'{label}必须是安全的相对标识，不得包含..')
    for field in ('dataset_root', 'init_checkpoint', 'checkpoint'):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    return args


def describe(args):
    return {
        'mode': args.mode, 'stage': args.stage,
        'config_name': f'pi05_star_libero_{args.stage}',
        'repo_id': args.repo_id, 'dataset_root': str(args.dataset_root) if args.dataset_root else None,
        'init_params': str(args.init_checkpoint / 'params') if args.init_checkpoint else None,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'exp_name': args.exp_name, 'batch_size': args.batch_size,
        'num_train_steps': args.num_train_steps, 'warmup_steps': args.warmup_steps,
        'save_interval': args.save_interval, 'num_workers': args.num_workers,
        'fsdp_devices': args.fsdp_devices, 'seed': args.seed,
        'adv_ind_dropout': args.mode != 'serve',
        'wandb_enabled': False, 'resume': args.resume,
        'discrete_state_input': args.discrete_state_input,
    }


def validate_inputs(args):
    if args.mode == 'serve':
        if not (args.checkpoint / 'params').is_dir():
            raise FileNotFoundError(f'找不到参数目录: {args.checkpoint / "params"}')
        stats = args.checkpoint / 'assets' / args.repo_id / 'norm_stats.json'
        if not stats.is_file():
            raise FileNotFoundError(f'checkpoint归一化资产与repo-id不匹配: {stats}')
        return
    info_path = args.dataset_root / 'meta/info.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    if str(info.get('codebase_version', '')).startswith('v3'):
        raise ValueError('当前LIBERO数据链使用逐Episode parquet，请先转换到兼容格式，勿直接混用v3 shards')
    frames = info.get('total_frames')
    if isinstance(frames, int) and frames < args.batch_size:
        raise ValueError(f'数据只有{frames}帧，小于batch_size={args.batch_size}，无法形成完整batch')
    required = {'image', 'wrist_image', 'state', 'actions', 'adv_ind'}
    missing = required - info.get('features', {}).keys()
    if missing:
        raise ValueError(f'数据schema缺少字段: {sorted(missing)}')
    if args.mode == 'train' and not (args.init_checkpoint / 'params').is_dir():
        raise FileNotFoundError(f'找不到参数目录: {args.init_checkpoint / "params"}')


def build_config(args):
    # 延迟导入；--help/--dry-run无需JAX、Torch、模型或数据。
    from openpi.training import config
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    base = config.get_config('pi05_star_libero')
    data = dataclasses.replace(
        base.data, repo_id=args.repo_id,
        base_config=dataclasses.replace(base.data.base_config, local_data_dir=str(args.dataset_root) if args.dataset_root else None),
        adv_ind_dropout=args.mode != 'serve',
    )
    changes = dict(
        name=f'pi05_star_libero_{args.stage}', data=data, exp_name=args.exp_name,
        model=dataclasses.replace(base.model, discrete_state_input=args.discrete_state_input),
        batch_size=args.batch_size, num_train_steps=args.num_train_steps,
        num_workers=args.num_workers, fsdp_devices=args.fsdp_devices, seed=args.seed,
        save_interval=args.save_interval, keep_period=None,
        wandb_enabled=False, overwrite=False, resume=args.resume,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps, peak_lr=5e-5,
            decay_steps=args.num_train_steps, decay_lr=5e-6,
        ),
    )
    if args.init_checkpoint:
        changes['weight_loader'] = weight_loaders.CheckpointWeightLoader(str(args.init_checkpoint / 'params'))
    return dataclasses.replace(base, **changes)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(describe(args), ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    validate_inputs(args)
    sys.path.insert(0, str(ROOT / 'src'))
    cfg = build_config(args)
    if args.mode == 'norm':
        from compute_norm_stats import compute_for_config
        compute_for_config(cfg, args.max_frames)
    elif args.mode == 'train':
        from train import main as train_main
        train_main(cfg)
    else:
        from openpi.policies.policy_config import create_trained_policy
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        policy = create_trained_policy(cfg, str(args.checkpoint))
        WebsocketPolicyServer(policy=policy, host='0.0.0.0', port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
