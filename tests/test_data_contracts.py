"""隔离执行真实函数体；不模拟成已通过 LeRobot/JAX/Orbax 集成。"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_function(relative, name, namespace):
    source = ast.parse((ROOT / relative).read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, 'exec'), namespace)
    return namespace[name]


class DataContracts(unittest.TestCase):
    def setUp(self):
        self.meta = Mock(return_value=NS(fps=20, tasks={}))
        self.dataset = Mock(return_value=NS(hf_dataset=None))
        self.fake = Mock(return_value='fake')
        self.loader = load_function('src/openpi/training/data_loader.py', 'create_torch_dataset', {
            'lerobot_dataset': NS(LeRobotDatasetMetadata=self.meta, LeRobotDataset=self.dataset),
            'FakeDataset': self.fake,
        })

    def config(self, **kw):
        return NS(**dict(repo_id='local/example', local_data_dir=kw.get('root'), action_sequence_keys=('actions',), prompt_from_task=False))

    def test_local_root_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'meta').mkdir()
            (Path(tmp) / 'meta/info.json').write_text('{}')
            self.loader(self.config(root=tmp), 3, None)
            self.meta.assert_called_once_with('local/example', root=Path(tmp))
            self.dataset.assert_called_once_with('local/example', root=Path(tmp), delta_timestamps={'actions': [0, .05, .1]})

    def test_missing_local_metadata_fails_before_remote_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.loader(self.config(root=tmp), 1, None)
        self.meta.assert_not_called()

    def test_remote_keeps_repo_id(self):
        self.loader(self.config(), 1, None)
        self.meta.assert_called_once_with('local/example')

    def test_fake_does_not_access_metadata(self):
        cfg = self.config(); cfg.repo_id = 'fake'
        self.assertEqual(self.loader(cfg, 1, None), 'fake')
        self.meta.assert_not_called()

    def test_config_declares_local_root(self):
        tree = ast.parse((ROOT / 'src/openpi/training/config.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DataConfig')
        self.assertIn('local_data_dir', [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)])

    def test_value_builder_preserves_requested_root(self):
        factory = lambda *a, **kw: NS(**kw)
        fn = load_function('scripts/train_value.py', 'build_value_data_config', {
            'Path': Path, '_config': NS(DataConfig=factory),
            '_transforms': NS(Group=factory, ResizeImages=factory, TokenizePrompt=factory, PadStatesAndActions=factory),
            'RemapValueLabelKey': factory, 'value_policy': NS(ValueInputs=factory), 'GemmaValueTokenizer': factory,
        })
        cfg = fn('/private/data/example', NS(max_token_len=48, action_dim=32), tokenizer_path='local.model')
        self.assertEqual(cfg.local_data_dir, '/private/data/example')
        self.assertEqual(cfg.action_sequence_keys, ())


class AdvantageMath(unittest.TestCase):
    def setUp(self):
        self.adv = load_function('scripts/label_advantage_from_vlm.py', '_compute_advantages', {'np': np})

    def test_exact_value_produces_zero_at_all_boundaries(self):
        value = np.array([-.75, -.5, -.25, 0])
        reward = np.array([-.25, -.25, -.25, 0])
        for n in [1, 2, 4, 50]:
            np.testing.assert_allclose(self.adv(value, reward, n), 0)

    def test_failure_terminal_has_no_bootstrap(self):
        np.testing.assert_allclose(self.adv(np.array([-1., -1.]), np.array([0., -1.]), 1), [0., 0.])

    def test_future_value_is_used_within_episode(self):
        np.testing.assert_allclose(self.adv(np.array([-.8, -.2]), np.array([-.1, 0]), 1), [.5, .2], atol=1e-7)


class CheckpointContracts(unittest.TestCase):
    def test_restore_uses_saved_optimizer_and_step(self):
        ckpt = Mock()
        ckpt.metadata.return_value = {'params': {}, 'step': {}, 'opt_state': {}}
        saved_opt = ('adam_moments', 42)
        ckpt.restore.return_value = {'params': {'weight': 1}, 'step': 42, 'opt_state': saved_opt}
        manager = Mock(); manager.__enter__ = Mock(return_value=ckpt); manager.__exit__ = Mock(return_value=False)
        ocp = NS(PyTreeCheckpointer=Mock(return_value=manager), args=NS(PyTreeRestore=lambda **kw: NS(**kw)))
        fn = load_function('scripts/train_value.py', 'load_checkpoint', {
            'Path': Path, 'TrainState': NS, 'logging': logging, 'console': NS(ok=str),
        })
        state = NS(step=0, params=Mock(), opt_state=('initial', 0), model_def='graph', ema_params=None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'orbax': NS(checkpoint=ocp), 'orbax.checkpoint': ocp}):
            result = fn(Path(tmp), state)
        self.assertEqual(result.step, 42)
        self.assertEqual(result.opt_state, saved_opt)
        self.assertIs(ckpt.restore.call_args.kwargs['args'].item['opt_state'], state.opt_state)

    def test_legacy_resume_rejected_before_mutation(self):
        ckpt = Mock(); ckpt.metadata.return_value = {'params': {}, 'step': {}}
        manager = Mock(); manager.__enter__ = Mock(return_value=ckpt); manager.__exit__ = Mock(return_value=False)
        ocp = NS(PyTreeCheckpointer=Mock(return_value=manager))
        fn = load_function('scripts/train_value.py', 'load_checkpoint', {'Path': Path})
        state = NS(params=Mock())
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'orbax': NS(checkpoint=ocp), 'orbax.checkpoint': ocp}):
            with self.assertRaises(ValueError): fn(Path(tmp), state)
        state.params.replace_by_pure_dict.assert_not_called()

    def test_checkpoint_saves_optimizer(self):
        ckpt = Mock()
        manager = Mock(); manager.__enter__ = Mock(return_value=ckpt); manager.__exit__ = Mock(return_value=False)
        ocp = NS(PyTreeCheckpointer=Mock(return_value=manager))
        fn = load_function('scripts/train_value.py', 'save_checkpoint', {'Path': Path, 'logging': logging, 'console': NS(ok=str)})
        state = NS(step=42, params=Mock(), opt_state=('moments', 42), ema_params=None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'orbax': NS(checkpoint=ocp), 'orbax.checkpoint': ocp}):
            fn(state, Path(tmp), 42)
        self.assertEqual(ckpt.save.call_args.args[1]['opt_state'], state.opt_state)
        self.assertEqual(ckpt.save.call_args.args[1]['step'], 42)


class TrainingBoundaries(unittest.TestCase):
    def test_same_value_train_validation_root_rejected(self):
        validate = load_function('scripts/train_value.py', 'validate_training_args', {'Path': Path})
        args = NS(batch_size=4, num_train_steps=5, log_interval=1, save_interval=5,
                  data_dir='/data/train', val_data_dir='/data/train/../train')
        with self.assertRaises(ValueError): validate(args)
        args.val_data_dir = '/data/val'; validate(args)

    def test_invalid_interval_rejected(self):
        validate = load_function('scripts/train_value.py', 'validate_training_args', {'Path': Path})
        with self.assertRaises(ValueError):
            validate(NS(batch_size=4, num_train_steps=5, log_interval=0, save_interval=5))

    def test_mixed_episode_shard_rejected_before_labeling(self):
        extract = load_function('scripts/label_advantage_from_vlm.py', '_extract_episode_id', {'_to_scalar_int': int})
        class Frame(dict):
            columns = ['episode_index']; empty = False
        self.assertEqual(extract(Frame(episode_index=[3, 3]), 0), 3)
        with self.assertRaises(ValueError): extract(Frame(episode_index=[3, 4]), 0)


class LoggingInitialization(unittest.TestCase):
    """不导入JAX/模型，隔离测试真实train.py中的日志初始化函数。"""

    def test_init_logging_without_root_handlers(self):
        fn = load_function('scripts/train.py', 'init_logging', {'logging': logging})
        root = logging.Logger('train_init_no_handlers')
        with patch.object(logging, 'getLogger', return_value=root):
            fn()
        self.assertEqual(len(root.handlers), 1)
        self.assertIsInstance(root.handlers[0], logging.StreamHandler)
        self.assertIsNotNone(root.handlers[0].formatter)
        root.handlers[0].close()

    def test_init_logging_reuses_existing_handler(self):
        fn = load_function('scripts/train.py', 'init_logging', {'logging': logging})
        root = logging.Logger('train_init_existing_handler')
        handler = logging.NullHandler()
        root.addHandler(handler)
        with patch.object(logging, 'getLogger', return_value=root):
            fn()
        self.assertEqual(root.handlers, [handler])
        self.assertIsNotNone(handler.formatter)
        handler.close()


if __name__ == '__main__':
    unittest.main()
