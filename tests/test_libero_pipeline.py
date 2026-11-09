import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/libero_pipeline.py'
spec = importlib.util.spec_from_file_location('libero_pipeline', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Pipeline(unittest.TestCase):
    def args(self, stage='recap', mode='train'):
        return mod.parse_args([mode, '--stage', stage, '--dataset-root', '/datasets/mixed',
                               '--repo-id', 'local/mixed', '--init-checkpoint', '/weights/sft'])

    def test_recap_spec_is_explicit(self):
        args = self.args()
        spec = mod.describe(args)
        self.assertEqual(spec['config_name'], 'pi05_star_libero_recap')
        self.assertEqual(spec['init_params'], '/weights/sft/params')
        self.assertEqual(spec['dataset_root'], '/datasets/mixed')
        self.assertFalse(spec['wandb_enabled'])

    def test_serve_disables_condition_dropout(self):
        args = mod.parse_args(['serve', '--stage', 'recap', '--repo-id', 'local/mixed', '--checkpoint', '/weights/recap'])
        self.assertFalse(mod.describe(args)['adv_ind_dropout'])
        self.assertIsNone(args.dataset_root)

    def test_invalid_inputs_fail_before_model_import(self):
        for extra in (['--batch-size', '0'], ['--warmup-steps', '5'], ['--exp-name', '../bad']):
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                mod.parse_args(['train', '--stage', 'recap', '--dataset-root', '/d',
                                '--repo-id', 'local/d', '--init-checkpoint', '/w', *extra])

    def test_dry_run_uses_no_model_dependencies(self):
        with patch.object(mod, 'build_config', side_effect=AssertionError('must not import')), patch('builtins.print'):
            mod.main(['train', '--stage', 'recap', '--dataset-root', '/missing',
                      '--repo-id', 'local/d', '--init-checkpoint', '/missing/w', '--dry-run'])

    def test_schema_and_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'meta').mkdir(); (root / 'params').mkdir()
            info = {'codebase_version': 'v2.1', 'features': {k: {} for k in ['image','wrist_image','state','actions','adv_ind']}}
            (root/'meta/info.json').write_text(json.dumps(info))
            args = self.args(); args.dataset_root = args.init_checkpoint = root
            mod.validate_inputs(args)
            info['total_frames'] = 1; (root/'meta/info.json').write_text(json.dumps(info))
            with self.assertRaises(ValueError): mod.validate_inputs(args)
            info['total_frames'] = 20
            info['codebase_version'] = 'v3.0'; (root/'meta/info.json').write_text(json.dumps(info))
            with self.assertRaises(ValueError): mod.validate_inputs(args)

    def test_same_asset_and_root_used_for_train_and_serve(self):
        @dataclasses.dataclass
        class Data:
            local_data_dir: str = None
        @dataclasses.dataclass
        class Factory:
            repo_id: str
            base_config: Data
            adv_ind_dropout: bool = True
        # replace() works on a generated dataclass containing the actual override names.
        args = self.args()
        fields = ['name','model','data','exp_name','batch_size','num_train_steps','num_workers','fsdp_devices','seed',
                  'save_interval','keep_period','wandb_enabled','overwrite','resume','lr_schedule','weight_loader']
        Config = dataclasses.make_dataclass('Config', [(f, object) for f in fields])
        base = Config(**{f: None for f in fields}); base.data = Factory('old/id', Data())
        Model = dataclasses.make_dataclass('Model', [('discrete_state_input', bool)])
        base.model = Model(False)
        training = ModuleType('openpi.training')
        training.config = NS(get_config=lambda name: base)
        training.optimizer = NS(CosineDecaySchedule=lambda **kw: kw)
        training.weight_loaders = NS(CheckpointWeightLoader=lambda path: path)
        with patch.dict(sys.modules, {'openpi': ModuleType('openpi'), 'openpi.training': training}):
            cfg = mod.build_config(args)
            self.assertEqual(cfg.data.base_config.local_data_dir, '/datasets/mixed')
            self.assertEqual(cfg.data.repo_id, 'local/mixed')
            self.assertEqual(cfg.weight_loader, '/weights/sft/params')
            args.mode = 'serve'; cfg = mod.build_config(args)
            self.assertFalse(cfg.data.adv_ind_dropout)
            self.assertFalse(cfg.overwrite)
            args.discrete_state_input = True
            self.assertTrue(mod.build_config(args).model.discrete_state_input)


if __name__ == '__main__': unittest.main()
