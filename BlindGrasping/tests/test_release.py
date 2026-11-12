"""CPU checks for released assets, launch configurations and actual policies."""

import ast
import contextlib
import copy
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_release import EXCLUDED, check_release
from tactile_config import load_feature_indices, load_tactile_points


def read_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text())


def read_checkpoint(path):
    import torch
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)


def observation_method():
    """Run the real observation writer without importing the simulator."""
    import torch
    tree = ast.parse((ROOT / 'tasks/shadow_hand.py').read_text())
    hand = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ShadowHand')
    method = next(n for n in hand.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_blind')
    scope = {'torch': torch, 'unscale': lambda x, lo, hi: (2 * x - hi - lo) / (hi - lo)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / 'tasks/shadow_hand.py'), 'exec'), scope)
    return scope['compute_blind']


def observation_fixture(record, batch):
    import torch
    points = load_tactile_points(ROOT / record['surface_point_file'])
    nodes = sum(len(v) for k, v in points.items() if k not in EXCLUDED)
    selected = load_feature_indices(ROOT / record['feature_index_file']
                                    if record['feature_index_file'] else None, nodes)
    features = len(selected)
    generator = torch.Generator().manual_seed(42)
    full = torch.randn(batch, nodes, 3, generator=generator) * .001
    tensor = lambda *shape: torch.zeros(*shape)
    hand = SimpleNamespace(
        num_shadow_hand_dofs=22, num_fingertips=5, num_envs=batch, num_actions=24,
        num_surface_points=nodes, num_tac_features=features,
        feature_factor=.5 if record['observations'] == 2940 else 1.,
        shadow_hand_dof_pos=tensor(batch, 22), shadow_hand_dof_vel=tensor(batch, 22),
        shadow_hand_dof_lower_limits=-torch.ones(22), shadow_hand_dof_upper_limits=torch.ones(22),
        hand_pos=tensor(batch, 3), hand_rot=tensor(batch, 4),
        object_pose=tensor(batch, 7), object_linvel=tensor(batch, 3),
        object_angvel=tensor(batch, 3), goal_pos=tensor(batch, 3),
        fingertip_state=tensor(batch, 5, 13), cps=tensor(batch),
        contact_vector=tensor(batch, 17), actions=tensor(batch, 24),
        displacements=full[:, selected, :], vel_obs_scale=.2,
        obs_buf=tensor(batch, record['observations']),
        states_buf=tensor(batch, 174 + 3 * features))
    return hand


class ReleaseTests(unittest.TestCase):
    def test_assets_and_checkpoint_manifests(self):
        self.assertEqual(len(check_release()), 4)

    def test_sodacan_retains_shadow_dimensions(self):
        path = ROOT / 'data/meshdata/sodacan/coacd/decomposed_new1.obj'
        vertices = [[float(x) for x in line.split()[1:4]]
                    for line in path.read_text().splitlines() if line.startswith('v ')]
        dimensions = [max(v[i] for v in vertices) - min(v[i] for v in vertices) for i in range(3)]
        for actual, expected in zip(dimensions, [.05485956, .05486222, .12734027]):
            self.assertAlmostEqual(actual, expected, places=7)
        self.assertFalse((ROOT / 'data/objects').exists())
        for path in (ROOT / 'data/meshdata').glob('*/coacd-shadow'):
            self.fail(f'Duplicate Shadow directory remains: {path}')

    def test_original_400_feature_order_and_link_mapping(self):
        points = load_tactile_points(ROOT / 'data/tactile/shadow-2000.json')
        selected = load_feature_indices(ROOT / 'data/tactile/shadow-400-indices.json', 1865)
        mapping = json.loads((ROOT / 'data/tactile/shadow-400-links.json').read_text())
        self.assertEqual(len(selected), 400)
        mark = 0
        for link, coordinates in points.items():
            if link in EXCLUDED:
                continue
            lo, hi = mapping[link]
            self.assertTrue(all(mark <= index < mark + len(coordinates)
                                for index in selected[lo:hi]), link)
            mark += len(coordinates)
        self.assertEqual(mark, 1865)

    def test_invalid_feature_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'indices.json'
            for indices in ([0, 0], [-1], [10], [True], []):
                path.write_text(json.dumps({'centers': indices}))
                with self.assertRaises(ValueError):
                    load_feature_indices(path, 10)

    def test_all_documented_evaluation_configurations(self):
        import yaml
        from omegaconf import OmegaConf
        launch_tree = ast.parse((ROOT / 'train.py').read_text())
        preprocess = next(n for n in launch_tree.body
                          if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_train_config')
        scope = {}
        exec(compile(ast.Module(body=[preprocess], type_ignores=[]),
                     str(ROOT / 'train.py'), 'exec'), scope)
        for obj in ('cube', 'egg', 'rock', 'sodacan'):
            launch_configurations = []
            for args in ([f'+release={obj}'],
                         ['--config-path', f'runs/{obj}-ckpt', '--config-name', 'config']):
                with self.subTest(object=obj, arguments=args):
                    process = subprocess.run(
                        [sys.executable, 'train.py', *args, '--cfg', 'job', '--resolve'],
                        cwd=ROOT, text=True, capture_output=True)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    configuration = yaml.safe_load(process.stdout)
                    launch_configurations.append(configuration)
                    self.assertEqual(configuration['task']['env']['objectType'], obj)
                    self.assertEqual(configuration['checkpoint'], f'runs/{obj}-ckpt/nn/ShadowHand.pth')
                    self.assertTrue(configuration['test'])
                    self.assertEqual(configuration['task']['env']['controlFrequencyInv'], 1)
                    self.assertFalse(configuration['task']['env']['useRelativeControl'])
                    self.assertFalse(configuration['pbt']['enabled'])
                    self.assertEqual(configuration['train']['params']['load_path'],
                                     configuration['checkpoint'])
                    self.assertTrue(configuration['train']['params']['load_checkpoint'])
                    self.assertEqual(configuration['train']['params']['config']['num_actors'],
                                     configuration['task']['env']['numEnvs'])
                    with contextlib.redirect_stdout(io.StringIO()):
                        processed = scope['preprocess_train_config'](
                            OmegaConf.create(configuration), copy.deepcopy(configuration['train']))
                    self.assertFalse(processed['params']['config']['population_based_training'])
            self.assertEqual(launch_configurations[0], launch_configurations[1])

    def test_real_policy_loading_and_observation_writes(self):
        import torch
        from rl_games.algos_torch import models, network_builder
        manifest = json.loads((ROOT / 'runs/checkpoints.json').read_text())
        writer = observation_method()
        network = read_yaml(ROOT / 'cfg/train/ShadowHandPPO.yaml')['params']['network']
        for obj, record in manifest['objects'].items():
            builder = network_builder.A2CBuilder()
            builder.load(network)
            with contextlib.redirect_stdout(io.StringIO()):
                model = models.ModelA2CContinuousLogStd(builder).build({
                    'actions_num': 24, 'input_shape': (record['observations'],),
                    'num_seqs': 1, 'value_size': 1,
                    'normalize_input': True, 'normalize_value': True})
            state = read_checkpoint(ROOT / record['checkpoint'])['model']
            model.load_state_dict(state, strict=True)
            model.eval()
            for batch in (1, 4):
                with self.subTest(object=obj, batch=batch):
                    hand = observation_fixture(record, batch)
                    original = hand.displacements.clone()
                    writer(hand)
                    actor = hand.obs_buf.clone()
                    writer(hand, asymm_obs=True)
                    self.assertTrue(torch.equal(original, hand.displacements))
                    self.assertTrue(torch.equal(actor, hand.obs_buf))
                    size = 3 * hand.num_tac_features
                    self.assertTrue(torch.equal(actor[:, 111:111 + size],
                                                hand.states_buf[:, 150:150 + size]))
                    with torch.no_grad():
                        result = model({'is_train': False, 'obs': hand.obs_buf,
                                        'rnn_states': None})
                    self.assertEqual(tuple(result['actions'].shape), (batch, 24))
                    self.assertTrue(bool(torch.isfinite(result['actions']).all()))
                    self.assertTrue(bool(torch.isfinite(result['values']).all()))


if __name__ == '__main__':
    unittest.main()
