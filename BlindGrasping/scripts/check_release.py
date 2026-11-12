"""Check the four-object release without importing Isaac Gym or PyTorch."""

import hashlib
import json
from pathlib import Path
import pickletools
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'utils'))
from tactile_config import load_feature_indices, load_tactile_points

EXCLUDED = {'robot0:ffknuckle_child', 'robot0:mfknuckle_child',
            'robot0:rfknuckle_child', 'robot0:lfknuckle_child',
            'robot0:thhub_child', 'robot0:thbase_child'}


def policy_shapes(path):
    """Read tensor dimensions only; do not execute checkpoint pickle objects."""
    with zipfile.ZipFile(path) as archive:
        entry = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        operations = list(pickletools.genops(archive.read(entry)))
    sizes = {}
    fields = {'a2c_network.actor_mlp.0.weight': 'observations',
              'a2c_network.mu.bias': 'actions'}
    for i, (_, name, _) in enumerate(operations):
        if not isinstance(name, str) or name not in fields:
            continue
        start = next(j for j in range(i + 1, len(operations))
                     if operations[j][0].name == 'BINPERSID')
        numbers = []
        for operation, value, _ in operations[start + 1:start + 7]:
            if operation.name.startswith('TUPLE'):
                break
            if operation.name in ('BININT', 'BININT1', 'BININT2'):
                numbers.append(value)
        sizes[fields[name]] = numbers[-1]
    if set(sizes) != {'observations', 'actions'}:
        raise ValueError(f'Missing actor shape metadata: {path}')
    return sizes


def check_release():
    manifest = json.loads((ROOT / 'runs/checkpoints.json').read_text())
    objects = manifest['objects']
    assert set(objects) == {'cube', 'egg', 'rock', 'sodacan'}
    known_objects = {p.name for p in (ROOT / 'data/meshdata').iterdir()
                     if p.is_dir() and (p / 'coacd').exists()}
    assert known_objects == set(objects), known_objects
    reports = []
    for obj, record in objects.items():
        directory = ROOT / 'data/meshdata' / obj / 'coacd'
        assert {p.name for p in directory.iterdir()} == {'coacd1.urdf', 'decomposed_new1.obj'}
        tree = ET.parse(directory / 'coacd1.urdf')
        for mesh in tree.findall('.//mesh'):
            assert (directory / mesh.attrib['filename']).is_file()
        points = load_tactile_points(ROOT / record['surface_point_file'])
        nodes = sum(len(p) for link, p in points.items() if link not in EXCLUDED)
        indices = load_feature_indices(ROOT / record['feature_index_file']
                                       if record['feature_index_file'] else None, nodes)
        assert 135 + 3 * len(indices) == record['observations']
        path = ROOT / record['checkpoint']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        sizes = policy_shapes(path)
        assert sizes == {'observations': record['observations'], 'actions': 24}, sizes
        reports.append(f'{obj}: asset, layout, checkpoint dimensions and checksum OK '
                       f"({record['observations']} observations, 24 actions)")
    return reports


if __name__ == '__main__':
    for report in check_release():
        print(report)
