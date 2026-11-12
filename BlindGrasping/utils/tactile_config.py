"""Load the fixed tactile layouts used by the released policies."""

import json
import math


def load_tactile_points(path):
    with open(path, encoding='utf-8') as stream:
        points = json.load(stream)
    if not isinstance(points, dict) or not points:
        raise ValueError('A tactile point file must map hand links to points')
    for link, coordinates in points.items():
        if not isinstance(coordinates, list):
            raise ValueError(f'Invalid tactile points for {link}')
        for point in coordinates:
            if (not isinstance(point, list) or len(point) != 3 or
                    any(not isinstance(x, (int, float)) or not math.isfinite(x)
                        for x in point)):
                raise ValueError(f'Invalid tactile point for {link}: {point}')
    return points


def load_feature_indices(path, node_count):
    if path is None:
        return list(range(node_count))
    with open(path, encoding='utf-8') as stream:
        indices = json.load(stream)['centers']
    if (not isinstance(indices, list) or not indices or
            any(type(i) is not int or not 0 <= i < node_count for i in indices) or
            len(set(indices)) != len(indices)):
        raise ValueError('Tactile feature indices must be unique and within '
                         f'the {node_count} available nodes')
    return indices
