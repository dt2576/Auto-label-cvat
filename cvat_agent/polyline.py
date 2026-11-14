"""Trace open centerline paths without bridging disconnected mask components."""
import numpy as np


def thin(mask):
    pixels = np.pad(np.asarray(mask, dtype=bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [pixels[:-2, 1:-1], pixels[:-2, 2:], pixels[1:-1, 2:],
                 pixels[2:, 2:], pixels[2:, 1:-1], pixels[2:, :-2],
                 pixels[1:-1, :-2], pixels[:-2, :-2]]
            count = np.sum(p, axis=0)
            transitions = np.sum([~p[i] & p[(i + 1) % 8] for i in range(8)], axis=0)
            if phase == 0:
                guard = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                guard = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = pixels[1:-1, 1:-1] & (count >= 2) & (count <= 6) & (transitions == 1) & guard
            if remove.any():
                pixels[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            return pixels[1:-1, 1:-1]


def mask_polylines(mask, max_points=60):
    if max_points and max_points < 2:
        raise ValueError('Polyline cần ít nhất 2 điểm hoặc 0')
    skeleton = thin(mask)
    vertices = set(zip(*np.nonzero(skeleton)))
    graph = {}
    for y, x in sorted(vertices):
        neighbors = []
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            other = (y + dy, x + dx)
            if other not in vertices:
                continue
            if dy and dx and ((y + dy, x) in vertices or (y, x + dx) in vertices):
                continue
            neighbors.append(other)
        graph[y, x] = neighbors
    visited = set()
    paths = []
    # Branches/endpoints first, then remaining cycles.
    starts = sorted(graph, key=lambda v: (len(graph[v]) == 2, v))
    for start in starts:
        for neighbor in graph[start]:
            edge = frozenset((start, neighbor))
            if edge in visited:
                continue
            visited.add(edge)
            path = [start, neighbor]
            while len(graph[path[-1]]) == 2:
                candidates = [v for v in graph[path[-1]] if frozenset((path[-1], v)) not in visited]
                if not candidates:
                    break
                nxt = candidates[0]
                visited.add(frozenset((path[-1], nxt)))
                if nxt == start:
                    break
                path.append(nxt)
            # Remove only exactly collinear pixels; never shortcut across a gap.
            simplified = [path[0]]
            for a, b, c in zip(path, path[1:], path[2:]):
                if (b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1]):
                    simplified.append(b)
            simplified.append(path[-1])
            limit = max_points or len(simplified)
            for offset in range(0, len(simplified)-1, limit-1):
                segment = simplified[offset:offset+limit]
                paths.append([float(value) for y, x in segment for value in (x, y)])
    return paths
