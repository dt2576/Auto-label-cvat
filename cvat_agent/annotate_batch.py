import json
import sys
import time
from pathlib import Path

try:
    from .cdp import evaluate
except ImportError:
    from cdp import evaluate


def fetch_json(url):
    return evaluate(f"fetch({json.dumps(url)}).then(r=>r.json())")


def unlock_and_show_labels():
    return evaluate("[...document.querySelectorAll('.cvat-label-item-button-lock-enabled')].forEach(x=>x.click());[...document.querySelectorAll('.cvat-label-item-button-hidden-enabled')].forEach(x=>x.click());'labels-opened'")


def without_server_ids(value):
    """CVAT PUT recreates annotations and rejects IDs, including nested shapes."""
    if isinstance(value, dict):
        return {key: without_server_ids(item) for key, item in value.items() if key != 'id'}
    if isinstance(value, list):
        return [without_server_ids(item) for item in value]
    return value


def apply(job_id, prediction_dir, organization, selected_labels=None, preserve_empty=False, annotation_policy='keep'):
    if annotation_policy not in {'keep', 'append', 'replace_selected'}:
        raise ValueError('Invalid annotation_policy')
    unlock_and_show_labels()
    files = sorted(Path(prediction_dir).glob('frame_*.json'))
    predictions = {int(path.stem.split('_')[1]): json.loads(path.read_text(encoding='utf-8')) for path in files}
    existing = fetch_json(f'/api/jobs/{job_id}/annotations?org={organization}')
    labels = {}
    page = 1
    while True:
        data = fetch_json(f'/api/labels?job_id={job_id}&page={page}')
        labels.update({label['name']: label['id'] for label in data.get('results', [])})
        if not data.get('next'):
            break
        page += 1
    frames = set(predictions)
    shapes = []
    added = 0
    for frame, items in predictions.items():
        for item in items:
            if selected_labels is not None and item['label'] not in selected_labels:
                raise ValueError(f"Prediction outside selected labels: {item['label']}")
            label_id = labels.get(item['label'])
            shape_type = item.get('type')
            points = item.get('points', [])
            if label_id is None or shape_type not in {'rectangle', 'polygon', 'mask', 'polyline'}:
                continue
            if shape_type == 'mask':
                from .mask_format import validate_mask
                validate_mask(points)
            if shape_type == 'rectangle' and len(points) != 4:
                continue
            if shape_type == 'polygon' and len(points) < 6:
                continue
            if shape_type == 'polyline' and len(points) < 4:
                continue
            shapes.append({'label_id': label_id, 'frame': frame, 'type': shape_type, 'points': points, 'outside': False, 'occluded': False, 'z_order': 0, 'rotation': 0, 'attributes': []})
            added += 1
    selected_ids = {labels[name] for name in (selected_labels or []) if name in labels}
    detected_pairs = {(shape['frame'], shape['label_id']) for shape in shapes}
    # Keep other frames and labels, including classes this model cannot infer.
    kept = [shape for shape in existing.get('shapes', [])
            if shape.get('frame') not in frames or shape.get('label_id') not in selected_ids
            or (preserve_empty and (shape.get('frame'), shape.get('label_id')) not in detected_pairs)]
    skipped = 0
    if annotation_policy in {'keep', 'append'}:
        kept = existing.get('shapes', [])
        if annotation_policy == 'keep':
            occupied = {(shape.get('frame'), shape.get('label_id')) for shape in kept}
            # A track can interpolate through frames without an explicit keyframe.
            # Protect its label throughout this run instead of adding duplicates.
            tracked_labels = {track.get('label_id') for track in existing.get('tracks', [])}
            filtered = [shape for shape in shapes
                        if (shape['frame'], shape['label_id']) not in occupied
                        and shape['label_id'] not in tracked_labels]
            skipped = len(shapes) - len(filtered)
            shapes = filtered
        added = len(shapes)
    cleared = len(existing.get('shapes', [])) - len(kept)
    payload = {'version': existing.get('version', 0), 'tags': existing.get('tags', []),
               'shapes': kept + shapes, 'tracks': existing.get('tracks', [])}
    payload = without_server_ids(payload)
    backup = Path(prediction_dir).parent / f'annotations_before_upload_{time.time_ns()}.json'
    backup.write_text(json.dumps(existing, ensure_ascii=False), encoding='utf-8')
    expression = f"fetch('/api/jobs/{job_id}/annotations?org={organization}',{{method:'PUT',credentials:'include',headers:{{'Content-Type':'application/json','X-CSRFToken':decodeURIComponent(document.cookie.split('; ').find(x=>x.startsWith('csrftoken='))?.split('=')[1]||'')}},body:JSON.stringify({json.dumps(payload,separators=(',', ':'))})}}).then(async r=>{{if(!r.ok) throw new Error('HTTP '+r.status+' '+await r.text()); return r.status}})"
    status = evaluate(expression)
    unlock_and_show_labels()
    evaluate("setTimeout(()=>location.reload(),300); 'reload-scheduled'")
    return {'frames': len(frames), 'cleared_shapes': cleared, 'added_shapes': added,
            'skipped_shapes': skipped, 'annotation_policy': annotation_policy,
            'backup': str(backup), 'status': status}


if __name__ == '__main__':
    print(json.dumps(apply(int(sys.argv[1]), sys.argv[2], sys.argv[3]), ensure_ascii=False))
