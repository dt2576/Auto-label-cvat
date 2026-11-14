"""Run local YOLO inference using the labels available in a CVAT job."""
import json
import os
from pathlib import Path

os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'


def label_mapping(names, labels):
    available = {label.lower().replace('_', ' '): label for label in labels}
    mapping = {}
    for index, name in names.items():
        candidates = [name]
        if name == 'person':
            candidates.append('pedestrian')
        for candidate in candidates:
            if candidate in available:
                mapping[index] = available[candidate]
                break
    return mapping


def infer_job(images, output, labels, mode, max_points=60):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    if mode not in {'bbox', 'segmentation'}:
        raise ValueError('Mode must be bbox or segmentation')
    root = Path(__file__).resolve().parent.parent
    model_name = os.environ.get('YOLO_SEG_MODEL' if mode == 'segmentation' else 'YOLO_BBOX_MODEL',
                                'yolo11s-seg.pt' if mode == 'segmentation' else 'yolo11s.pt')
    model_path = Path(model_name)
    if not model_path.is_absolute():
        model_path = root / 'models' / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f'Model chưa có trên máy: {model_path}. Hãy tải model trước khi xử lý dữ liệu.')
    model = YOLO(str(model_path))
    mapping = label_mapping(model.names, labels)
    if not mapping:
        raise ValueError('Không có label CVAT khớp các lớp của YOLO. YOLO COCO không hỗ trợ road, sky hay lane.')
    unsupported = [label for label in labels if label not in mapping.values()]
    if unsupported:
        print('YOLO bỏ qua label không hỗ trợ: ' + ', '.join(unsupported), flush=True)
    files = sorted(Path(images).glob('*'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = '0' if torch.cuda.is_available() else 'cpu'
    for current, path in enumerate(files, 1):
        from .run_control import checkpoint
        checkpoint()
        result = model.predict(str(path), classes=list(mapping), device=device, conf=0.25,
                               imgsz=640, verbose=False)[0]
        items = []
        for index, box in enumerate(result.boxes):
            if mode == 'bbox':
                points = box.xyxy[0].cpu().tolist()
            else:
                if result.masks is None:
                    continue
                polygon = result.masks.xy[index]
                if max_points and len(polygon) > max_points:
                    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
                    epsilon = 0.001
                    perimeter = cv2.arcLength(contour, True)
                    while len(polygon) > max_points and epsilon <= 1:
                        polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True).reshape(-1, 2)
                        epsilon *= 1.5
                points = polygon.reshape(-1).tolist()
                if len(points) < 6:
                    continue
            items.append({'label': mapping[int(box.cls[0])], 'confidence': float(box.conf[0]),
                          'type': 'rectangle' if mode == 'bbox' else 'polygon', 'points': points})
        (output / f'{path.stem}.json').write_text(json.dumps(items, ensure_ascii=False), encoding='utf-8')
        print(f'[frame {current}/{len(files)}] {path.name}: {len(items)} detections', flush=True)
    return list(dict.fromkeys(mapping.values()))
