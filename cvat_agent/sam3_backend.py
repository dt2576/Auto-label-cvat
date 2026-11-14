"""Offline SAM3 image inference through Hugging Face Transformers."""
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = ('config.json', 'processor_config.json', 'tokenizer.json',
               'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json',
               'merges.txt', 'model.safetensors')


def model_directory():
    path = Path(os.environ.get('SAM3_MODEL_DIR', str(ROOT / 'models/sam3')))
    return path if path.is_absolute() else ROOT / path


def model_status():
    directory = model_directory()
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    return {'ready': not missing, 'missing': missing}


def require_model():
    if not model_status()['ready']:
        raise RuntimeError('SAM3 chưa có trọng số trên máy. Xin quyền tại https://huggingface.co/facebook/sam3, '
                           'sau đó chạy download-sam3.cmd để đăng nhập và tải model.')
    return model_directory()


def download_model():
    # Downloads weights only; this command never reads CVAT images or cookies.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    try:
        snapshot_download('facebook/sam3', local_dir=str(model_directory()),
                          allow_patterns=list(MODEL_FILES), max_workers=2)
    except GatedRepoError as error:
        raise RuntimeError('Tài khoản Hugging Face chưa được cấp quyền SAM3. '
                           'Mở https://huggingface.co/facebook/sam3 và yêu cầu quyền truy cập.') from error
    require_model()
    print('SAM3 đã tải đủ. Tải lại giao diện Agent để chọn SAM3.', flush=True)


def prompt_variants(description):
    label = description['label'].strip().lower().replace('_', ' ')
    if label.startswith('lane/'):
        name = label.split('/', 1)[1]
        if name == 'crosswalk':
            return ['individual painted crosswalk stripes']
        if name == 'road curb':
            return ['road curb']
        return [name.replace('other', '') + ' painted lane markings']
    aliases = {
        'area/direct': ['drivable road in the current lane ahead of the camera',
                        'the directly drivable road area'],
        'area/alternative': ['adjacent drivable lane going in the same direction',
                             'alternative drivable road area beside the current lane'],
        'bus': ['bus', 'city bus', 'coach'],
        'road': ['road', 'road surface', 'street'],
        'vegetation': ['vegetation', 'trees', 'bushes and plants'],
        'tree': ['tree', 'trees', 'tree canopy'],
        'trees': ['trees', 'tree', 'tree canopy'],
        'person': ['person', 'people'],
        'pedestrian': ['person', 'pedestrian'],
        'motorcycle': ['motorcycle', 'motorbike'],
        'bicycle': ['bicycle', 'bike'],
        'truck': ['truck', 'lorry'],
        'building': ['building', 'buildings'],
        'sidewalk': ['sidewalk', 'pavement'],
    }
    values = aliases.get(label, [label.replace('/', ' '), description['prompt']])
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def mask_polygons(mask, max_points):
    import cv2
    import numpy as np
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 1 or len(contour) < 3:
            continue
        # Choose the smallest approximation error that satisfies the point cap.
        # Uniform vertex sampling can cut corners and destroy narrow features.
        polygon = contour
        if max_points and len(contour) > max_points:
            low, high = 0.0, cv2.arcLength(contour, True)
            best = None
            for _ in range(28):
                epsilon = (low + high) / 2
                candidate = cv2.approxPolyDP(contour, epsilon, True)
                if len(candidate) > max_points:
                    low = epsilon
                else:
                    high = epsilon
                    if len(candidate) >= 3 and cv2.contourArea(candidate) > 0:
                        best = candidate
            if best is not None:
                polygon = best
            else:
                # Preserve the contour rather than produce a degenerate polygon.
                print('[sam3] Không thể giảm điểm mà vẫn giữ polygon hợp lệ; giữ đường bao gốc.', flush=True)
        if cv2.contourArea(polygon) > 0:
            polygons.append(polygon.reshape(-1).astype(float).tolist())
    return polygons


def merge_results(results):
    """Deduplicate prompts for ONE label; never suppress another label."""
    import cv2
    import numpy as np
    import torch
    nonempty = [result for result in results if result['scores'].numel()]
    # Transformers leaves empty masks at model resolution rather than image size.
    # Exclude empty variants before concatenation with resized nonempty masks.
    if not nonempty:
        return {key: value.detach().cpu() for key, value in results[0].items()}
    results = nonempty
    merged = {key: torch.cat([result[key].detach().cpu().to(
        dtype=torch.bool if key == 'masks' else torch.float32) for result in results])
        for key in ('scores', 'boxes', 'masks')}
    small_masks = [cv2.resize(mask.numpy().astype(np.uint8), (128, 128),
                             interpolation=cv2.INTER_NEAREST).astype(bool) for mask in merged['masks']]
    keep = []
    for index in merged['scores'].argsort(descending=True).tolist():
        duplicate = False
        for previous in keep:
            a, b = merged['boxes'][index], merged['boxes'][previous]
            left, top = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
            right, bottom = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
            intersection = max(0, right - left) * max(0, bottom - top)
            if not intersection:
                continue
            union = np.logical_or(small_masks[index], small_masks[previous]).sum()
            if union:
                duplicate = np.logical_and(small_masks[index], small_masks[previous]).sum() / union >= 0.7
            else:
                area_a = float((a[2] - a[0]) * (a[3] - a[1]))
                area_b = float((b[2] - b[0]) * (b[3] - b[1]))
                duplicate = intersection / max(1, area_a + area_b - intersection) >= 0.85
            if duplicate:
                break
        if not duplicate:
            keep.append(index)
    return {key: tensor[keep] for key, tensor in merged.items()}


def convert_result(result, label, mode, max_points):
    items = []
    for index, score in enumerate(result['scores']):
        if mode == 'bbox':
            points = result['boxes'][index].detach().float().cpu().tolist()
        elif mode == 'polyline':
            from .polyline import mask_polylines
            for points in mask_polylines(result['masks'][index].detach().cpu().numpy(), max_points):
                items.append({'type': 'polyline', 'label': label, 'confidence': float(score), 'points': points})
            continue
        elif mode == 'mask':
            from .mask_format import encode_mask
            points = encode_mask(result['masks'][index].detach().cpu().numpy())
            if points is not None:
                items.append({'type': 'mask', 'label': label, 'confidence': float(score), 'points': points})
            continue
        else:
            for points in mask_polygons(result['masks'][index].detach().cpu().numpy(), max_points):
                items.append({'type': 'polygon', 'label': label, 'confidence': float(score), 'points': points})
            continue
        items.append({'type': 'rectangle' if mode == 'bbox' else 'polygon', 'label': label,
                      'confidence': float(score), 'points': points})
    return items


def infer_job(images, output, descriptions, mode, max_points=60):
    directory = require_model()
    if mode not in {'bbox', 'segmentation', 'mask', 'polyline'}:
        raise ValueError('Mode must be bbox, segmentation or mask')
    if mode == 'segmentation' and max_points and max_points < 3:
        raise ValueError('SAM3 polygon cần ít nhất 3 điểm, hoặc 0 để không giới hạn')
    if mode == 'polyline':
        if any(not item['label'].startswith('lane/') for item in descriptions):
            raise ValueError('Chế độ polyline chỉ nhận các nhãn lane/...')
        if max_points and max_points < 2:
            raise ValueError('Polyline cần ít nhất 2 điểm hoặc 0')
    confidence = float(os.environ.get('SAM3_CONFIDENCE', '0.3'))
    if not 0 < confidence < 1:
        raise ValueError('SAM3 confidence phải lớn hơn 0 và nhỏ hơn 1')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    device = os.environ.get('SAM3_DEVICE', 'auto')
    if device == 'auto':
        # Leave headroom for activations; 4 GB cards default to RAM/CPU.
        device = 'cuda:0' if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 6 * 1024**3 else 'cpu'
    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    if device == 'cpu':
        torch.set_num_threads(max(1, int(os.environ.get('SAM3_CPU_THREADS', str(min(os.cpu_count() or 1, 8))))))
    print(f'[sam3] Thiết bị: {device}; chỉ đọc model cục bộ, không gửi ảnh ra dịch vụ AI.', flush=True)
    model = Sam3Model.from_pretrained(str(directory), local_files_only=True, dtype=dtype).to(device).eval()
    processor = Sam3Processor.from_pretrained(str(directory), local_files_only=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in Path(images).iterdir() if path.suffix.lower()
                   in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'})
    with torch.inference_mode():
        for current, path in enumerate(files, 1):
            from .run_control import checkpoint
            checkpoint()
            with Image.open(path) as opened:
                image = opened.convert('RGB')
            image_inputs = processor(images=image, return_tensors='pt').to(device)
            sizes = image_inputs['original_sizes'].cpu().tolist()
            vision = model.get_vision_features(pixel_values=image_inputs['pixel_values'].to(dtype=dtype))
            items = []
            for prompt_index, description in enumerate(descriptions, 1):
                print(f'[sam3] Frame {current}/{len(files)} · prompt {prompt_index}/{len(descriptions)}', flush=True)
                results = []
                variants = prompt_variants(description)
                if os.environ.get('SAM3_SPEED', 'thorough') == 'fast':
                    variants = variants[:1]
                for variant_index, prompt in enumerate(variants, 1):
                    checkpoint()
                    print(f'[sam3] Biến thể {variant_index}/{len(variants)} · confidence {confidence}', flush=True)
                    inputs = processor(text=prompt, original_sizes=sizes, return_tensors='pt').to(device)
                    prediction = model(vision_embeds=vision, input_ids=inputs['input_ids'],
                                       attention_mask=inputs.get('attention_mask'))
                    result = processor.post_process_instance_segmentation(
                        prediction, threshold=confidence, mask_threshold=0.5, target_sizes=sizes)[0]
                    results.append({key: value.detach().cpu().to(dtype=torch.bool if key == 'masks' else torch.float32)
                                    for key, value in result.items() if key in {'scores', 'boxes', 'masks'}})
                    del prediction, result
                merged = merge_results(results)
                label_items = convert_result(merged, description['label'], mode, max_points)
                items.extend(label_items)
                print(f'[sam3] Label {prompt_index}/{len(descriptions)}: {len(label_items)} shape', flush=True)
                del results, merged
            (output / f'{path.stem}.json').write_text(json.dumps(items, ensure_ascii=False), encoding='utf-8')
            print(f'[frame {current}/{len(files)}] {path.name}: {len(items)} detections', flush=True)
            del vision, image_inputs
    return list(dict.fromkeys(item['label'] for item in descriptions))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download_model()
    else:
        print(json.dumps(model_status()))
