import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from .cdp import ensure_browser, evaluate
    from .job_data import save_chunk
    from .label_agent import LabelDescriptionAgent
    from .shape_rules import allowed_labels
except ImportError:
    from cdp import ensure_browser, evaluate
    from job_data import save_chunk
    from label_agent import LabelDescriptionAgent
    from shape_rules import allowed_labels


ROOT = Path(__file__).resolve().parent.parent


def fetch_json(url):
    try:
        return evaluate(f"fetch({json.dumps(url)}).then(r=>{{if(!r.ok) throw new Error('HTTP '+r.status); return r.json()}})")
    except RuntimeError as error:
        if 'HTTP 401' in str(error):
            raise RuntimeError('CVAT chưa đăng nhập trong Chrome profile. Hãy đăng nhập tại tab CVAT vừa mở rồi chạy lại job.') from error
        raise


def parse_job_url(url):
    match = re.search(r'/tasks/(\d+)/jobs/(\d+)', url)
    if not match:
        raise ValueError('URL must contain /tasks/<task>/jobs/<job>')
    return int(match.group(1)), int(match.group(2))


def labels_for_job(job_id, destination):
    labels = []
    page = 1
    while True:
        data = fetch_json(f'/api/labels?job_id={job_id}&page={page}')
        labels.extend(item['name'] for item in data.get('results', []))
        if not data.get('next'):
            break
        page += 1
    labels = allowed_labels(labels)
    destination.write_text(json.dumps(labels, ensure_ascii=False), encoding='utf-8')
    return labels


def prediction_summary(predictions, labels):
    found = set()
    files = sorted(Path(predictions).glob('frame_*.json'))
    shapes = 0
    for path in files:
        for item in json.loads(path.read_text(encoding='utf-8')):
            label = item.get('label')
            if label:
                found.add(label)
                shapes += 1
    missing = [label for label in labels if label not in found]
    return {'files': len(files), 'shapes': shapes, 'found_labels': len(found), 'missing_labels': missing}


def collect_images(job_id, start_frame, count, destination):
    images = destination / 'images'
    images.mkdir(parents=True, exist_ok=True)
    chunk_root = destination / 'chunks'
    frame = start_frame
    chunk = 0
    while frame < start_frame + count:
        for attempt in range(1, 4):
            try:
                save_chunk(job_id, chunk_root, chunk)
                break
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(
                        f'Không tải được chunk {chunk} sau 3 lần; đã tải {frame - start_frame}/{count} frame. '
                        f'Lỗi gốc: {error}'
                    ) from error
                print(f'[download] Chunk {chunk}, lần {attempt}/3 thất bại: {error}. Đang thử lại...', flush=True)
                time.sleep(attempt)
        source = chunk_root / f'job_{job_id}_chunk_{chunk}'
        candidates = [path for path in source.rglob('*') if path.is_file()
                      and path.suffix.lower() in {'.jpeg', '.jpg', '.png', '.bmp', '.webp', '.tif', '.tiff'}]
        candidates.sort(key=lambda path: [int(part) if part.isdigit() else part.lower()
                                         for part in re.split(r'(\d+)', path.relative_to(source).as_posix())])
        if not candidates:
            raise RuntimeError(f'Chunk {chunk} không chứa ảnh được hỗ trợ; đã tải {frame - start_frame}/{count} frame')
        for image in candidates:
            shutil.copy2(image, images / f'frame_{frame:04d}{image.suffix.lower()}')
            frame += 1
            if frame >= start_frame + count:
                break
        chunk += 1
    if frame != start_frame + count:
        raise RuntimeError(f'Expected {count} frames, collected {frame - start_frame}')
    return images


def run(command):
    environment = os.environ.copy()
    environment['DOCKER_CONFIG'] = str(ROOT / '.docker')
    environment['DOCKER_HOST'] = 'npipe:////./pipe/dockerDesktopLinuxEngine'
    (ROOT / '.docker').mkdir(exist_ok=True)
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    if process.wait() != 0:
        raise RuntimeError('Command failed: ' + ' '.join(command))


def docker_environment():
    environment = os.environ.copy()
    environment['DOCKER_CONFIG'] = str(ROOT / '.docker')
    environment['DOCKER_HOST'] = 'npipe:////./pipe/dockerDesktopLinuxEngine'
    (ROOT / '.docker').mkdir(exist_ok=True)
    return environment


def ensure_worker(checkpoint):
    try:
        urllib.request.urlopen('http://127.0.0.1:8790/health', timeout=1).close()
        return
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    environment = docker_environment()
    start = subprocess.run(['docker', 'start', 'cvat-sam3-worker'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
    if start.returncode != 0:
        run(['docker', 'run', '-d', '--name', 'cvat-sam3-worker', '--no-healthcheck', '--gpus', 'all', '-p', '127.0.0.1:8790:8790', '-v', f'{ROOT}:/work', '-e', 'PYTHONPATH=/work/vendor/sam3', 'cvat.pytorch.sam3:latest', 'python', '/work/cvat_agent/sam3_worker.py', container_path(checkpoint)])
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            urllib.request.urlopen('http://127.0.0.1:8790/health', timeout=1).close()
            return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError('SAM3 worker chưa sẵn sàng tại cổng 8790')


def run_worker(image_dir, output_dir, first_frame, labels_path, mode, max_points, progress_path):
    request = json.dumps({'image_dir': container_path(image_dir), 'output_dir': container_path(output_dir), 'first_frame': first_frame, 'labels_path': container_path(labels_path), 'mode': mode, 'max_points': max_points, 'progress_path': container_path(progress_path)}).encode()
    result = {}

    def request_worker():
        try:
            with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8790/infer', data=request, headers={'Content-Type': 'application/json'}), timeout=3600) as response:
                result.update(json.loads(response.read()))
        except Exception as error:
            result['error'] = str(error)

    thread = threading.Thread(target=request_worker)
    thread.start()
    last = None
    while thread.is_alive():
        if Path(progress_path).exists():
            value = json.loads(Path(progress_path).read_text(encoding='utf-8'))
            current = f"[frame {value['current']}/{value['total']}] frame_{value['frame']:04d}.jpeg"
            if current != last:
                print(current, flush=True)
                last = current
        time.sleep(0.5)
    thread.join()
    if result.get('state') != 'done':
        raise RuntimeError(result.get('error', 'SAM3 worker inference failed'))


def normalize_predictions(predictions):
    for path in Path(predictions).glob('frame_*.json'):
        items = json.loads(path.read_text(encoding='utf-8'))
        normalized = []
        for item in items:
            points = item.get('points', [])
            shape_type = shape_for_label(item.get('label'))
            if shape_type is None or len(points) < 4:
                continue
            if shape_type == 'rectangle' and len(points) != 4:
                coordinates = list(zip(points[::2], points[1::2]))
                points = [min(x for x, _ in coordinates), min(y for _, y in coordinates), max(x for x, _ in coordinates), max(y for _, y in coordinates)]
            if shape_type == 'polygon' and len(points) < 6:
                continue
            if shape_type == 'polyline' and len(points) < 4:
                continue
            if shape_type == 'polyline' and points[:2] == points[-2:]:
                points = points[:-2]
            item['type'] = shape_type
            item['points'] = points
            normalized.append(item)
        path.write_text(json.dumps(normalized, ensure_ascii=False), encoding='utf-8')


def container_path(path):
    return f"/work/{Path(path).resolve().relative_to(ROOT).as_posix()}"


def log(step, message):
    from .run_control import checkpoint
    checkpoint()
    print(f'[agent {step}] {message}', flush=True)


def ensure_ui():
    try:
        urllib.request.urlopen('http://127.0.0.1:8787/api/health', timeout=0.5).close()
    except (urllib.error.URLError, TimeoutError, OSError):
        subprocess.Popen([sys.executable, '-m', 'cvat_agent.ui', '--open'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(url, organization, model, mode, output_root, max_points=80, prompt_file=None):
    if mode in {'mask', 'polyline'} and model != 'sam3':
        raise ValueError('Mask/Polyline cần chọn SAM3')
    if model not in {'yolo', 'sam3'}:
        raise ValueError('Model must be yolo or sam3')
    if model == 'sam3':
        from .sam3_backend import require_model
        require_model()
    ensure_ui()
    log('1/8', 'Đọc URL job và kết nối Chrome CDP')
    task_id, job_id = parse_job_url(url)
    ensure_browser(url)
    evaluate(f'location.href={json.dumps(url)}')
    time.sleep(3)
    log('2/8', f'Tải metadata task={task_id}, job={job_id}')
    job = fetch_json(f'/api/jobs/{job_id}')
    meta = fetch_json(f'/api/jobs/{job_id}/data/meta')
    run_dir = Path(output_root) / f'job_{job_id}' / f'{model}_{time.time_ns()}'
    run_dir.mkdir(parents=True)
    labels_path = run_dir / 'labels.json'
    log('3/8', 'Lấy toàn bộ label từ CVAT API theo tất cả trang')
    labels = labels_for_job(job_id, labels_path)
    prompts_path = run_dir / 'label_descriptions.json'
    descriptions = LabelDescriptionAgent().build(labels, prompts_path)
    if prompt_file:
        descriptions = json.loads(Path(prompt_file).read_text(encoding='utf-8'))
        prompts_path.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2), encoding='utf-8')
        labels = [item['label'] for item in descriptions]
    log('4/8', f'Tạo {len(labels)} mô tả prompt bằng LabelDescriptionAgent')
    images = collect_images(job_id, int(meta['start_frame']), int(job['frame_count']), run_dir)
    log('5/8', f'Tải {int(job["frame_count"])} frame vào {images}')
    predictions = run_dir / 'predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    log('6/8', f'Chạy {model.upper()} mode={mode}, max_points={max_points}')
    if model == 'sam3':
        from .sam3_backend import infer_job
        supported_labels = infer_job(images, predictions, descriptions, mode, max_points)
    else:
        from .yolo_infer import infer_job
        supported_labels = infer_job(images, predictions, labels, mode, max_points)
    summary = prediction_summary(predictions, labels)
    log('7/8', f"Hậu kiểm {summary['shapes']} shape, {summary['found_labels']}/{len(labels)} label")
    if summary['missing_labels']:
        log('7/8', 'Label chưa có mask: ' + ', '.join(summary['missing_labels']))
    policy = os.environ.get('CVAT_ANNOTATION_POLICY', 'keep')
    log('8/8', f'Upload vào CVAT, annotation_policy={policy}')
    from .annotate_batch import apply
    upload = apply(job_id, predictions, organization, supported_labels, annotation_policy=policy)
    log('8/8', f"Thêm {upload['added_shapes']}, xóa {upload['cleared_shapes']}, bỏ qua {upload['skipped_shapes']} shape. Backup: {upload['backup']}")
    return {'task_id': task_id, 'job_id': job_id, 'frames': int(job['frame_count']), 'model': model, 'mode': mode, **summary}


if __name__ == '__main__':
    try:
        result = main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 80, sys.argv[7] if len(sys.argv) > 7 else None)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        print(f'[agent error] {error}', flush=True)
        sys.exit(1)
