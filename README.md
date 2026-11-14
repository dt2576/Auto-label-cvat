# autolabel_yolov

Standalone YOLO11 agent for object detection and instance segmentation. This project is independent from the SAM3 agent in the parent repository.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Detection

```powershell
python -m yolo_agent.infer --source images --output runs\bbox --mode bbox --model yolo11x.pt
```

## Segmentation

```powershell
python -m yolo_agent.infer --source images --output runs\seg --mode segmentation --model yolo11x-seg.pt
```

Output JSON contains `label`, `confidence`, `type`, and `points`. YOLO downloads the selected model on first use.

## Docker

```powershell
docker build -t autolabel-yolov .
docker run --rm --gpus all -v ${PWD}:/work autolabel-yolov python -m yolo_agent.infer --source /work/images --output /work/runs/seg --mode segmentation --model yolo11x-seg.pt
```

## Verify

```powershell
python -m compileall -q yolo_agent
```

## CVAT full flow

### SAM3

Giao diện có bộ chọn **Model: YOLO11 / SAM3**. SAM3 dùng `Sam3Model` và `Sam3Processor` từ Transformers, chạy trên máy; không gửi ảnh tới Hugging Face hay Meta để nhận diện.

1. Cài thư viện bằng `.\.venv\Scripts\python.exe -m pip install -r requirements-sam3.txt` (máy này đã cài).
2. Tạo tài khoản và yêu cầu quyền truy cập tại https://huggingface.co/facebook/sam3.
3. Sau khi được cấp quyền, nhấp đúp `download-sam3.cmd`, đăng nhập bằng token Hugging Face trong cửa sổ lệnh. Không gửi token vào chat. Lệnh này chỉ tải model, không đọc dữ liệu CVAT.
4. Tải lại giao diện, chọn **SAM3**, nhập nhãn có trong job (ví dụ `car, road, vegetation`), chọn bbox hoặc segmentation và bấm Gửi.

Trọng số Transformers lưu tại `models/sam3/`; file `sam3.pt` của thư viện Meta không thay thế trực tiếp định dạng này.
Inference chỉ đọc model cục bộ (`local_files_only=True`), tắt telemetry và chế độ tải online.
GPU dưới 6 GB mặc định dùng CPU/RAM, có thể chậm. Biến `SAM3_DEVICE=cpu` hoặc `SAM3_DEVICE=cuda:0` cho phép chọn thiết bị; chạy CUDA trên GPU 4 GB có thể hết bộ nhớ. Có thể đổi thư mục qua `SAM3_MODEL_DIR` và ngưỡng qua `SAM3_CONFIDENCE` (mặc định 0.3). Giao diện có ô **Confidence SAM3** để chỉnh theo từng lần chạy. Giảm ngưỡng có thể giảm bỏ sót nhưng tăng nhận nhầm.
CPU mặc định dùng tối đa 8 luồng; có thể đổi qua `SAM3_CPU_THREADS`. Model đã được kiểm tra trên ảnh mẫu bằng CPU của máy này: khoảng 35 giây cho một ảnh, một prompt; thời gian thực tế thay đổi theo số nhãn và tải máy.
SAM3 tìm mask theo nhiều prompt ngắn cho từng nhãn, gộp kết quả và loại trùng trong cùng nhãn. Các nhãn khác nhau được giữ độc lập (ví dụ người nằm trong vùng xe bus). Mỗi phần rời hợp lệ của mask xuất thành polygon riêng; lỗ trong polygon chưa được biểu diễn. Chất lượng cho road/lane cần kiểm tra trên dữ liệu thực tế; không tạo nhãn giả chỉ để đủ số lớp.
Khi SAM3 không có kết quả mới cho một nhãn trên một frame, tool giữ shape hiện có của nhãn đó. Khi có kết quả mới, các shape của nhãn/frame đó được thay thế. Trước khi upload, annotation hiện tại được sao lưu vào `annotations_before_upload_<timestamp>.json` trong thư mục lần chạy.
Các lần chạy lưu trong thư mục con riêng tại `runs/job_<id>/`, giữ kết quả lần trước.

### Windows đã cài đặt

Nhấp đúp `start-cvat.cmd`, hoặc chạy `.\.venv\Scripts\python.exe -m cvat_agent.ui`.
Mở http://127.0.0.1:8787, nhập URL job và Organization, bấm **Mở profile đăng nhập** để đăng nhập CVAT, chọn kiểu annotation rồi nhập yêu cầu (ví dụ `car, pedestrian, bus, traffic light`).

Luồng CVAT dùng YOLO11s và YOLO11s-seg để phù hợp GPU 4 GB; model lưu tại `models/`.
Có thể đổi model bằng biến môi trường `YOLO_BBOX_MODEL` và `YOLO_SEG_MODEL`.
YOLO COCO hỗ trợ 80 lớp; `person` được ánh xạ sang `pedestrian` khi cần. Các label như road, sky, lane, rider không được mô hình này hỗ trợ.
Khi chạy, shape của các label được YOLO xử lý trên các frame đã xử lý sẽ được thay thế. Các label khác, tags và tracks được giữ lại.

The CVAT runner keeps the same browser/CDP flow as the parent agent. It reads the job labels, downloads every frame, maps YOLO classes to available CVAT labels, removes annotations for processed frames, opens locked or hidden labels, and uploads bbox or segmentation results.

```powershell
python -m cvat_agent.ui
```

Or run directly:

```powershell
python -m cvat_agent.job_runner "https://cvat.edufun.ai/tasks/56/jobs/130" cohort-5a yolo bbox runs 80
```

### Giữ hoặc reset annotation khi chạy lại

Trường **Annotation cũ khi chạy lại** áp dụng cho cả project semantic và bbox:
- **Giữ** (mặc định): giữ annotation cũ; chỉ thêm dự đoán cho cặp frame/label chưa có shape. Nếu label đã có track thì không thêm dự đoán label đó trong lượt chạy. Không dùng chế độ này để bổ sung object còn thiếu của một label đã có trên cùng frame.
- **Thêm**: giữ toàn bộ annotation cũ và thêm kết quả mới. Có thể sinh trùng object hoặc polygon chồng lấn; cần review, đặc biệt với semantic.
- **Reset**: thay shape của nhãn được xử lý trên các frame của lượt chạy, kể cả khi kết quả mới rỗng. Không xóa nhãn khác, frame khác, tags hoặc tracks; không phải reset toàn job.

Save thay đổi trên CVAT trước khi chạy. Mỗi upload lưu backup JSON trong thư mục run. Nút Xóa hội thoại chỉ xóa giao diện chat, không xóa annotation CVAT. Chọn đúng URL job của từng project; chọn segmentation cho semantic, bbox cho nhóm object của project bbox. Các chế độ này không tự giải quyết phân loại area/direct–alternative hoặc polyline.

### Xuất mask trực tiếp

Chọn SAM3 và kiểu annotation **Mask · SAM3** để gửi mask RLE trực tiếp lên CVAT, không qua polygon. Chế độ này giữ lỗ và các vùng rời rạc trong mask; không dùng giới hạn điểm polygon. Giữ/Thêm/Reset và Tạm dừng/Tiếp tục vẫn áp dụng. Muốn thay polygon cũ cùng nhãn bằng mask, chọn Reset cho nhãn cần chuyển. Mask chưa tự loại bỏ chồng lấn giữa các nhãn semantic; cần review trên CVAT. Định dạng đối chiếu với cvat-sdk/cvat_sdk/masks.py của CVAT.

### Polyline cho lane

Chọn SAM3 → Polyline · SAM3 và chỉ nhập nhãn lane/.... Tool tạo mask rồi lấy đường tim, tách nhánh và các vùng rời rạc, xuất shape polyline lên CVAT. Giới hạn điểm chia đường dài thành các đoạn nối tiếp thay vì cắt góc. Chức năng tạo nhãn sơ bộ: cần review phân loại vạch đơn/đôi, màu vạch, crosswalk và road curb. Mask dự đoán sai hoặc nối liền hai vạch có thể tạo đường tim sai. Không tự đổi bbox hoặc polygon cũ; dùng Giữ/Thêm/Reset theo nhu cầu.

### Hủy lượt chạy

Nút Hủy lượt chạy dừng tiến trình xử lý, kể cả khi đang tạm dừng. Kết quả đã lưu trong runs vẫn còn. Hủy không hoàn tác annotation nếu yêu cầu upload đã được gửi tới CVAT; kiểm tra job và backup nếu cần. Lượt bị hủy không có chức năng tiếp tục; hãy tạo lượt mới.
