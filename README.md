# Auto-label CVAT

Công cụ hỗ trợ tạo annotation trên CVAT bằng YOLO11 và SAM3. Có giao diện cục bộ, xuất bbox/polygon/mask/polyline, giữ hoặc thay annotation, tạm dừng và hủy lượt chạy. Kết quả là nhãn sơ bộ, cần kiểm tra theo guideline trước khi nộp.

## 1. Chuẩn bị trên máy Windows mới

Cài các phần mềm sau:

- **Git** để clone và cập nhật code.
- **Python 3.12** — phiên bản đã được kiểm tra với project.
- **Google Chrome** để đăng nhập và kết nối CVAT.
- Tài khoản CVAT có quyền sửa job cần xử lý.
- Nếu dùng SAM3: tài khoản Hugging Face được cấp quyền đọc model `facebook/sam3`.

GPU không bắt buộc, nhưng SAM3 trên CPU có thể chậm. GPU NVIDIA cần driver và bản PyTorch có hỗ trợ CUDA phù hợp. Các lệnh cài thư viện bên dưới không bảo đảm GPU đã được bật; xem phần kiểm tra GPU.

Cần Internet để cài thư viện và tải trọng số. Trọng số SAM3 khoảng 3,44 GB; cần thêm dung lượng cho thư viện, ảnh tải từ CVAT và kết quả từng lần chạy.

Nếu dùng server CVAT có sẵn, không cần Docker hay cài CVAT local. Nếu dùng CVAT local, cần cài và khởi động CVAT riêng; repo này không tự cài CVAT. Launcher và cách tìm Chrome hiện dành cho Windows.

## 2. Clone code và cài thư viện

Mở PowerShell tại thư mục muốn đặt project:

```powershell
git clone https://github.com/dt2576/Auto-label-cvat.git
cd Auto-label-cvat
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-sam3.txt
```

Không cần activate môi trường; các lệnh dùng trực tiếp Python trong `.venv`. Nếu chỉ dùng YOLO, có thể cài `requirements.txt` thay cho `requirements-sam3.txt`.

**Clone/pull chỉ tải code, không tải thư viện, model hoặc profile đăng nhập.** Mỗi máy cần tạo `.venv`, tải model và đăng nhập riêng. Phiên bản thư viện chưa được khóa hoàn toàn nên môi trường mới có thể cần xử lý khác biệt dependency.

## 3. Tải model

### SAM3

1. Đăng nhập Hugging Face và xin quyền tại [facebook/sam3](https://huggingface.co/facebook/sam3).
2. Đợi tài khoản được duyệt.
3. Tạo token tại [Hugging Face Access Tokens](https://huggingface.co/settings/tokens), với quyền đọc repository model đã được cấp quyền.
4. Chạy trong thư mục project:

```powershell
.\download-sam3.cmd
```

Nhập token bắt đầu bằng `hf_` trong cửa sổ đăng nhập của chương trình. Không ghi token vào code, README, Git hoặc chat. Chờ tải hoàn tất trước khi chạy inference.

Model lưu tại `models/sam3/`, gồm trọng số `model.safetensors`, config và tokenizer. File `sam3.pt` của thư viện Meta không thay thế trực tiếp bộ model Transformers này. Sau lần tải thành công, không cần tải lại mỗi lần mở tool hoặc pull code.

Kiểm tra bộ file model:

```powershell
.\.venv\Scripts\python.exe -m cvat_agent.sam3_backend
```

### YOLO11 (nếu sử dụng)

Tool CVAT cần trọng số local trước khi chạy. Tải hai model mặc định từ Ultralytics:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -fL --retry 3 -o models/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
curl.exe -fL --retry 3 -o models/yolo11s-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt
```

`yolo11s.pt` dùng cho bbox; `yolo11s-seg.pt` dùng cho polygon. Có thể đổi model qua `YOLO_BBOX_MODEL` và `YOLO_SEG_MODEL`. YOLO mặc định dùng lớp COCO, không hỗ trợ đầy đủ taxonomy của bài: ví dụ `rider`, `road`, `sky`, các lớp `lane/...` và biển báo tổng quát.

## 4. Mở tool và chạy job

```powershell
.\start-cvat.cmd
```

Giữ cửa sổ chương trình mở khi sử dụng. Truy cập [http://127.0.0.1:8787](http://127.0.0.1:8787).

1. Nhập **CVAT job URL** của bạn, ví dụ `https://your-cvat-server/tasks/71/jobs/203`.
2. Nhập Organization nếu job thuộc tổ chức; để trống nếu dùng workspace cá nhân. Không nhập `None` hoặc `null`.
3. Bấm **Mở profile đăng nhập**, đăng nhập CVAT trong cửa sổ Chrome tool mở. Tài khoản CVAT local và server online không tự đồng bộ.
4. Save các chỉnh sửa đang có trong CVAT.
5. Chọn model, kiểu annotation, chế độ xử lý annotation cũ và các thông số.
6. Nhập đúng tên nhãn đã có trong job, phân cách bằng dấu phẩy hoặc xuống dòng. Ví dụ `pedestrian, car, bus`.
7. Kiểm tra danh sách nhãn được chọn trong chat và theo dõi tiến trình.
8. Khi hoàn tất, kiểm tra và sửa kết quả trên CVAT trước khi nộp.

Tên `traffic light` và `traffic_light`, hoặc `pedestrian` và `person`, là các nhãn khác nhau. Nhập tên cụ thể để chọn chính xác; yêu cầu dạng nhóm như “xe” có thể chọn nhiều nhãn.

### Chọn kiểu đầu ra

| Kiểu | Model | Kết quả |
| --- | --- | --- |
| Bounding box | YOLO11 hoặc SAM3 | Hình chữ nhật quanh đối tượng |
| Segmentation polygon | YOLO11 hoặc SAM3 | Đường bao khép kín |
| Mask · SAM3 | SAM3 | Mask RLE trực tiếp, giữ lỗ và chi tiết pixel, không dùng giới hạn điểm polygon |
| Polyline · SAM3 | SAM3 | Đường tim từ mask; chỉ nhận nhãn `lane/...` |

Trong bài BBox + Polygon + Polyline, bbox dành cho 10 lớp đối tượng; polygon chỉ cho `area/direct`, `area/alternative`; polyline dành cho các nhãn `lane/...`. Chạy thành từng lượt riêng trên cùng job. Trong bài Semantic, dùng polygon/mask cho các lớp semantic. Không tự thêm lớp ngoài guideline chỉ vì job có sẵn label đó.

### Giữ, Thêm và Reset

| Chế độ | Hành vi |
| --- | --- |
| **Giữ** (mặc định) | Giữ shape cũ. Chỉ thêm khi cặp frame/label chưa có shape. Nếu label đã có track thì không thêm dự đoán của label đó trong lượt chạy. |
| **Thêm** | Giữ cũ và thêm mới; có thể tạo annotation trùng/chồng nhau. |
| **Reset** | Thay shape của nhãn được model xử lý trên các frame của lượt chạy, kể cả khi kết quả mới rỗng. Giữ nhãn khác, frame khác, tags và tracks. |

Reset `pedestrian` không xóa `person` cũ. Muốn thay polygon bằng mask cho một nhãn, chọn Mask và Reset nhãn đó. Reset không phải xóa toàn job.

Trước mỗi upload, tool lưu `annotations_before_upload_<timestamp>.json` trong thư mục lần chạy. Backup là file cục bộ; chưa có nút khôi phục tự động.

### Tạm dừng, tiếp tục và hủy

- **Tạm dừng:** chờ điểm dừng an toàn giữa các bước xử lý; một phép suy luận đang chạy không dừng tức thì.
- **Tiếp tục:** giữ model và tiến độ trong bộ nhớ, tiếp tục lượt hiện tại. Không tắt chương trình khi muốn tiếp tục.
- **Hủy lượt chạy:** dừng tiến trình, giữ kết quả đã lưu trên máy. Lượt bị hủy không có chức năng tiếp tục.
- **Xóa hội thoại:** chỉ xóa giao diện chat, không xóa annotation CVAT.

Tool thường upload sau khi suy luận xong toàn lượt. Hủy không hoàn tác yêu cầu upload đã gửi tới CVAT. Chỉ chạy một lượt tại một thời điểm, tránh nhiều cửa sổ cùng cập nhật một job.

## 5. Chất lượng và tốc độ

- **SAM3 Nhanh:** một mô tả mỗi nhãn; có thể bỏ sót thêm.
- **SAM3 Kỹ:** nhiều mô tả với các nhãn có biến thể; tốn thời gian hơn.
- **Confidence:** ngưỡng lọc, không phải phần trăm chính xác. Ngưỡng thấp có thể tăng cả phát hiện lẫn nhận nhầm.
- **Điểm polygon/polyline:** giá trị thấp có thể làm polygon mất chi tiết hoặc chia polyline thành nhiều đoạn. `0` nghĩa là không giới hạn số điểm.
- Polygon lấy đường bao ngoài nên không biểu diễn đầy đủ lỗ bên trong; dùng Mask nếu cần giữ các lỗ này.

**Giới hạn hiện tại:** SAM3 chưa phân biệt đáng tin cậy `area/direct` và `area/alternative`. Các mask khác lớp vẫn có thể chồng nhau. Polyline từ mask có thể rất vụn, sai nhánh hoặc gán nhiều lớp lên cùng vạch; đây là tính năng thử nghiệm, không nên coi đầu ra là nhãn hoàn chỉnh. Giữ/Thêm/Reset không giải quyết lỗi phân loại hoặc chồng lấn giữa các lớp.

Nên kiểm tra trên job nhỏ trước khi chạy hàng loạt. Model không tự bảo đảm đúng guideline, thuộc tính occluded/truncated hay chất lượng pixel.

### Kiểm tra GPU

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Nếu CUDA là `False` dù máy có NVIDIA GPU, kiểm tra driver và cài bản PyTorch phù hợp theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/), dùng Python của `.venv`.

SAM3 tự chọn CPU khi GPU dưới 6 GB VRAM. Ngưỡng này là chính sách của tool, không phải bảo đảm GPU 6 GB luôn đủ. Có thể đặt trước khi mở tool:

```powershell
$env:SAM3_DEVICE = 'cpu'
$env:SAM3_CPU_THREADS = '8'
.\start-cvat.cmd
```

`SAM3_DEVICE=cuda:0` ép dùng GPU nhưng có thể gây hết VRAM. Không nên ép GPU 4 GB khi chưa thử trên job nhỏ.

## 6. Cập nhật code trên máy đã cài

Đợi lượt chạy hoàn tất hoặc hủy, sau đó tắt cửa sổ tool:

```powershell
cd C:\duong-dan\Auto-label-cvat
git pull origin main
.\.venv\Scripts\python.exe -m pip install -r requirements-sam3.txt
.\start-cvat.cmd
```

Thay đường dẫn bằng thư mục thực tế. F5 giao diện sau khi mở lại. Không cần tạo lại `.venv`, tải lại model hoặc đăng nhập Hugging Face nếu các file đã có. Nếu Git báo thay đổi local hoặc conflict, xử lý chúng trước; không xóa code đang sửa để ép pull.

## 7. Lỗi thường gặp

| Lỗi | Cách kiểm tra |
| --- | --- |
| `Failed to fetch` | Kiểm tra tool còn chạy không; mở lại `start-cvat.cmd`, truy cập đúng cổng 8787 và F5. |
| `py` không được nhận diện | Kiểm tra đã cài Python 3.12 và Python Launcher. |
| SAM3 thiếu trọng số | Chạy `download-sam3.cmd`; cần tài khoản đã được duyệt và token có quyền đọc model. |
| `Invalid token format` | Nhập toàn bộ token `hf_...`, không dán đường link hay mã device login. |
| Không tìm thấy Chrome/CDP | Cài Chrome, dùng nút mở profile của tool; kiểm tra cổng 9222 không bị chặn hoặc chiếm. |
| Không đăng nhập CVAT được | Kiểm tra đúng server và tài khoản; tài khoản online không tự dùng được trên CVAT local. |
| Polyline báo nhãn không hợp lệ | Chỉ nhập các nhãn `lane/...`, chọn SAM3. |
| Vẫn thấy nhãn ngoài yêu cầu | Có thể là nhãn cũ được giữ. Kiểm tra log nhãn được chọn; Reset chỉ thay nhãn trong lượt chạy. |
| Chạy chậm/hết bộ nhớ | Giảm số nhãn, thử Nhanh, đóng ứng dụng nặng; kiểm tra CPU/GPU trước khi chạy hàng loạt. |

Log lỗi giao diện nằm trong cửa sổ chạy tool; chi tiết lượt chạy có trong Activity log. Kết quả và backup nằm trong `runs/job_<id>/<model>_<timestamp>/`.

## 8. Dữ liệu và Git

`.gitignore` loại trừ `.venv/`, `models/`, `profile/`, `runs/`, `data/` và `.docker/`. Các thư mục này không được tải về khi clone. Không commit token, cookie, profile trình duyệt, ảnh hoặc annotation riêng tư.

SAM3 inference đọc trọng số local, không gửi ảnh sang Hugging Face/Meta để nhận diện. Tool vẫn tải ảnh từ và upload annotation về server CVAT bạn chọn. Điều này không có nghĩa toàn bộ workflow hoạt động không cần mạng.

## 9. Kiểm tra dành cho người phát triển

Cài dependencies SAM3 trước khi chạy toàn bộ tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m compileall -q cvat_agent yolo_agent
```
