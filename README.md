# LoEC Segmentation

Demo phân đoạn ảnh ít mẫu xuyên miền bằng Python 3.11, PyTorch 2.6, torchvision 0.21, Flask, Pillow và NumPy. Giao diện Bootstrap/JavaScript, font Inter và tài nguyên CSS được giữ cục bộ.

## Chạy trên Windows

Mở PowerShell tại `LoEC-Demo`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Mở **http://127.0.0.1:5000**, đợi trạng thái **Sẵn sàng**. Chọn Support Image, Support Mask, Query Image rồi bấm **Bắt đầu phân đoạn**. Query Ground Truth là tùy chọn. Website hỗ trợ kéo thả/preview, hai kết quả trước/sau LCM, overlay, tải bốn PNG, bảng foreground IoU khi có GT và đặt lại.

Dùng Python trực tiếp trong `.venv` nên không cần kích hoạt môi trường. Nếu môi trường đặt `PIP_NO_INDEX=1`, chạy `$env:PIP_NO_INDEX='0'` trước khi cài. Cài thư viện cần Internet; khởi động/inference không tải ImageNet hay tài nguyên giao diện. CPU được hỗ trợ; backend tự chọn CUDA nếu bản PyTorch đã cài và GPU hỗ trợ CUDA. Flask tắt debug/reloader, nạp mô hình một lần mỗi tiến trình; inference dùng lock và CPU tối đa 4 thread.

## Mô hình và checkpoint

- `model/loec_model.py` chứa nguyên định nghĩa hai class ở cell 4 và 6 của `notebook59e1c34f21.ipynb` mới, không chạy mã huấn luyện/dataset của notebook.
- ResNet-50 dừng ở layer3, dilation/stride đúng notebook, **bỏ ReLU cuối**. Huấn luyện dùng **ImageNet-1K V1**, đóng băng layer0/layer1, giữ BatchNorm eval. Demo dùng `pretrained=False, refine=False` vì checkpoint chứa đầy đủ trọng số.
- Mặc định **`weights/last_loec_model.pth` — epoch 20**, là mô hình dùng đánh giá kết quả mới. `best_loec_model.pth` có training loss thấp nhất, ở epoch 18 theo log notebook; không phải checkpoint chọn bằng validation mIoU. Các file PTH là state dict trực tiếp gồm 258 key, không chứa metadata epoch; thông tin epoch lấy từ code và log notebook.
- Backend dùng Python `zipfile` trích checkpoint được chọn từ **`results.zip` hiện tại**. Cache trong `weights/` được so kích thước và CRC với ZIP; cache cũ/hỏng được giải nén lại qua file tạm rồi thay thế. Kiểm thử đồng bộ cả best và last với ZIP mới.
- Thiếu ZIP/checkpoint được chọn hoặc trọng số không tương thích: báo lỗi rõ ràng, `/health` và `/predict` trả 503; không đổi sang checkpoint khác. Nạp bằng `map_location`, `weights_only=True` khi hỗ trợ và **`strict=True`**, sau đó `eval()` và `torch.inference_mode()`.

Có thể chọn best bằng cấu hình backend trước khi khởi động, không thêm bộ chọn trên giao diện:

```powershell
$env:LOEC_CHECKPOINT='best_loec_model.pth'
.\.venv\Scripts\python.exe app.py
```

Trở lại mặc định bằng `Remove-Item Env:LOEC_CHECKPOINT -ErrorAction SilentlyContinue`. Chỉ chấp nhận hai tên best/last ở trên.

## Dữ liệu và kết quả

- Support/Query RGB resize **480 × 480**, bilinear, float32 [0,1], ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- Support Mask/Query GT resize nearest, nhãn 0/1. Nhận mask 0/1 và 0/255; mask xám 8-bit dùng ngưỡng `>127`. Không tự đảo màu. Support phải còn nền và tiền cảnh sau resize. Mask nhiều lớp cần chuyển thành mask của lớp muốn phân đoạn trước khi tải.
- Support là list tensor `[1,3,480,480]`, mask là list tensor `[1,480,480]`; Query `[1,3,480,480]`. **`q_ori` float32 `[1,480,480,3]` trong [0,1]**, RGB trước chuẩn hóa, đúng dataset notebook.
- Hai forward dùng chung dữ liệu/checkpoint, **`use_lem=False`**; lần lượt `use_mask_correction=False/True`. LCM giữ confidence foreground trừ background, patch 16 × 16 (lưới 30 × 30), Top-K=1, similarity RGB từ khoảng cách Euclidean bình phương chuẩn hóa theo từng patch. **`mask_k=0.6`, `mask_b=0.7`** đúng lần đánh giá trong notebook.
- Eval với `refine=False` trả tensor score **`[1,2,480,480]`**; mask lấy bằng argmax. LEM chỉ dùng khi huấn luyện. Các nhánh output huấn luyện/refinement được giữ theo notebook, demo không huấn luyện lại.
- Foreground IoU tính trực tiếp tại **480 × 480**, riêng cho hai prediction, chỉ khi có GT; hai vùng đều rỗng quy ước IoU=1. Đây là IoU của một cặp ảnh, không phải mIoU benchmark. Không có GT thì API trả `null` và ẩn bảng.
- Mask 0/255 và overlay xanh được xuất về **kích thước Query gốc**, giữ bố cục/tỷ lệ. LCM có thể tăng hoặc giảm IoU; luôn hiển thị kết quả thực.

Giới hạn 10 MB/ảnh, 41 MB/request, 20 triệu pixel/ảnh, chỉ JPG/JPEG/PNG tĩnh. Không lưu upload/kết quả lên đĩa.

## API và kiểm thử

- `GET /`: giao diện Bootstrap.
- `GET /health`: `{ready, device, checkpoint, error}`, HTTP 200 khi sẵn sàng, 503 khi lỗi khởi tạo.
- `POST /predict`: multipart với `support_image`, `support_mask`, `query_image`, tùy chọn `query_gt`. Trả `results.no_lcm`/`results.with_lcm` gồm `mask`, `overlay` (data URL PNG), `iou`, cùng `inputs`, `score_shapes`, `elapsed_seconds`, `device`, `iou_resolution`. Lỗi dữ liệu: 400; quá dung lượng: 413; lỗi inference: 500 và traceback trong terminal.

```powershell
.\.venv\Scripts\python.exe test_inference.py
```

10 kiểm thử đối chiếu AST hai class với notebook, byte cả hai checkpoint với ZIP, strict load, đóng băng/BatchNorm, tiền xử lý và phép tính LCM. API chạy **forward thật** có/không LCM và có/không GT, xác nhận dùng chung dữ liệu/mô hình, score, IoU, PNG và kích thước xuất. Kiểm tra thêm ảnh thiếu/hỏng, mask rỗng, dung lượng, cache cũ, ZIP/checkpoint thiếu, trọng số hỏng/sai kiến trúc. Các ca lỗi checkpoint dùng thư mục tạm, không sửa nguồn gốc.

Kiểm thử giao diện cần Chrome đã cài. Khi Flask đang chạy, mở terminal thứ hai:

```powershell
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe test_browser.py
```

Script Chrome headless gọi API thực, kiểm tra upload/preview, loading, hai kết quả, GT/IoU, tải bốn PNG, reset, lỗi ảnh, responsive và JavaScript errors. `browser-*.png` là ảnh chụp smoke test.

Đã xác minh trong workspace: **10/10 kiểm thử mô hình/API qua trên CPU**, strict load cả hai checkpoint không thiếu/thừa key, score hai chế độ `[1,2,480,480]`. Chrome headless qua các luồng trên ở chiều rộng 320–1920 px, không có JavaScript errors; `pip check` không phát hiện lỗi phụ thuộc. SHA-256 xác nhận notebook, ZIP, `LoEC.py` và mã giao diện gốc không đổi.

**Giới hạn:** ảnh tổng hợp chỉ xác minh đường chạy kỹ thuật, không chứng minh chất lượng. Chưa đánh giá lại benchmark hoặc kiểm thử CUDA. GT phải đúng nội dung/bố cục Query; backend không xác minh ngữ nghĩa này. `LoEC.py`, notebook và ZIP gốc được giữ nguyên; `LoEC.py` chỉ là tài liệu tham chiếu, không nằm trong đường chạy inference.
