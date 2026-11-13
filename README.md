# LoEC Segmentation Demo

Demo Flask đã cập nhật theo **`notebookf7987d0818.ipynb`**, so sánh ba cấu hình trên cùng ảnh và checkpoint:

| Kết quả | LCM | Top-K | Cập nhật score tiền cảnh |
| --- | --- | --- | --- |
| Không sử dụng LCM | Tắt | — | Giữ nguyên |
| LCM Cosine | Cosine giữa vector RGB trung bình của patch | 3 | Cộng tổng `0.6 × (similarity − 0.7)` của ba anchor |
| LCM Euclidean | Bình phương khoảng cách RGB, chuẩn hóa min–max | 1 | `score + score × (similarity − 0.7) × 0.6` |

Đây là các biến thể triển khai trong notebook. Demo không chạy các cell huấn luyện hoặc quét dataset Kaggle. LCM có thể tăng hoặc giảm IoU; giao diện hiển thị kết quả thực tế.

## 1. Tải checkpoint từ Kaggle

**Repository không kèm trọng số đã huấn luyện.** Khi thiếu checkpoint, trang vẫn mở nhưng báo chưa sẵn sàng và khóa nút dự đoán.

**Chọn tệp ngay trên giao diện:** chạy Flask, mở trang web, tại phần **Mô hình đã huấn luyện** chọn tệp `.pth` ở bất kỳ thư mục nào rồi bấm **Nạp mô hình**. Không cần đổi tên file, chép vào dự án hoặc khởi động lại server. Chấp nhận state dict của mô hình trong notebook, tối đa 256 MB; tệp được nạp với `weights_only=True` và `strict=True`. Nếu chọn sai tệp, mô hình trước đó vẫn được giữ. Tệp chọn qua giao diện chỉ có hiệu lực trong lần chạy server hiện tại; lần sau chọn lại hoặc dùng cách đặt file cố định dưới đây.

1. Mở notebook tương ứng trên Kaggle. Nếu chưa có kết quả huấn luyện, gắn dataset cần thiết rồi chạy phần huấn luyện; nên dùng GPU Kaggle.
2. Sau khi chạy xong, mở phần **Output** của phiên bản notebook đã lưu và tải **`last_loec_model.pth`** từ `/kaggle/working/`. Notebook lưu file này sau khi hoàn tất vòng huấn luyện. File `.ipynb` và ảnh kết quả không chứa trọng số.
3. Tạo thư mục `weights` tại dự án và chép file vào đúng đường dẫn:

```text
D:\DEV\XLA\weights\last_loec_model.pth
```

```powershell
New-Item -ItemType Directory -Force weights
```

Nếu tải **`results.zip`**, có thể đặt ZIP tại `D:\DEV\XLA\results.zip`; file `last_loec_model.pth` phải nằm ngay ở gốc ZIP. Backend tự giải nén vào `weights/`. Khi ZIP tồn tại, ZIP là nguồn ưu tiên; cache được đối chiếu kích thước/CRC và cập nhật nếu khác. Khi không có ZIP, backend dùng file trong `weights/`.

Có thể dùng `best_loec_model.pth` bằng cách đặt file tương ứng trong `weights/` hoặc ZIP và cấu hình trước khi chạy:

```powershell
$env:LOEC_CHECKPOINT='best_loec_model.pth'
```

Notebook chọn best theo **training loss**, không phải validation mIoU. Mặc định demo dùng last. Để trở về mặc định:

```powershell
Remove-Item Env:LOEC_CHECKPOINT -ErrorAction SilentlyContinue
```

Dùng trọng số từ lần huấn luyện bạn muốn đối chiếu trong notebook mới; tên file giống nhau không chứng minh cùng lần huấn luyện. Backend nạp state dict bằng `strict=True`, không thay bằng trọng số ngẫu nhiên khi thiếu hoặc lỗi checkpoint.

## 2. Cài đặt và chạy trên Windows

Dùng **Python 3.11**, mở PowerShell tại dự án:

```powershell
Set-Location D:\DEV\XLA
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Mở **http://127.0.0.1:5000**. Khi có checkpoint hợp lệ, trạng thái chuyển thành **Sẵn sàng**. Dừng server bằng `Ctrl+C`. Sau khi thêm/đổi checkpoint, khởi động lại server rồi tải lại trang.

Các lệnh trên cài bản CPU và không cần kích hoạt môi trường ảo. Nếu Python không được nhận diện, dùng `py -3.11 -m venv .venv` thay lệnh tạo môi trường. Nếu pip bị cấu hình `PIP_NO_INDEX=1`, đặt `$env:PIP_NO_INDEX='0'` trước khi cài. Cài thư viện cần Internet; chạy demo với checkpoint có sẵn không tải thêm ImageNet hoặc tài nguyên giao diện. Backend tự chọn CUDA nếu bản PyTorch đã cài hỗ trợ CUDA và GPU khả dụng; hướng dẫn trên dùng CPU.

Kiểm tra trạng thái từ terminal khác:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Nếu `ready=false`, đọc trường `error` để biết checkpoint còn thiếu hoặc lỗi trọng số. Nếu cổng 5000 đang bận, dừng tiến trình Flask cũ trước khi chạy lại.

## 3. Thử với ảnh mẫu

Trong `demo_samples/chest_xray/`, chọn:

| Ô trên giao diện | File |
| --- | --- |
| Ảnh hỗ trợ | `support_image.png` |
| Mặt nạ hỗ trợ | `support_mask.png` |
| Ảnh truy vấn | `query_image.png` |
| Mặt nạ tham chiếu (tùy chọn) | `query_ground_truth.png` |

Bấm **Bắt đầu phân đoạn**, đợi ba lượt suy luận. Giao diện hiển thị mask, overlay và nút tải PNG cho từng cấu hình (tổng sáu PNG). Khi có Ground Truth, bảng hiển thị foreground IoU của cả ba kết quả.

Với ảnh riêng: support và query cần cùng loại đối tượng muốn phân đoạn. Support mask phải khớp ảnh support, nền đen và đối tượng trắng, có cả nền lẫn tiền cảnh. Chấp nhận mask nhị phân 0/1 hoặc 0/255; ảnh xám dùng ngưỡng >127. Mask nhiều lớp hoặc bảng màu cần chuyển thành mask nhị phân của lớp cần phân đoạn trước. GT phải khớp ảnh query và dùng cùng quy ước nhãn.

Giới hạn: JPG/JPEG/PNG tĩnh, 10 MB/ảnh, 41 MB/request, 20 triệu pixel/ảnh. Upload và kết quả không lưu xuống đĩa.

## Chi tiết triển khai

- `model/loec_model.py`: giữ nguyên định nghĩa hai class ở cell 4 và 6 của notebook mới. ResNet-50 dừng ở layer3, điều chỉnh dilation/stride, bỏ ReLU cuối. Khởi tạo `pretrained=False, refine=False` rồi nạp checkpoint.
- `services/inference.py`: chuẩn hóa ảnh RGB 480 × 480 bằng ImageNet mean/std; mask resize nearest. `q_ori` có dạng `[1,480,480,3]`, float32 trong [0,1]. Ba lượt forward dùng cùng input, `use_lem=False`, `mask_k=0.6`, `mask_b=0.7`, patch 16 × 16. Chạy `eval()` và `torch.inference_mode()` dưới lock.
- Score mỗi cấu hình là `[1,2,480,480]`; mask lấy argmax. Foreground IoU tính tại 480 × 480; hai vùng đều rỗng quy ước IoU=1. Đây là IoU của một cặp ảnh, không phải benchmark mIoU. PNG xuất về kích thước query gốc.
- `app.py`: nạp mô hình một lần khi khởi động. `LoEC.py` và hai notebook làm tài liệu tham chiếu, không được chạy khi mở demo.

## API và kiểm tra

- `GET /`: giao diện.
- `GET /health`: `{ready, device, checkpoint, error}`, HTTP 200 khi sẵn sàng hoặc 503 khi khởi tạo lỗi.
- `POST /checkpoint`: multipart gồm trường `checkpoint` chứa tệp `.pth`, tối đa 256 MB. Nạp mô hình và trả trạng thái sẵn sàng; không ghi tệp vào thư mục dự án. Tệp lỗi trả 400, đang nạp yêu cầu khác trả 409, quá dung lượng trả 413.
- `POST /predict`: multipart gồm `support_image`, `support_mask`, `query_image`, tùy chọn `query_gt`. Trả `results` và `score_shapes` với ba key **`no_lcm`, `lcm_cosine`, `lcm_euclidean`**; mỗi kết quả gồm `mask`, `overlay`, `iou`. Thiếu GT thì IoU là `null`. Key cũ `with_lcm` được thay bằng hai cấu hình rõ tên.

```powershell
.\.venv\Scripts\python.exe test_inference.py
```

Bộ kiểm tra đối chiếu AST class với notebook mới, tiền xử lý, anchor/similarity, công thức cập nhật cả hai LCM, ba kết quả API có/không GT, PNG, checkpoint và lỗi khởi tạo. Khi chưa tải trọng số, các ca cần checkpoint huấn luyện được đánh dấu **skipped**; một ca riêng dùng state dict ngẫu nhiên trong thư mục tạm để kiểm tra đường chạy kỹ thuật. Ca đó không đánh giá chất lượng và không ghi trọng số ngẫu nhiên vào `weights/` của dự án.

Kiểm tra giao diện tùy chọn, cần Chrome, checkpoint đã tải và server đang chạy:

```powershell
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe test_browser.py
```

Smoke test không kiểm chứng chất lượng benchmark hoặc CUDA.
