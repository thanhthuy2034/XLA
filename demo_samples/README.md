# Các bộ ảnh demo

Mỗi thư mục là một lần thử riêng. Chọn các file trong **cùng thư mục**:

| Ô trên giao diện | File |
| --- | --- |
| Ảnh hỗ trợ | `support_image.png` |
| Mặt nạ hỗ trợ | `support_mask.png` |
| Ảnh truy vấn | `query_image.png` |
| Mặt nạ tham chiếu, tùy chọn để tính IoU | `query_ground_truth.png` |

Các bộ bổ sung:

- **`pedestrians_fudan/`**: tìm người đi bộ, hai ảnh khác nhau trong Penn-Fudan (`FudanPed00001`, `FudanPed00002`).
- **`pedestrians_penn/`**: tìm người đi bộ ngoài trời, nhiều người trong ảnh (`PennPed00015`, `PennPed00016`).
- **`eiffel_tower/`**: tìm tháp Eiffel giữa hai ảnh khác góc nhìn/ánh sáng, từ ảnh ví dụ trong repository FSS-1000 (ảnh 1 và 4).
- `chest_xray/`: bộ ảnh phổi có sẵn trước đó.

Mask bổ sung được chuyển từ nhãn nguồn sang PNG đen/trắng: nền bằng 0, mọi nhãn đối tượng khác 0 bằng 255. Với Penn-Fudan, các instance người được gộp thành một lớp tiền cảnh. Không dùng phép chuyển ảnh palette sang ảnh xám để suy ra class ID. Ảnh và mask giữ nguyên kích thước nguồn, không cắt ghép.

Nên bắt đầu với `pedestrians_fudan/`. Đây là dữ liệu demo, không phải tập đánh giá benchmark; kết quả tùy checkpoint và ảnh được chọn. Không trộn mask của bộ này với ảnh hỗ trợ của bộ khác.

## Nguồn

- [FSS-1000 — HKUSTCV](https://github.com/HKUSTCV/FSS-1000/tree/master/imgs/example/support): ảnh Eiffel và mask ví dụ do dự án cung cấp.
- [Penn-Fudan — trang dữ liệu gốc](https://www.cis.upenn.edu/~jshi/ped_html/): dữ liệu người đi bộ và nhãn phân đoạn.
- [Bản sao Penn-Fudan dùng tải các file](https://github.com/Charan-Sammeta/2D_polygon_semantic_segmentation/tree/b08f7ba46fe82eec32744146d6dd7360df03a651/archive/PennFudanPed): tải riêng các cặp ảnh/mask thay vì cả archive.

File nguồn đã tải và bảng đối chiếu nằm trong `_sources/manifest.json`. Các quyền đối với ảnh vẫn thuộc tác giả/chủ sở hữu nguồn; không có tuyên bố cấp phép lại ảnh trong dự án này.
