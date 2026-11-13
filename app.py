"""Local Flask entry point. The checkpoint is loaded once at startup."""
import logging
import threading

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 41 * 1024 * 1024
app.json.ensure_ascii = False
service = None
startup_error = None
checkpoint_lock = threading.Lock()
try:
    from services.inference import InferenceService, read_image
    service = InferenceService()
except Exception as exc:
    startup_error = str(exc)
    logging.exception('Không thể khởi tạo mô hình LoEC')


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/health')
def health():
    return jsonify(ready=service is not None, device=str(service.device) if service else None,
                   checkpoint=service.checkpoint_path.name if service else None,
                   error=startup_error), 200 if service else 503


@app.post('/predict')
def predict():
    active_service = service
    if active_service is None:
        return jsonify(error=f'Mô hình chưa sẵn sàng: {startup_error}'), 503
    try:
        images = [read_image(request.files.get(key), label) for key, label in [
            ('support_image', 'Ảnh hỗ trợ'), ('support_mask', 'Mặt nạ hỗ trợ'),
            ('query_image', 'Ảnh truy vấn')]]
        gt_upload = request.files.get('query_gt')
        gt = read_image(gt_upload, 'Query Ground Truth') if gt_upload and gt_upload.filename else None
        return jsonify(active_service.predict(*images, ground_truth=gt))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except RequestEntityTooLarge:
        raise
    except BadRequest:
        return jsonify(error='Dữ liệu multipart tải lên không hợp lệ.'), 400
    except Exception:
        app.logger.exception('Lỗi suy luận LoEC')
        return jsonify(error='Suy luận thất bại. Kiểm tra log backend để biết nguyên nhân; nếu thiếu bộ nhớ, thử ảnh nhỏ hơn hoặc chạy trên CPU.'), 500


@app.post('/checkpoint')
def load_checkpoint():
    global service, startup_error
    # Checkpoint files are much larger than image requests.
    request.max_content_length = 257 * 1024 * 1024
    if not checkpoint_lock.acquire(blocking=False):
        return jsonify(error='Đang nạp mô hình. Vui lòng chờ hoàn tất.'), 409
    try:
        upload = request.files.get('checkpoint')
        if upload is None or not upload.filename:
            return jsonify(error='Hãy chọn tệp checkpoint .pth đã tải từ notebook.'), 400
        name = secure_filename(upload.filename)
        if not name.lower().endswith('.pth'):
            return jsonify(error='Checkpoint phải là tệp .pth (không chọn .ipynb hoặc .zip).'), 400
        upload.stream.seek(0, 2)
        size = upload.stream.tell()
        upload.stream.seek(0)
        if size == 0 or size > 256 * 1024 * 1024:
            return jsonify(error='Checkpoint phải có dữ liệu và không vượt quá 256 MB.'), 400
        # Replace only after validation succeeds; a bad upload keeps the current model.
        replacement = InferenceService(checkpoint_stream=upload.stream, checkpoint_name=name)
        service = replacement
        startup_error = None
        return jsonify(ready=True, checkpoint=name, device=str(service.device), error=None)
    except RequestEntityTooLarge:
        raise
    except (RuntimeError, ValueError, BadRequest):
        app.logger.exception('Không thể nạp checkpoint đã chọn')
        return jsonify(error='Không nạp được checkpoint. Hãy chọn state dict .pth được lưu bằng torch.save(model.state_dict(), ...) trong notebook LoEC.'), 400
    except Exception:
        app.logger.exception('Lỗi nạp checkpoint')
        return jsonify(error='Nạp mô hình thất bại. Kiểm tra log backend và bộ nhớ khả dụng.'), 500
    finally:
        checkpoint_lock.release()


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    if request.endpoint == 'load_checkpoint':
        return jsonify(error='Tệp checkpoint vượt giới hạn 256 MB.'), 413
    return jsonify(error='Tổng dung lượng yêu cầu vượt 41 MB; mỗi ảnh tối đa 10 MB.'), 413


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
