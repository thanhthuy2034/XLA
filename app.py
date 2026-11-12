"""Local Flask entry point. The checkpoint is loaded once at startup."""
import logging

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 41 * 1024 * 1024
app.json.ensure_ascii = False
service = None
startup_error = None
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
    if service is None:
        return jsonify(error=f'Mô hình chưa sẵn sàng: {startup_error}'), 503
    try:
        images = [read_image(request.files.get(key), label) for key, label in [
            ('support_image', 'Ảnh hỗ trợ'), ('support_mask', 'Mặt nạ hỗ trợ'),
            ('query_image', 'Ảnh truy vấn')]]
        gt_upload = request.files.get('query_gt')
        gt = read_image(gt_upload, 'Query Ground Truth') if gt_upload and gt_upload.filename else None
        return jsonify(service.predict(*images, ground_truth=gt))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except RequestEntityTooLarge:
        raise
    except BadRequest:
        return jsonify(error='Dữ liệu multipart tải lên không hợp lệ.'), 400
    except Exception:
        app.logger.exception('Lỗi suy luận LoEC')
        return jsonify(error='Suy luận thất bại. Kiểm tra log backend để biết nguyên nhân; nếu thiếu bộ nhớ, thử ảnh nhỏ hơn hoặc chạy trên CPU.'), 500


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return jsonify(error='Tổng dung lượng yêu cầu vượt 41 MB; mỗi ảnh tối đa 10 MB.'), 413


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
