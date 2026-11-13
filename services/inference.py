"""Preprocessing and three-way inference using the notebook checkpoint."""
import base64
import inspect
import io
import os
from pathlib import Path
import shutil
import threading
import time
import warnings
import zipfile
import zlib

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from model.loec_model import SSP_MatchingNet

ROOT = Path(__file__).resolve().parents[1]
SIZE = (480, 480)
DEFAULT_CHECKPOINT = 'last_loec_model.pth'
CHECKPOINT_NAMES = {'last_loec_model.pth', 'best_loec_model.pth'}
MASK_K = 0.6
MASK_B = 0.7
VARIANTS = {
    "no_lcm": dict(use_mask_correction=False),
    "lcm_cosine": dict(use_mask_correction=True, lcm_method="cosine",
                       lcm_k=3, lcm_update_mode="additive"),
    "lcm_euclidean": dict(use_mask_correction=True, lcm_method="euclidean",
                          lcm_k=1, lcm_update_mode="multiplicative"),
}
MAX_FILE_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def extract_checkpoint(checkpoint_name=None):
    """Use the selected member of the current archive; refresh stale cached weights."""
    name = checkpoint_name if checkpoint_name is not None else os.environ.get('LOEC_CHECKPOINT', DEFAULT_CHECKPOINT)
    if name not in CHECKPOINT_NAMES:
        raise ValueError('LOEC_CHECKPOINT chỉ nhận last_loec_model.pth hoặc best_loec_model.pth.')
    archive_path = ROOT / 'results.zip'
    target = ROOT / 'weights' / name
    if not archive_path.is_file():
        if target.is_file():
            return target
        raise FileNotFoundError(
            f'Chưa có checkpoint. Tải {name} từ Output của notebook Kaggle, '
            f'đặt tại {target}; hoặc đặt results.zip tại {ROOT}.')
    with zipfile.ZipFile(archive_path) as archive:
        try:
            member = archive.getinfo(name)
        except KeyError as exc:
            raise FileNotFoundError(f'Không có checkpoint {name} trong {archive_path.name}; không dùng trọng số thay thế.') from exc
        # A matching filename alone cannot distinguish old and newly trained weights.
        if target.is_file() and target.stat().st_size == member.file_size:
            checksum = 0
            with target.open('rb') as cached:
                for chunk in iter(lambda: cached.read(1024 * 1024), b''):
                    checksum = zlib.crc32(chunk, checksum)
            if checksum == member.CRC:
                return target
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_suffix('.tmp')
        try:
            with archive.open(member) as source, temporary.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def read_image(upload, label):
    if upload is None or not upload.filename:
        raise ValueError(f'Thiếu tệp: {label}.')
    if Path(upload.filename).suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
        raise ValueError(f'{label}: chỉ hỗ trợ JPG, JPEG và PNG.')
    data = upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f'{label}: dung lượng tối đa là 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {'PNG', 'JPEG'}:
                    raise ValueError(f'{label}: nội dung tệp phải là PNG hoặc JPEG.')
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError(f'{label}: không hỗ trợ ảnh động.')
                return ImageOps.exif_transpose(image).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError(f'{label}: ảnh không hợp lệ hoặc vượt giới hạn 20 triệu pixel.') from exc


def rgb_tensor(image):
    resized = image.convert('RGB').resize(SIZE, Image.Resampling.BILINEAR)
    raw = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    normalized = ((raw - mean) / std).permute(2, 0, 1).unsqueeze(0)
    return normalized, raw.unsqueeze(0), resized


def binary_mask(image, require_foreground=False):
    # Convert palette colors as well as grayscale; preserve literal 0/1 masks.
    gray = np.asarray(image.convert('L'))
    threshold = 0 if gray.max() <= 1 else 127
    binary = Image.fromarray((gray > threshold).astype(np.uint8))
    array = np.array(binary.resize(SIZE, Image.Resampling.NEAREST), dtype=np.int64)
    if require_foreground and not array.any():
        raise ValueError(f'Mặt nạ hỗ trợ không có vùng tiền cảnh sau khi resize {SIZE[0]} × {SIZE[1]}.')
    if require_foreground and array.all():
        raise ValueError('Mặt nạ hỗ trợ cần có cả vùng nền và vùng tiền cảnh.')
    return array


def png_data(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def foreground_iou(prediction, target):
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


class InferenceService:
    def __init__(self, checkpoint_stream=None, checkpoint_name=None):
        self.lock = threading.Lock()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.checkpoint_path = Path(checkpoint_name or 'selected_model.pth') if checkpoint_stream is not None else extract_checkpoint()
        self.model = SSP_MatchingNet(pretrained=False, refine=False)
        kwargs = {'map_location': self.device}
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = True
        elif checkpoint_stream is not None:
            raise RuntimeError('Cần PyTorch hỗ trợ weights_only=True để nạp tệp đã chọn.')
        try:
            state = torch.load(checkpoint_stream if checkpoint_stream is not None else self.checkpoint_path, **kwargs)
            self.load_result = self.model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise RuntimeError(f'Không thể nạp checkpoint {self.checkpoint_path.name} bằng strict=True: {exc}') from exc
        self.model.to(self.device).eval()

    def predict(self, support, support_mask, query, ground_truth=None):
        s, _, support_view = rgb_tensor(support)
        q, q_ori, query_view = rgb_tensor(query)
        mask = binary_mask(support_mask, require_foreground=True)
        gt = binary_mask(ground_truth) if ground_truth is not None else None
        s, q, q_ori = s.to(self.device), q.to(self.device), q_ori.to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        outputs, shapes = {}, {}
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            for name, config in VARIANTS.items():
                score = self.model([s], [mask_tensor], q, **config,
                                   mask_k=MASK_K, mask_b=MASK_B, q_ori=q_ori, use_lem=False)
                if not isinstance(score, torch.Tensor) or tuple(score.shape) != (1, 2, *SIZE) or not torch.isfinite(score).all():
                    raise RuntimeError('Mô hình trả về score sai kích thước hoặc chứa NaN/Inf.')
                shapes[name] = list(score.shape)
                prediction = torch.argmax(score, dim=1)[0].cpu().numpy().astype(np.uint8)
                # Return masks/overlays at the original Query resolution.
                mask_image = Image.fromarray(prediction * 255).resize(query.size, Image.Resampling.NEAREST)
                original = np.array(query.convert('RGB'), dtype=np.float32)
                foreground = np.asarray(mask_image) > 0
                original[foreground] = original[foreground] * 0.55 + np.array([30, 144, 255]) * 0.45
                outputs[name] = {
                    'mask': png_data(mask_image),
                    'overlay': png_data(Image.fromarray(original.astype(np.uint8))),
                    'iou': foreground_iou(prediction, gt) if gt is not None else None,
                }
        return {'results': outputs, 'score_shapes': shapes,
                'elapsed_seconds': round(time.perf_counter() - started, 3),
                'device': str(self.device), 'iou_resolution': list(SIZE) if gt is not None else None,
                'inputs': {'support': png_data(support_view),
                           'support_mask': png_data(Image.fromarray(mask.astype(np.uint8) * 255)),
                           'query': png_data(query_view)}}
