"""Technical smoke/API tests; synthetic fixtures do not evaluate model quality."""
import ast
import base64
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
from werkzeug.datastructures import FileStorage

from app import app, service
from model.loec_model import SSP_MatchingNet
from services.inference import binary_mask, foreground_iou, read_image, rgb_tensor
import services.inference as inference


def image_bytes(image):
    stream = io.BytesIO()
    image.save(stream, 'PNG')
    stream.seek(0)
    return stream


class LoECTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model = service.model if service else SSP_MatchingNet(pretrained=False).eval()
        cls.client = app.test_client()
        rng = np.random.default_rng(42)
        cls.support = Image.fromarray(rng.integers(0, 256, (180, 240, 3), dtype=np.uint8))
        cls.query = Image.fromarray(rng.integers(0, 256, (160, 220, 3), dtype=np.uint8))
        mask = np.zeros((180, 240), dtype=np.uint8)
        mask[40:140, 60:180] = 255
        cls.mask = Image.fromarray(mask)

    def payload(self, gt=False):
        data = {'support_image': (image_bytes(self.support), 'support.png'),
                'support_mask': (image_bytes(self.mask), 'mask.png'),
                'query_image': (image_bytes(self.query), 'query.png')}
        if gt:
            data['query_gt'] = (image_bytes(self.mask), 'ground_truth.png')
        return data

    def test_exact_notebook_extraction(self):
        notebook = json.loads(Path('notebookf7987d0818.ipynb').read_text(encoding='utf-8'))
        extracted = ast.parse(Path('model/loec_model.py').read_text(encoding='utf-8'))
        for node in extracted.body:
            if not isinstance(node, ast.ClassDef):
                continue
            originals = [n for cell in notebook['cells'] if cell['cell_type'] == 'code'
                         for n in ast.parse(''.join(cell['source'])).body
                         if isinstance(n, ast.ClassDef) and n.name == node.name]
            self.assertEqual(ast.dump(node), ast.dump(originals[0]))

    @unittest.skipIf(service is None, 'Requires downloaded trained checkpoint')
    def test_checkpoint_and_strict_load(self):
        state = torch.load(service.checkpoint_path, map_location='cpu', weights_only=True)
        result = SSP_MatchingNet(pretrained=False, refine=False).load_state_dict(state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(len(state), 258)
        if (inference.ROOT / 'results.zip').is_file():
            with zipfile.ZipFile(inference.ROOT / 'results.zip') as archive:
                self.assertEqual(service.checkpoint_path.read_bytes(), archive.read(service.checkpoint_path.name))
        self.assertEqual(service.load_result.missing_keys, [])
        self.assertEqual(service.load_result.unexpected_keys, [])
        self.assertFalse(service.model.training)
        self.assertFalse(service.model.refine)
        for layer in [service.model.layer0, service.model.layer1]:
            self.assertTrue(all(not p.requires_grad for p in layer.parameters()))

    def test_frozen_backbone_and_batchnorm(self):
        model = self.model
        try:
            model.train()
            self.assertFalse(model.layer0.training)
            self.assertFalse(model.layer1.training)
            self.assertTrue(model.layer2.training)
            self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)))
        finally:
            model.eval()

    def test_preprocessing(self):
        normalized, raw, _ = rgb_tensor(self.support)
        self.assertEqual(tuple(normalized.shape), (1, 3, 480, 480))
        self.assertEqual(tuple(raw.shape), (1, 480, 480, 3))
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(raw.dtype, torch.float32)
        self.assertTrue(0 <= raw.min() <= raw.max() <= 1)
        reconstructed = normalized[0].permute(1, 2, 0) * torch.tensor([.229, .224, .225]) + torch.tensor([.485, .456, .406])
        torch.testing.assert_close(raw[0], reconstructed)
        one_mask = Image.fromarray((np.asarray(self.mask) > 0).astype(np.uint8))
        np.testing.assert_array_equal(binary_mask(one_mask), binary_mask(self.mask))
        with self.assertRaises(ValueError):
            binary_mask(Image.new('L', (100, 100)), require_foreground=True)
        self.assertEqual(foreground_iou(np.zeros((2, 2)), np.zeros((2, 2))), 1)
        self.assertAlmostEqual(foreground_iou(np.array([1, 1, 0]), np.array([1, 0, 1])), 1/3)

    def test_home_health_and_static(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/health').status_code, 200 if service else 503)
        self.assertEqual(self.client.get('/health').json['ready'], service is not None)
        html = self.client.get('/').get_data(as_text=True)
        for key in inference.VARIANTS:
            self.assertIn('id="mask-' + key + '"', html)
        for path in ['/static/css/style.css', '/static/css/bootstrap.min.css', '/static/js/app.js']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_lcm_patch_selection_and_rgb_similarity(self):
        # Highest confidence at patch (1, 1), independent of the RGB similarity.
        score = torch.zeros(1, 2, 480, 480)
        score[:, 1, 16:32, 16:32] = 5
        rgb = torch.zeros(1, 480, 480, 3)
        rgb[:, 16:32, 16:32] = 1
        rgb[:, :16, 16:32] = 0.5
        indices, similarity = self.model.mask_correction(score, rgb, K=1, method='euclidean')
        self.assertEqual(indices.tolist(), [[31]])
        self.assertEqual(tuple(similarity.shape), (1, 1, 30, 30))
        self.assertAlmostEqual(similarity[0, 0, 1, 1].item(), 1.0)
        self.assertAlmostEqual(similarity[0, 0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(similarity[0, 0, 0, 1].item(), 0.75)
        other_indices, other_similarity = self.model.mask_correction(score, rgb.permute(0, 3, 1, 2), K=1, method='euclidean')
        torch.testing.assert_close(indices, other_indices)
        torch.testing.assert_close(similarity, other_similarity)

    @unittest.skipIf(service is None, 'Requires downloaded trained checkpoint')
    def test_real_paired_inference_and_downloads(self):
        # Both requests use the real checkpoint and real forward, no API mocking.
        model_id = id(service.model)
        for use_gt in [False, True]:
            calls = []
            def capture(model, args, kwargs, score):
                calls.append((args, kwargs, score.detach().cpu().clone()))
            handle = service.model.register_forward_hook(capture, with_kwargs=True)
            try:
                response = self.client.post('/predict', data=self.payload(use_gt))
            finally:
                handle.remove()
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(id(service.model), model_id)
            self.assertEqual(len(calls), 3)
            for i in range(3):
                first, second = calls[0][0][i], calls[1][0][i]
                self.assertIs(first[0] if isinstance(first, list) else first,
                              second[0] if isinstance(second, list) else second)
            self.assertIs(calls[0][1]['q_ori'], calls[1][1]['q_ori'])
            for call, use_lcm in zip(calls, [False, True, True]):
                self.assertEqual(call[1]['use_mask_correction'], use_lcm)
                self.assertFalse(call[1]['use_lem'])
                self.assertEqual(call[1]['mask_k'], 0.6)
                self.assertEqual(call[1]['mask_b'], 0.7)
                self.assertIsInstance(call[2], torch.Tensor)
                self.assertTrue(torch.isfinite(call[2]).all())
            plain, corrected = calls[0][2], calls[2][2]
            with torch.inference_mode():
                _, similarity = self.model.mask_correction(plain, calls[0][1]['q_ori'].cpu(), K=1, method='euclidean')
                similarity = torch.nn.functional.interpolate(similarity, (480, 480), mode='nearest')
            torch.testing.assert_close(corrected[:, 0], plain[:, 0])
            torch.testing.assert_close(corrected[:, 1], plain[:, 1] + plain[:, 1] * (similarity[:, 0] - 0.7) * 0.6)
            result = response.json
            self.assertEqual(result['iou_resolution'], [480, 480] if use_gt else None)
            for variant, call in zip(inference.VARIANTS, calls):
                self.assertEqual(result['score_shapes'][variant], [1, 2, 480, 480])
                item = result['results'][variant]
                prediction = call[2].argmax(1)[0].numpy().astype(np.uint8)
                if use_gt:
                    gt = binary_mask(self.mask)
                    intersection = np.count_nonzero((prediction == 1) & (gt == 1))
                    union = np.count_nonzero((prediction == 1) | (gt == 1))
                    self.assertEqual(item['iou'], intersection / union if union else 1.0)
                else:
                    self.assertIsNone(item['iou'])
                for kind in ['mask', 'overlay']:
                    image = Image.open(io.BytesIO(base64.b64decode(item[kind].split(',')[1])))
                    self.assertEqual(image.format, 'PNG')
                    self.assertEqual(image.size, self.query.size)
                    if kind == 'mask':
                        self.assertTrue(set(np.unique(np.array(image))) <= {0, 255})
                        expected = Image.fromarray(prediction * 255).resize(self.query.size, Image.Resampling.NEAREST)
                        np.testing.assert_array_equal(np.array(image), np.array(expected))
            print(f'Paired real inference: GT={use_gt}, device={result["device"]}, seconds={result["elapsed_seconds"]}')

    @unittest.skipIf(service is None, 'Requires downloaded trained checkpoint')
    def test_input_errors(self):
        self.assertEqual(self.client.post('/predict', data={}).status_code, 400)
        data = self.payload()
        data['query_image'] = (io.BytesIO(b'not an image'), 'broken.png')
        self.assertEqual(self.client.post('/predict', data=data).status_code, 400)
        data = self.payload()
        data['support_mask'] = (image_bytes(Image.new('L', (40, 40))), 'empty.png')
        self.assertEqual(self.client.post('/predict', data=data).status_code, 400)
        data = self.payload()
        data['query_image'] = (image_bytes(self.query), 'query.gif')
        self.assertEqual(self.client.post('/predict', data=data).status_code, 400)
        with self.assertRaises(ValueError):
            read_image(FileStorage(stream=io.BytesIO(b'x' * (10 * 1024 * 1024 + 1)), filename='large.png'), 'Test')
        response = self.client.post('/predict', data=b'x' * (41 * 1024 * 1024 + 1), content_type='application/octet-stream')
        self.assertEqual(response.status_code, 413)

    def test_checkpoint_extraction_refresh_and_missing_member(self):
        original_root = inference.ROOT
        try:
            with tempfile.TemporaryDirectory() as folder:
                inference.ROOT = Path(folder)
                with zipfile.ZipFile(inference.ROOT / 'results.zip', 'w') as archive:
                    for name in inference.CHECKPOINT_NAMES:
                        archive.writestr(name, b'checkpoint extraction fixture')
                target = inference.extract_checkpoint()
                self.assertEqual(target.name, 'last_loec_model.pth')
                self.assertEqual(target.read_bytes(), b'checkpoint extraction fixture')
                target.write_bytes(b'stale cached weights')
                self.assertEqual(inference.extract_checkpoint().read_bytes(), b'checkpoint extraction fixture')
                with patch.dict('os.environ', {'LOEC_CHECKPOINT': 'best_loec_model.pth'}):
                    self.assertEqual(inference.extract_checkpoint().name, 'best_loec_model.pth')
                with patch.dict('os.environ', {'LOEC_CHECKPOINT': 'unknown.pth'}):
                    with self.assertRaisesRegex(ValueError, 'LOEC_CHECKPOINT'):
                        inference.extract_checkpoint()
                # A cached last file and a best file are never a fallback for a missing ZIP member.
                with zipfile.ZipFile(inference.ROOT / 'results.zip', 'w') as archive:
                    archive.writestr('best_loec_model.pth', b'not the selected checkpoint')
                with self.assertRaisesRegex(FileNotFoundError, 'last_loec_model.pth'):
                    inference.extract_checkpoint()
                (inference.ROOT / 'results.zip').unlink()
                self.assertEqual(inference.extract_checkpoint(), target)
                target.unlink()
                with self.assertRaisesRegex(FileNotFoundError, 'checkpoint'):
                    inference.extract_checkpoint()
        finally:
            inference.ROOT = original_root

    def test_corrupt_and_incompatible_checkpoint_startup(self):
        original_root = inference.ROOT
        try:
            with tempfile.TemporaryDirectory() as folder:
                inference.ROOT = Path(folder)
                incompatible = io.BytesIO()
                torch.save({'unrelated.weight': torch.zeros(1)}, incompatible)
                for contents in [b'corrupt checkpoint', incompatible.getvalue()]:
                    with zipfile.ZipFile(inference.ROOT / 'results.zip', 'w') as archive:
                        archive.writestr('last_loec_model.pth', contents)
                    with self.assertLogs(level='ERROR'):
                        namespace = runpy.run_path(str(original_root / 'app.py'), run_name='checkpoint_failure_test')
                    self.assertIsNone(namespace['service'])
                    client = namespace['app'].test_client()
                    self.assertEqual(client.get('/health').status_code, 503)
                    self.assertFalse(client.get('/health').json['ready'])
                    self.assertIn('last_loec_model.pth', client.get('/health').json['error'])
                    self.assertIn('strict=True', client.get('/health').json['error'])
                    self.assertEqual(client.post('/predict').status_code, 503)
                    self.assertEqual(client.get('/').status_code, 200)
        finally:
            inference.ROOT = original_root

    def test_cosine_anchor_similarity(self):
        score = torch.zeros(1, 2, 32, 32)
        score[:, 1, :16, :16] = 3
        score[:, 1, :16, 16:] = 2
        score[:, 1, 16:, :16] = 1
        rgb = torch.zeros(1, 32, 32, 3)
        rgb[:, :16, :16, 0] = 1
        rgb[:, :16, 16:, 1] = 1
        rgb[:, 16:, :16, 2] = 1
        rgb[:, 16:, 16:, :2] = 1
        indices, similarity = self.model.mask_correction(score, rgb)
        self.assertEqual(indices.tolist(), [[0, 1, 2]])
        expected = torch.tensor([[[[1., 0.], [0., 2 ** -.5]],
                                  [[0., 1.], [0., 2 ** -.5]],
                                  [[0., 0.], [1., 0.]]]])
        torch.testing.assert_close(similarity, expected, atol=1e-6, rtol=1e-5)

    def test_three_way_api_with_temporary_random_weights(self):
        # Technical execution only: never install random weights in the demo.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'weights').mkdir()
            torch.save(self.model.state_dict(), root / 'weights' / inference.DEFAULT_CHECKPOINT)
            with patch.object(inference, 'ROOT', root):
                technical_service = inference.InferenceService()
            calls = []
            def capture(model, args, kwargs, output):
                calls.append((kwargs, output.detach().cpu().clone()))
            handle = technical_service.model.register_forward_hook(capture, with_kwargs=True)
            try:
                with patch('app.service', technical_service):
                    for use_gt in [False, True]:
                        calls.clear()
                        response = self.client.post('/predict', data=self.payload(use_gt))
                        self.assertEqual(response.status_code, 200, response.json)
                        self.assertEqual(len(calls), 3)
                        baseline = calls[0][1]
                        for index, (name, config) in enumerate(inference.VARIANTS.items()):
                            kwargs, score = calls[index]
                            for key, value in config.items():
                                self.assertEqual(kwargs[key], value)
                            self.assertEqual(response.json['score_shapes'][name], [1, 2, 480, 480])
                            item = response.json['results'][name]
                            self.assertEqual(item['iou'] is None, not use_gt)
                            if use_gt:
                                self.assertEqual(item['iou'], foreground_iou(
                                    score.argmax(1)[0].numpy(), binary_mask(self.mask)))
                            for kind in ['mask', 'overlay']:
                                image = Image.open(io.BytesIO(base64.b64decode(item[kind].split(',')[1])))
                                self.assertEqual(image.size, self.query.size)
                            if index:
                                _, sim = technical_service.model.mask_correction(
                                    baseline, kwargs['q_ori'].cpu(), K=config['lcm_k'], method=config['lcm_method'])
                                sim = torch.nn.functional.interpolate(sim, (480, 480), mode='nearest')
                                delta = .6 * (sim - .7)
                                expected = (baseline[:, 1] + delta.sum(1) if index == 1 else
                                            baseline[:, 1] + baseline[:, 1] * delta[:, 0])
                                torch.testing.assert_close(score[:, 0], baseline[:, 0])
                                torch.testing.assert_close(score[:, 1], expected)
            finally:
                handle.remove()

    def test_checkpoint_cache_and_missing_download(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(inference, 'ROOT', Path(folder)):
            with self.assertRaisesRegex(FileNotFoundError, 'Kaggle'):
                inference.extract_checkpoint()
            with zipfile.ZipFile(Path(folder) / 'results.zip', 'w') as archive:
                archive.writestr(inference.DEFAULT_CHECKPOINT, b'current archive')
            target = inference.extract_checkpoint()
            target.write_bytes(b'stale cache')
            self.assertEqual(inference.extract_checkpoint().read_bytes(), b'current archive')
            (Path(folder) / 'results.zip').unlink()
            self.assertEqual(inference.extract_checkpoint(), target)

    def test_checkpoint_upload_and_failed_replacement(self):
        import app as app_module
        state = io.BytesIO()
        torch.save(self.model.state_dict(), state)
        state.seek(0)
        with patch('app.service', None), patch('app.startup_error', 'Missing checkpoint'):
            self.assertEqual(self.client.get('/health').status_code, 503)
            response = self.client.post('/checkpoint', data={'checkpoint': (state, 'my_trained_model.pth')})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertTrue(response.json['ready'])
            self.assertEqual(response.json['checkpoint'], 'my_trained_model.pth')
            self.assertEqual(self.client.get('/health').status_code, 200)
            self.assertIsNone(self.client.get('/health').json['error'])
            loaded = app_module.service
            self.assertEqual(loaded.load_result.missing_keys, [])
            for tensor in ['layer0.0.weight', 'layer3.5.conv3.weight']:
                torch.testing.assert_close(loaded.model.state_dict()[tensor].cpu(), self.model.state_dict()[tensor].cpu())
            for contents, name in [(b'corrupt', 'broken.pth'), (b'{}', 'notebook.ipynb'), (b'', 'empty.pth')]:
                response = self.client.post('/checkpoint', data={'checkpoint': (io.BytesIO(contents), name)})
                self.assertEqual(response.status_code, 400, response.json)
                self.assertIs(app_module.service, loaded)
            incompatible = io.BytesIO()
            torch.save({'other.weight': torch.zeros(1)}, incompatible)
            incompatible.seek(0)
            self.assertEqual(self.client.post('/checkpoint', data={
                'checkpoint': (incompatible, 'wrong_architecture.pth')}).status_code, 400)
            self.assertIs(app_module.service, loaded)

    def test_checkpoint_upload_missing_and_busy(self):
        import app as app_module
        self.assertEqual(self.client.post('/checkpoint', data={}).status_code, 400)
        with app_module.checkpoint_lock:
            self.assertEqual(self.client.post('/checkpoint', data={}).status_code, 409)
        response = self.client.post('/checkpoint', data=b'', content_type='application/octet-stream',
                                    environ_overrides={'CONTENT_LENGTH': str(258 * 1024 * 1024)})
        self.assertEqual(response.status_code, 413)


if __name__ == '__main__':
    unittest.main(verbosity=2)
