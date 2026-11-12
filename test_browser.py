"""Optional UI smoke test: start app.py, then pip install playwright and run this.
Uses installed Chrome, synthetic fixtures and the real localhost inference API.
"""
import io
import json

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def fixture(image, name):
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    return {'name': name, 'mimeType': 'image/png', 'buffer': buffer.getvalue()}


def main():
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (160, 220, 3), dtype=np.uint8))
    array = np.zeros((160, 220), dtype=np.uint8)
    array[30:130, 50:170] = 255
    mask = Image.fromarray(array)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, accept_downloads=True)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:5000/')
        health = page.request.get('http://127.0.0.1:5000/health').json()
        assert health['ready'] and health['checkpoint'] == 'last_loec_model.pth', health
        page.wait_for_function("document.getElementById('model-status').textContent === 'Sẵn sàng'")
        assert page.locator('#predict-button').is_disabled()
        assert page.locator('#results').is_hidden()
        page.evaluate('document.fonts.ready')
        assert page.evaluate("document.fonts.check('500 16px Inter')")
        assert page.locator('.app-header').bounding_box()['height'] == 72
        for width, height in [(1920, 1080), (1366, 768)]:
            page.set_viewport_size({'width': width, 'height': height})
            box = page.locator('#predict-button').bounding_box()
            assert box['y'] + box['height'] < height
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path='browser-empty.png', full_page=True)
        for key, img in [('support_image', image), ('support_mask', mask), ('query_image', image)]:
            page.locator('#' + key).set_input_files(fixture(img, key + '.png'))
        page.locator('#preview-query_image').wait_for(state='visible')
        assert page.locator('#predict-button').is_enabled()
        with page.expect_response(lambda response: response.url.endswith('/predict')) as prediction:
            page.locator('#predict-button').click()
            assert page.locator('#predict-button').is_disabled()
            assert page.locator('#spinner').is_visible()
            assert page.locator('#button-label').inner_text() == 'Đang xử lý...'
            assert page.locator('#reset-button').is_disabled()
        result = prediction.value.json()
        assert result['score_shapes'] == {'no_lcm': [1, 2, 480, 480], 'with_lcm': [1, 2, 480, 480]}
        page.locator('#results').wait_for(state='visible', timeout=60000)
        assert page.locator('.prediction-mask').count() == 2
        assert page.locator('.prediction-overlay').count() == 2
        cards = page.locator('.comparison-card')
        left, right = cards.nth(0).bounding_box(), cards.nth(1).bounding_box()
        assert left['y'] == right['y'] and left['height'] == right['height']
        assert page.locator('#iou-panel').is_hidden()
        for key in ['no_lcm', 'with_lcm']:
            for kind in ['mask', 'overlay']:
                locator = page.locator('#' + kind + '-' + key)
                locator.evaluate('(image) => image.decode()')
                assert locator.evaluate('(image) => [image.naturalWidth, image.naturalHeight]') == [220, 160]
        for index in range(4):
            with page.expect_download() as download:
                page.locator('#results a[download]').nth(index).click()
            with Image.open(download.value.path()) as downloaded:
                assert downloaded.size == (220, 160)
                assert downloaded.format == 'PNG'
        page.locator('#query_gt').set_input_files(fixture(mask, 'gt.png'))
        with page.expect_response(lambda response: response.url.endswith('/predict')) as prediction:
            page.locator('#predict-button').click()
        result = prediction.value.json()
        assert result['iou_resolution'] == [480, 480]
        page.locator('#results').wait_for(state='visible', timeout=60000)
        assert page.locator('#iou-body tr').count() == 2
        assert page.locator('#iou-panel').is_visible()
        for index, key in enumerate(['no_lcm', 'with_lcm']):
            assert 0 <= result['results'][key]['iou'] <= 1
            assert page.locator('#iou-body tr').nth(index).locator('td').nth(1).inner_text() == f"{result['results'][key]['iou'] * 100:.2f}%"
        page.screenshot(path='browser-desktop.png', full_page=True)
        for width in [1920, 1024, 820, 768, 390, 360, 320]:
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), width
            assert page.locator('.comparison-card').nth(0).bounding_box()['height'] == page.locator('.comparison-card').nth(1).bounding_box()['height'], width
            assert page.evaluate('''() => [...document.querySelectorAll('main *, header *')].filter(el => {
                const css = getComputedStyle(el);
                return el.getClientRects().length && [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()) && parseFloat(css.fontSize) < 13;
            }).length === 0''')
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path='browser-mobile.png', full_page=True)
        page.locator('#reset-button').click()
        assert page.locator('#results').is_hidden()
        assert page.locator('#predict-button').is_disabled()
        assert page.locator('#support_image').input_value() == ''
        page.locator('#query_image').set_input_files({'name': 'bad.png', 'mimeType': 'image/png', 'buffer': b'broken image'})
        page.locator('#error-message').wait_for(state='visible')
        assert page.locator('#predict-button').is_disabled()
        # Real backend validation error, using a valid but empty support mask.
        for key, img in [('support_image', image), ('support_mask', Image.new('L', (80, 80))), ('query_image', image)]:
            page.locator('#' + key).set_input_files(fixture(img, key + '.png'))
        page.locator('#predict-button').click()
        page.locator('#error-message').wait_for(state='visible')
        assert page.locator('#results').is_hidden()
        page.wait_for_function("!document.getElementById('predict-button').disabled")
        assert page.locator('#spinner').is_hidden()
        assert page.locator('#reset-button').is_enabled()
        offline_page = browser.new_page()
        offline_page.route('**/health', lambda route: route.abort())
        offline_page.goto('http://127.0.0.1:5000/')
        offline_page.locator('#model-status.failed').wait_for()
        assert offline_page.locator('#model-status').inner_text() == 'Chưa sẵn sàng'
        assert offline_page.locator('#predict-button').is_disabled()
        assert offline_page.locator('#error-message').is_visible()
        offline_page.close()
        assert not errors, errors
        print(json.dumps({'browser': 'Chrome headless', 'checkpoint': health['checkpoint'],
                          'score_shapes': result['score_shapes'], 'upload': True,
                          'prediction_without_gt': True, 'prediction_with_gt': True,
                          'downloaded_pngs': 4, 'reset': True, 'corrupt_preview': True,
                          'responsive_widths': [1920, 1366, 1024, 820, 768, 390, 360, 320],
                          'loading_state': True, 'backend_validation_error': True,
                          'javascript_errors': errors}))
        browser.close()


if __name__ == '__main__':
    main()
