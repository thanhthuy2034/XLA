'use strict';
const form = document.getElementById('predict-form');
const button = document.getElementById('predict-button');
const errorBox = document.getElementById('error-message');
const statusText = document.getElementById('form-status');
const requiredKeys = ['support_image', 'support_mask', 'query_image'];
const previewURLs = new Map();
let ready = false;
let busy = false;

function showError(message) { errorBox.textContent = message; errorBox.hidden = !message; }
function updateButton() { button.disabled = busy || !ready || !requiredKeys.every(key => document.getElementById(key).files.length); }
function invalidateResults() { document.getElementById('results').hidden = true; }
function chooseFile(input) {
  invalidateResults();
  showError('');
  const previous = previewURLs.get(input.id);
  if (previous) { URL.revokeObjectURL(previous); previewURLs.delete(input.id); }
  const preview = document.getElementById(`preview-${input.id}`);
  if (preview) { preview.hidden = true; preview.removeAttribute('src'); preview.nextElementSibling.hidden = false; }
  let file = input.files[0];
  if (file && (!/\.(png|jpe?g)$/i.test(file.name) || file.size > 10 * 1024 * 1024)) {
    showError('Chỉ hỗ trợ JPG, JPEG, PNG; mỗi tệp tối đa 10 MB.');
    input.value = ''; file = null;
  }
  document.getElementById(`name-${input.id}`).textContent = file ? file.name : 'Chưa chọn ảnh';
  if (preview && file) {
    const url = URL.createObjectURL(file); previewURLs.set(input.id, url);
    preview.onload = () => { preview.hidden = false; preview.nextElementSibling.hidden = true; };
    preview.onerror = () => { input.value = ''; preview.hidden = true; preview.nextElementSibling.hidden = false; document.getElementById(`name-${input.id}`).textContent = 'Ảnh không hợp lệ'; showError('Không đọc được ảnh đã chọn. Vui lòng chọn ảnh hợp lệ.'); updateButton(); };
    preview.src = url;
  }
  statusText.textContent = requiredKeys.every(key => document.getElementById(key).files.length) ? 'Dữ liệu đã đủ. Sẵn sàng phân đoạn.' : 'Tải đủ ba tệp để bắt đầu.';
  updateButton();
}

document.querySelectorAll('input[type=file]').forEach(input => input.addEventListener('change', () => chooseFile(input)));
document.querySelectorAll('.drop-zone').forEach(zone => {
  ['dragenter', 'dragover'].forEach(type => zone.addEventListener(type, event => { event.preventDefault(); if (!busy) zone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(type => zone.addEventListener(type, event => { event.preventDefault(); zone.classList.remove('dragging'); }));
  zone.addEventListener('drop', event => {
    if (busy) return;
    if (event.dataTransfer.files.length !== 1) { showError('Mỗi ô chỉ nhận một tệp ảnh.'); return; }
    const input = document.getElementById(zone.dataset.input);
    input.files = event.dataTransfer.files; chooseFile(input);
  });
});

async function health() {
  const badge = document.getElementById('model-status');
  try {
    const response = await fetch('/health'); const data = await response.json();
    ready = data.ready === true;
    badge.textContent = ready ? 'Sẵn sàng' : 'Chưa sẵn sàng';
    badge.className = `status-pill ${ready ? 'ready' : 'failed'}`;
    if (!ready) showError(data.error || 'Mô hình chưa sẵn sàng.');
  } catch (error) { ready = false; badge.textContent = 'Chưa sẵn sàng'; badge.className = 'status-pill failed'; showError('Không kết nối được backend. Kiểm tra Flask và tải lại trang.'); }
  updateButton();
}

function render(data) {
  const iouBody = document.getElementById('iou-body'); iouBody.replaceChildren();
  const hasGT = data.results.no_lcm.iou !== null;
  [['no_lcm', 'Không sử dụng LCM'], ['with_lcm', 'Có sử dụng LCM']].forEach(([key, title]) => {
    ['mask', 'overlay'].forEach(kind => {
      document.getElementById(`${kind}-${key}`).src = data.results[key][kind];
      document.getElementById(`download-${kind}-${key}`).href = data.results[key][kind];
    });
    if (hasGT) { const row = document.createElement('tr'); [title, `${(data.results[key].iou * 100).toFixed(2)}%`].forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }); iouBody.append(row); }
  });
  document.getElementById('iou-panel').hidden = !hasGT;
  document.getElementById('run-info').textContent = `${data.elapsed_seconds.toFixed(2)} giây · ${data.device.toUpperCase()} · 2 lượt suy luận`;
  const section = document.getElementById('results'); section.hidden = false;
  section.scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block:'start'});
}

form.addEventListener('submit', async event => {
  event.preventDefault(); if (busy || !ready || !form.reportValidity()) return;
  const body = new FormData(form); busy = true; updateButton(); showError(''); invalidateResults();
  form.querySelectorAll('input,button[type=reset]').forEach(el => el.disabled = true);
  document.getElementById('spinner').hidden = false;
  document.getElementById('button-label').textContent = 'Đang xử lý...';
  document.getElementById('button-arrow').hidden = true;
  form.setAttribute('aria-busy', 'true');
  statusText.textContent = 'Đang chạy hai lượt suy luận thực tế. Vui lòng chờ…';
  try {
    const response = await fetch('/predict', {method:'POST', body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Yêu cầu dự đoán thất bại.');
    render(data); statusText.textContent = 'Đã hoàn tất hai lượt suy luận.';
  } catch (error) { showError(error.message || 'Không thể kết nối backend.'); statusText.textContent = 'Phân đoạn chưa thành công.'; }
  finally {
    busy = false; form.querySelectorAll('input,button[type=reset]').forEach(el => el.disabled = false);
    document.getElementById('spinner').hidden = true;
    document.getElementById('button-label').textContent = 'Bắt đầu phân đoạn';
    document.getElementById('button-arrow').hidden = false;
    form.setAttribute('aria-busy', 'false'); updateButton();
  }
});
form.addEventListener('reset', () => {
  button.disabled = true;
  previewURLs.forEach(url => URL.revokeObjectURL(url)); previewURLs.clear();
  document.querySelectorAll('.preview').forEach(img => { img.hidden = true; img.removeAttribute('src'); img.nextElementSibling.hidden = false; });
  document.querySelectorAll('.file-name').forEach(el => el.textContent = 'Chưa chọn ảnh');
  document.getElementById('name-query_gt').textContent = 'Chưa chọn ảnh';
  document.querySelectorAll('#results img').forEach(img => img.removeAttribute('src'));
  document.querySelectorAll('#results a[download]').forEach(link => link.removeAttribute('href'));
  invalidateResults(); showError(''); statusText.textContent = 'Tải đủ ba tệp để bắt đầu.';
  setTimeout(updateButton, 0);
});
health();
