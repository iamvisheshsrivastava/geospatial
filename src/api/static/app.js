// ─── Tab navigation ───────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
      b.classList.add('text-slate-600');
    });
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    btn.classList.remove('text-slate-600');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// ─── Health check ─────────────────────────────────────────────────────────────
async function refreshHealth() {
  const pill = document.getElementById('health-pill');
  try {
    const data = await fetch('/health').then(r => r.json());
    const loaded = [data.classifier_loaded, data.anomaly_detector_loaded, data.segmentation_loaded];
    const count  = loaded.filter(Boolean).length;
    if (count === 3) {
      pill.className = 'flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-green-50 text-green-700 border border-green-200';
      pill.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-green-500 inline-block"></span><span>All models ready</span>`;
    } else if (count > 0) {
      pill.className = 'flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200';
      pill.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-amber-500 inline-block"></span><span>${count}/3 models loaded</span>`;
    } else {
      pill.className = 'flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-red-50 text-red-700 border border-red-200';
      pill.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-red-500 inline-block"></span><span>Models not loaded</span>`;
    }
  } catch {
    pill.className = 'flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-red-50 text-red-700 border border-red-200';
    pill.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-red-500 inline-block"></span><span>API unreachable</span>`;
  }
}
refreshHealth();
setInterval(refreshHealth, 30000);

// ─── Progress bar ─────────────────────────────────────────────────────────────
function startProgress(barId, labelId, estimatedMs = 15000) {
  const bar   = document.getElementById(barId);
  const label = document.getElementById(labelId);
  if (!bar) return;
  const wrapper = bar.closest('[id^="prog-"]');
  if (wrapper) wrapper.classList.remove('hidden');

  let pct = 0, secs = 0;
  bar.style.transition = 'none';
  bar.style.width = '0%';
  label.textContent = '0%  0s';

  // Elapsed-seconds ticker
  const elapsedTimer = setInterval(() => {
    secs++;
    label.textContent = `${Math.round(pct)}%  ${secs}s`;
  }, 1000);

  // Phase 1: jump to 30% quickly (request sent, waiting for model)
  const phase1 = setInterval(() => {
    pct = Math.min(pct + 3, 30);
    bar.style.width = pct + '%';
    label.textContent = `${Math.round(pct)}%  ${secs}s`;
    if (pct >= 30) clearInterval(phase1);
  }, 60);

  // Phase 2: crawl to 88% over estimatedMs (inference running)
  const step = 58 / (estimatedMs / 200);
  const phase2 = setInterval(() => {
    pct = Math.min(pct + step, 88);
    bar.style.width = pct + '%';
    label.textContent = `${Math.round(pct)}%  ${secs}s`;
    if (pct >= 88) clearInterval(phase2);
  }, 200);

  return { bar: phase2, elapsed: elapsedTimer };
}

function finishProgress(barId, labelId, timerObj) {
  if (!timerObj) return;
  clearInterval(timerObj.bar   || timerObj);   // handle old callers passing raw id
  clearInterval(timerObj.elapsed);
  const bar   = document.getElementById(barId);
  const label = document.getElementById(labelId);
  if (!bar) return;
  bar.style.transition = 'width 0.3s ease';
  bar.style.width = '100%';
  label.textContent = '100% ✓';
  const wrapper = bar.closest('[id^="prog-"]');
  setTimeout(() => { if (wrapper) wrapper.classList.add('hidden'); bar.style.width = '0%'; }, 1000);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function setupDropzone(dropzoneId, inputId, onFile) {
  const dz    = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  dz.addEventListener('click', () => input.click());
  input.addEventListener('change', () => { if (input.files[0]) onFile(input.files[0]); });
  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('dropzone-hover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dropzone-hover'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dropzone-hover');
    const f = e.dataTransfer.files[0];
    if (f) { input.files = e.dataTransfer.files; onFile(f); }
  });
}

function showImagePreview(imgId, wrapperId, file) {
  const wrap = document.getElementById(wrapperId);
  const img  = document.getElementById(imgId);
  if (file && file.type.startsWith('image/')) {
    img.src = URL.createObjectURL(file);
    wrap.classList.remove('hidden');
  }
}

// Load a sample image from /static/samples/ and return a File object
async function loadSample(filename) {
  const url  = `/static/samples/${filename}`;
  const resp = await fetch(url);
  const blob = await resp.blob();
  return new File([blob], filename, { type: 'image/jpeg' });
}

function showSamplePreview(imgId, wrapperId, filename) {
  const wrap = document.getElementById(wrapperId);
  const img  = document.getElementById(imgId);
  img.src = `/static/samples/${filename}`;
  wrap.classList.remove('hidden');
}

function renderHeatmap(canvasId, matrix) {
  const canvas = document.getElementById(canvasId);
  const rows = matrix.length, cols = matrix[0].length;
  canvas.width  = cols;
  canvas.height = rows;
  const ctx  = canvas.getContext('2d');
  const data = ctx.createImageData(cols, rows);
  const flat = matrix.flat();
  const min  = Math.min(...flat), max = Math.max(...flat);
  flat.forEach((v, i) => {
    const t  = (v - min) / (max - min + 1e-8);
    data.data[i*4]   = Math.round(255 * Math.min(1, t * 2));
    data.data[i*4+1] = Math.round(255 * Math.min(1, (1-t) * 2));
    data.data[i*4+2] = 50;
    data.data[i*4+3] = 255;
  });
  ctx.putImageData(data, 0, 0);
}

// Clear old result and show empty state
function clearResult(resultId, emptyId) {
  document.getElementById(resultId).classList.add('hidden');
  document.getElementById(emptyId).classList.remove('hidden');
}

const CLASS_EMOJI = {
  AnnualCrop:'🌾', Forest:'🌲', HerbaceousVegetation:'🌿', Highway:'🛣️',
  Industrial:'🏭', Pasture:'🐄', PermanentCrop:'🍇', Residential:'🏘️',
  River:'🏞️', SeaLake:'🌊'
};

// ─── TAB 1: CLASSIFY ─────────────────────────────────────────────────────────
let fileClassify = null;

setupDropzone('dropzone-classify', 'file-classify', f => {
  fileClassify = f;
  document.getElementById('fname-classify').textContent = f.name;
  showImagePreview('img-classify', 'preview-classify', f);
});

// Sample buttons
document.querySelectorAll('.sample-classify').forEach(btn => {
  btn.addEventListener('click', async () => {
    const file = await loadSample(btn.dataset.file);
    fileClassify = file;
    document.getElementById('fname-classify').textContent = btn.dataset.label;
    showSamplePreview('img-classify', 'preview-classify', btn.dataset.file);
    document.getElementById('msg-classify').textContent = `Sample loaded: ${btn.dataset.label}`;
  });
});

document.getElementById('btn-classify').addEventListener('click', async () => {
  if (!fileClassify) { document.getElementById('msg-classify').textContent = 'Choose an image first.'; return; }

  // IMPROVEMENT 3: clear old result immediately
  clearResult('result-classify', 'result-classify-empty');
  document.getElementById('result-classify-empty').querySelector('div:last-child').textContent = 'Classifying...';

  const btn = document.getElementById('btn-classify');
  btn.disabled = true; btn.textContent = 'Classifying...';

  // IMPROVEMENT 2: start progress bar
  const timer = startProgress('prog-classify-fill', 'prog-classify-pct', 15000);
  document.getElementById('msg-classify').textContent = 'Sending to model...';

  try {
    const fd = new FormData(); fd.append('file', fileClassify);
    const r  = await fetch('/predict', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Prediction failed');

    finishProgress('prog-classify-fill', 'prog-classify-pct', timer);
    document.getElementById('result-classify-empty').classList.add('hidden');
    document.getElementById('result-classify').classList.remove('hidden');
    document.getElementById('predicted-class').textContent = d.predicted_class;
    document.getElementById('confidence-pct').textContent  = `${Math.round(d.confidence * 100)}%`;
    document.getElementById('class-emoji').textContent     = CLASS_EMOJI[d.predicted_class] || '🛰️';

    const barsEl = document.getElementById('prob-bars');
    barsEl.innerHTML = '';
    Object.entries(d.probabilities).sort((a,b) => b[1]-a[1]).forEach(([cls, prob]) => {
      const pct   = Math.round(prob * 100);
      const isTop = cls === d.predicted_class;
      barsEl.innerHTML += `
        <div class="flex items-center gap-2 text-xs">
          <span class="w-4">${CLASS_EMOJI[cls]||'•'}</span>
          <span class="w-36 truncate ${isTop?'font-semibold text-forest':'text-slate-600'}">${cls}</span>
          <div class="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
            <div class="prob-bar h-full rounded-full ${isTop?'bg-forest':'bg-slate-300'}" style="width:${pct}%"></div>
          </div>
          <span class="w-8 text-right ${isTop?'font-bold text-forest':'text-slate-500'}">${pct}%</span>
        </div>`;
    });
    document.getElementById('msg-classify').textContent = '✓ Classification complete';
  } catch(e) {
    finishProgress('prog-classify-fill', 'prog-classify-pct', timer);
    document.getElementById('result-classify-empty').querySelector('div:last-child').textContent = 'Upload an image and run classification to see results';
    document.getElementById('msg-classify').textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false; btn.textContent = 'Run Classification';
  }
});

// ─── TAB 2: ANOMALY ──────────────────────────────────────────────────────────
let fileAnomaly = null;

setupDropzone('dropzone-anomaly', 'file-anomaly', f => {
  fileAnomaly = f;
  showImagePreview('img-anomaly', 'preview-anomaly', f);
});

document.querySelectorAll('.sample-anomaly').forEach(btn => {
  btn.addEventListener('click', async () => {
    fileAnomaly = await loadSample(btn.dataset.file);
    showSamplePreview('img-anomaly', 'preview-anomaly', btn.dataset.file);
    document.getElementById('msg-anomaly').textContent = `Sample loaded: ${btn.dataset.label}`;
  });
});

document.getElementById('btn-anomaly').addEventListener('click', async () => {
  if (!fileAnomaly) { document.getElementById('msg-anomaly').textContent = 'Choose an image first.'; return; }
  clearResult('result-anomaly', 'result-anomaly-empty');
  const btn = document.getElementById('btn-anomaly');
  btn.disabled = true; btn.textContent = 'Analysing...';
  const timer = startProgress('prog-anomaly-fill', 'prog-anomaly-pct', 12000);
  try {
    const fd = new FormData(); fd.append('file', fileAnomaly);
    const r  = await fetch('/anomaly', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Anomaly detection failed');

    finishProgress('prog-anomaly-fill', 'prog-anomaly-pct', timer);
    document.getElementById('result-anomaly-empty').classList.add('hidden');
    document.getElementById('result-anomaly').classList.remove('hidden');
    const badge = document.getElementById('anomaly-badge');
    if (d.is_anomaly) {
      badge.className = 'flex items-center gap-3 mb-4 p-3 rounded-lg bg-red-50';
      document.getElementById('anomaly-icon').textContent  = '🚨';
      document.getElementById('anomaly-label').textContent = 'ANOMALY DETECTED';
      document.getElementById('anomaly-label').className   = 'text-xl font-bold text-red-600';
      document.getElementById('anomaly-score').className   = 'text-2xl font-bold text-red-600';
    } else {
      badge.className = 'flex items-center gap-3 mb-4 p-3 rounded-lg bg-green-50';
      document.getElementById('anomaly-icon').textContent  = '✅';
      document.getElementById('anomaly-label').textContent = 'Normal — No anomaly';
      document.getElementById('anomaly-label').className   = 'text-xl font-bold text-green-700';
      document.getElementById('anomaly-score').className   = 'text-2xl font-bold text-green-700';
    }
    document.getElementById('anomaly-score').textContent = d.anomaly_score.toFixed(4);
    renderHeatmap('heatmap-anomaly', d.heatmap);
    document.getElementById('msg-anomaly').textContent = '✓ Analysis complete';
  } catch(e) {
    finishProgress('prog-anomaly-fill', 'prog-anomaly-pct', timer);
    document.getElementById('msg-anomaly').textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; btn.textContent = 'Detect Anomaly'; }
});

// ─── TAB 3: CHANGE DETECTION ─────────────────────────────────────────────────
let fileBefore = null, fileAfter = null;

setupDropzone('dropzone-before', 'file-before', f => {
  fileBefore = f;
  const img = document.getElementById('img-before');
  if (f.type.startsWith('image/')) { img.src = URL.createObjectURL(f); img.classList.remove('hidden'); }
});
setupDropzone('dropzone-after', 'file-after', f => {
  fileAfter = f;
  const img = document.getElementById('img-after');
  if (f.type.startsWith('image/')) { img.src = URL.createObjectURL(f); img.classList.remove('hidden'); }
});

document.getElementById('sample-change').addEventListener('click', async () => {
  fileBefore = await loadSample('forest.jpg');
  fileAfter  = await loadSample('change_after.jpg');
  const ib = document.getElementById('img-before');
  ib.src = '/static/samples/forest.jpg'; ib.classList.remove('hidden');
  const ia = document.getElementById('img-after');
  ia.src = '/static/samples/change_after.jpg'; ia.classList.remove('hidden');
  document.getElementById('msg-change').textContent = 'Sample loaded: forest before/after deforestation';
});

document.getElementById('btn-change').addEventListener('click', async () => {
  if (!fileBefore || !fileAfter) { document.getElementById('msg-change').textContent = 'Upload both images first.'; return; }
  clearResult('result-change', 'result-change-empty');
  const btn = document.getElementById('btn-change');
  btn.disabled = true; btn.textContent = 'Analysing...';
  const timer = startProgress('prog-change-fill', 'prog-change-pct', 15000);
  try {
    const fd = new FormData(); fd.append('before', fileBefore); fd.append('after', fileAfter);
    const r  = await fetch('/change-detect', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Change detection failed');

    finishProgress('prog-change-fill', 'prog-change-pct', timer);
    document.getElementById('result-change-empty').classList.add('hidden');
    document.getElementById('result-change').classList.remove('hidden');
    const badge = document.getElementById('change-badge');
    if (d.is_changed) {
      badge.className = 'flex items-center gap-3 mb-4 p-3 rounded-lg bg-purple-50';
      document.getElementById('change-icon').textContent  = '🔄';
      document.getElementById('change-label').textContent = 'CHANGE DETECTED';
      document.getElementById('change-label').className   = 'text-xl font-bold text-purple-700';
      document.getElementById('change-score').className   = 'text-2xl font-bold text-purple-700';
    } else {
      badge.className = 'flex items-center gap-3 mb-4 p-3 rounded-lg bg-green-50';
      document.getElementById('change-icon').textContent  = '✅';
      document.getElementById('change-label').textContent = 'No significant change';
      document.getElementById('change-label').className   = 'text-xl font-bold text-green-700';
      document.getElementById('change-score').className   = 'text-2xl font-bold text-green-700';
    }
    document.getElementById('change-score').textContent = d.change_score.toFixed(4);
    renderHeatmap('heatmap-change', d.change_map);
    document.getElementById('msg-change').textContent = '✓ Analysis complete';
  } catch(e) {
    finishProgress('prog-change-fill', 'prog-change-pct', timer);
    document.getElementById('msg-change').textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; btn.textContent = 'Detect Changes'; }
});

// ─── TAB 4: SEGMENTATION ─────────────────────────────────────────────────────
let fileSegment = null;
const confSlider = document.getElementById('conf-thresh');
confSlider.addEventListener('input', () => {
  document.getElementById('conf-val').textContent = parseFloat(confSlider.value).toFixed(2);
});

setupDropzone('dropzone-segment', 'file-segment', f => {
  fileSegment = f;
  showImagePreview('img-segment', 'preview-segment', f);
});

document.getElementById('sample-segment').addEventListener('click', async () => {
  fileSegment = await loadSample('tree_crowns.jpg');
  showSamplePreview('img-segment', 'preview-segment', 'tree_crowns.jpg');
  document.getElementById('msg-segment').textContent = 'Sample loaded: tree crowns aerial view';
});

document.getElementById('btn-segment').addEventListener('click', async () => {
  if (!fileSegment) { document.getElementById('msg-segment').textContent = 'Choose an image first.'; return; }
  clearResult('result-segment', 'result-segment-empty');
  const btn = document.getElementById('btn-segment');
  btn.disabled = true; btn.textContent = 'Detecting...';
  const timer = startProgress('prog-segment-fill', 'prog-segment-pct', 45000);
  try {
    const fd = new FormData(); fd.append('file', fileSegment);
    const thresh = parseFloat(confSlider.value);
    const r  = await fetch(`/segment?confidence_threshold=${thresh}`, { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Segmentation failed');

    finishProgress('prog-segment-fill', 'prog-segment-pct', timer);
    document.getElementById('result-segment-empty').classList.add('hidden');
    document.getElementById('result-segment').classList.remove('hidden');
    document.getElementById('tree-count').textContent = d.num_trees;

    const list = document.getElementById('tree-detections');
    list.innerHTML = '';
    if (d.detections.length === 0) {
      list.innerHTML = '<div class="text-sm text-slate-400 text-center py-3">No trees detected at this confidence threshold</div>';
    } else {
      d.detections.forEach((det, i) => {
        const conf = Math.round(det.score * 100);
        list.innerHTML += `
          <div class="flex items-center gap-2 text-xs bg-slate-50 rounded-lg px-3 py-2">
            <span class="text-green-600 font-bold w-16">Tree #${i+1}</span>
            <div class="flex-1 bg-slate-200 rounded-full h-1.5"><div class="bg-green-500 h-full rounded-full" style="width:${conf}%"></div></div>
            <span class="text-slate-500 w-10 text-right">${conf}%</span>
            <span class="text-slate-400 w-24 text-right">${det.mask_area_px.toLocaleString()} px²</span>
          </div>`;
      });
    }
    document.getElementById('msg-segment').textContent = `✓ Found ${d.num_trees} tree crown${d.num_trees!==1?'s':''}`;
  } catch(e) {
    finishProgress('prog-segment-fill', 'prog-segment-pct', timer);
    document.getElementById('msg-segment').textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; btn.textContent = 'Detect Tree Crowns'; }
});

// ─── TAB 5: LIDAR ────────────────────────────────────────────────────────────
let fileLidar = null;
setupDropzone('dropzone-lidar', 'file-lidar', f => {
  fileLidar = f;
  const prev = document.getElementById('preview-lidar');
  prev.textContent = `📁 ${f.name}  (${(f.size/1024/1024).toFixed(1)} MB)`;
  prev.classList.remove('hidden');
});

// Sample button — load the built-in synthetic forest LAS file
document.getElementById('sample-lidar').addEventListener('click', async () => {
  const btn = document.getElementById('sample-lidar');
  btn.textContent = 'Loading...'; btn.disabled = true;
  try {
    const resp = await fetch('/static/samples/sample.las');
    const blob = await resp.blob();
    fileLidar = new File([blob], 'sample.las', { type: 'application/octet-stream' });
    const prev = document.getElementById('preview-lidar');
    prev.textContent = `📁 sample.las  (${(fileLidar.size/1024/1024).toFixed(1)} MB)`;
    prev.classList.remove('hidden');
    document.getElementById('msg-lidar').textContent = 'Sample loaded: synthetic forest LiDAR scan (50k points)';
  } catch(e) {
    document.getElementById('msg-lidar').textContent = `Failed to load sample: ${e.message}`;
  } finally {
    btn.textContent = '📡 Forest LiDAR scan'; btn.disabled = false;
  }
});

document.getElementById('btn-lidar').addEventListener('click', async () => {
  if (!fileLidar) { document.getElementById('msg-lidar').textContent = 'Choose a LAS/LAZ file first.'; return; }
  clearResult('result-lidar', 'result-lidar-empty');
  const btn = document.getElementById('btn-lidar');
  btn.disabled = true; btn.textContent = 'Processing...';
  const timer = startProgress('prog-lidar-fill', 'prog-lidar-pct', 30000);
  document.getElementById('msg-lidar').textContent = 'Analysing point cloud (this may take 30s)...';
  try {
    const fd = new FormData(); fd.append('file', fileLidar);
    const r  = await fetch('/pointcloud', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Point cloud processing failed');

    finishProgress('prog-lidar-fill', 'prog-lidar-pct', timer);
    document.getElementById('result-lidar-empty').classList.add('hidden');
    document.getElementById('result-lidar').classList.remove('hidden');

    const s = d.stats;
    const statItems = [
      {icon:'🌲', label:'Trees detected',    value: d.num_trees_detected},
      {icon:'📏', label:'Mean canopy height', value:`${s.mean_canopy_height_m.toFixed(1)} m`},
      {icon:'⬆️', label:'Max canopy height',  value:`${s.max_canopy_height_m.toFixed(1)} m`},
      {icon:'🌿', label:'Canopy cover',       value:`${(s.canopy_cover_fraction*100).toFixed(1)}%`},
      {icon:'🔢', label:'Stem density',       value:`${s.stem_density_per_ha.toFixed(0)}/ha`},
      {icon:'📍', label:'Total points',       value: s.num_points.toLocaleString()},
    ];
    document.getElementById('lidar-stats').innerHTML = statItems.map(item => `
      <div class="bg-orange-50 rounded-lg p-3 flex items-start gap-2">
        <span class="text-lg">${item.icon}</span>
        <div><div class="text-xs text-slate-500">${item.label}</div><div class="font-bold text-slate-800">${item.value}</div></div>
      </div>`).join('');

    const treeList = document.getElementById('lidar-trees');
    treeList.innerHTML = '';
    d.trees.slice(0, 20).forEach(tree => {
      treeList.innerHTML += `
        <div class="flex items-center gap-2 text-xs bg-slate-50 rounded-lg px-3 py-2">
          <span class="text-orange-500 font-bold w-14">Tree #${tree.tree_id}</span>
          <span class="text-slate-600 w-20">${tree.height_m.toFixed(1)} m tall</span>
          <span class="text-slate-400">radius ${tree.crown_radius_m.toFixed(1)} m</span>
          <span class="text-slate-400 ml-auto">${tree.num_points} pts</span>
        </div>`;
    });
    if (d.trees.length > 20) treeList.innerHTML += `<div class="text-xs text-slate-400 text-center py-1">...and ${d.trees.length-20} more trees</div>`;
    document.getElementById('msg-lidar').textContent = '✓ Analysis complete';
  } catch(e) {
    finishProgress('prog-lidar-fill', 'prog-lidar-pct', timer);
    document.getElementById('msg-lidar').textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; btn.textContent = 'Analyse Point Cloud'; }
});

// ─── TAB 6: XAI / GRADCAM ────────────────────────────────────────────────────
let fileExplain = null;

setupDropzone('dropzone-explain', 'file-explain', f => {
  fileExplain = f;
  showImagePreview('img-explain', 'preview-explain', f);
});

document.querySelectorAll('.sample-explain').forEach(btn => {
  btn.addEventListener('click', async () => {
    fileExplain = await loadSample(btn.dataset.file);
    showSamplePreview('img-explain', 'preview-explain', btn.dataset.file);
    document.getElementById('msg-explain').textContent = `Sample loaded: ${btn.dataset.label}`;
  });
});

document.getElementById('btn-explain').addEventListener('click', async () => {
  if (!fileExplain) { document.getElementById('msg-explain').textContent = 'Choose an image first.'; return; }
  clearResult('result-explain', 'result-explain-empty');
  const btn = document.getElementById('btn-explain');
  btn.disabled = true; btn.textContent = 'Running GradCAM...';
  const timer = startProgress('prog-explain-fill', 'prog-explain-pct', 20000);
  try {
    const fd = new FormData(); fd.append('file', fileExplain);
    const r = await fetch('/explain', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'GradCAM failed');

    finishProgress('prog-explain-fill', 'prog-explain-pct', timer);
    document.getElementById('result-explain-empty').classList.add('hidden');
    document.getElementById('result-explain').classList.remove('hidden');
    document.getElementById('explain-emoji').textContent  = CLASS_EMOJI[d.predicted_class] || '🛰️';
    document.getElementById('explain-class').textContent  = d.predicted_class;
    document.getElementById('explain-conf').textContent   = `${Math.round(d.confidence * 100)}% confidence`;
    document.getElementById('gradcam-img').src = `data:image/png;base64,${d.gradcam_b64}`;
    document.getElementById('msg-explain').textContent = '✓ GradCAM complete';
  } catch(e) {
    finishProgress('prog-explain-fill', 'prog-explain-pct', timer);
    document.getElementById('msg-explain').textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; btn.textContent = 'Run GradCAM Explanation'; }
});

// ─── TAB 7: POVERTY PROXY ────────────────────────────────────────────────────
let filePoverty = null;

setupDropzone('dropzone-poverty', 'file-poverty', f => {
  filePoverty = f;
  showImagePreview('img-poverty', 'preview-poverty', f);
});

document.querySelectorAll('.sample-poverty').forEach(btn => {
  btn.addEventListener('click', async () => {
    filePoverty = await loadSample(btn.dataset.file);
    showSamplePreview('img-poverty', 'preview-poverty', btn.dataset.file);
    document.getElementById('msg-poverty').textContent = `Sample loaded: ${btn.dataset.label}`;
  });
});

document.getElementById('btn-poverty').addEventListener('click', async () => {
  if (!filePoverty) { document.getElementById('msg-poverty').textContent = 'Choose an image first.'; return; }
  clearResult('result-poverty', 'result-poverty-empty');
  const btn = document.getElementById('btn-poverty');
  btn.disabled = true; btn.textContent = 'Estimating...';
  const timer = startProgress('prog-poverty-fill', 'prog-poverty-pct', 15000);
  try {
    const fd = new FormData(); fd.append('file', filePoverty);
    const r = await fetch('/poverty-proxy', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Poverty proxy failed');

    finishProgress('prog-poverty-fill', 'prog-poverty-pct', timer);
    document.getElementById('result-poverty-empty').classList.add('hidden');
    document.getElementById('result-poverty').classList.remove('hidden');

    // Badge styling
    const badge = document.getElementById('poverty-badge');
    badge.style.borderColor = d.wealth_color;
    badge.style.backgroundColor = d.wealth_color + '15';
    document.getElementById('poverty-label').textContent = d.wealth_label;
    document.getElementById('poverty-label').style.color = d.wealth_color;
    document.getElementById('poverty-interpretation').textContent = d.interpretation;
    document.getElementById('poverty-score').textContent = d.wealth_index.toFixed(3);
    document.getElementById('poverty-score').style.color = d.wealth_color;
    document.getElementById('poverty-score-pct').textContent = `${Math.round(d.wealth_index * 100)}%`;

    // Progress bar
    const bar = document.getElementById('poverty-bar');
    bar.style.backgroundColor = d.wealth_color;
    setTimeout(() => { bar.style.width = `${Math.round(d.wealth_index * 100)}%`; }, 100);

    // Contributors
    const contribs = document.getElementById('poverty-contributors');
    contribs.innerHTML = '';
    d.top_contributors.forEach(c => {
      const pct = Math.round(c.probability * 100);
      contribs.innerHTML += `
        <div class="flex items-center gap-2 text-xs">
          <span class="w-36 truncate font-medium text-slate-700">${CLASS_EMOJI[c.class] || '•'} ${c.class}</span>
          <div class="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
            <div class="h-full rounded-full bg-amber-400" style="width:${pct}%"></div>
          </div>
          <span class="text-slate-500 w-10 text-right">${pct}%</span>
          <span class="text-slate-400 w-16 text-right">×${c.weight.toFixed(2)} wt</span>
        </div>`;
    });
    document.getElementById('poverty-methodology').textContent = d.methodology;
    document.getElementById('msg-poverty').textContent = '✓ Wealth index estimated';
  } catch(e) {
    finishProgress('prog-poverty-fill', 'prog-poverty-pct', timer);
    document.getElementById('msg-poverty').textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; btn.textContent = 'Estimate Wealth Index'; }
});

// ─── TAB 8: SPECTRAL ANALYSIS ────────────────────────────────────────────────
let fileSpectral = null;

setupDropzone('dropzone-spectral', 'file-spectral', f => {
  fileSpectral = f;
  showImagePreview('img-spectral', 'preview-spectral', f);
});

document.querySelectorAll('.sample-spectral').forEach(btn => {
  btn.addEventListener('click', async () => {
    fileSpectral = await loadSample(btn.dataset.file);
    showSamplePreview('img-spectral', 'preview-spectral', btn.dataset.file);
    document.getElementById('msg-spectral').textContent = `Sample loaded: ${btn.dataset.label}`;
  });
});

document.getElementById('btn-spectral').addEventListener('click', async () => {
  if (!fileSpectral) { document.getElementById('msg-spectral').textContent = 'Choose an image first.'; return; }
  clearResult('result-spectral', 'result-spectral-empty');
  const btn = document.getElementById('btn-spectral');
  btn.disabled = true; btn.textContent = 'Analysing...';
  const timer = startProgress('prog-spectral-fill', 'prog-spectral-pct', 8000);
  try {
    const fd = new FormData(); fd.append('file', fileSpectral);
    const r = await fetch('/spectral', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Spectral analysis failed');

    finishProgress('prog-spectral-fill', 'prog-spectral-pct', timer);
    document.getElementById('result-spectral-empty').classList.add('hidden');
    document.getElementById('result-spectral').classList.remove('hidden');

    document.getElementById('spectral-veg').src   = `data:image/png;base64,${d.vegetation_b64}`;
    document.getElementById('spectral-water').src  = `data:image/png;base64,${d.water_b64}`;
    document.getElementById('spectral-urban').src  = `data:image/png;base64,${d.urban_b64}`;

    const fmt = v => (v >= 0 ? '+' : '') + v.toFixed(3);
    document.getElementById('spectral-veg-val').textContent   = `mean VARI: ${fmt(d.vegetation_mean)}`;
    document.getElementById('spectral-water-val').textContent = `mean ExWI: ${fmt(d.water_mean)}`;
    document.getElementById('spectral-urban-val').textContent = `mean ExUI: ${fmt(d.urban_mean)}`;

    document.getElementById('spectral-interpretation').textContent = d.interpretation;
    document.getElementById('msg-spectral').textContent = '✓ Spectral analysis complete';
  } catch(e) {
    finishProgress('prog-spectral-fill', 'prog-spectral-pct', timer);
    document.getElementById('msg-spectral').textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; btn.textContent = 'Analyse Spectral Indices'; }
});
