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
  const text = document.getElementById('health-text');
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

function renderHeatmap(canvasId, matrix) {
  const canvas = document.getElementById(canvasId);
  const rows = matrix.length, cols = matrix[0].length;
  canvas.width  = cols;
  canvas.height = rows;
  const ctx  = canvas.getContext('2d');
  const data = ctx.createImageData(cols, rows);
  // Flatten and find range
  const flat = matrix.flat();
  const min  = Math.min(...flat), max = Math.max(...flat);
  flat.forEach((v, i) => {
    const t  = (v - min) / (max - min + 1e-8);
    const r  = Math.round(255 * Math.min(1, t * 2));
    const g  = Math.round(255 * Math.min(1, (1 - t) * 2));
    data.data[i*4]   = r;
    data.data[i*4+1] = g;
    data.data[i*4+2] = 50;
    data.data[i*4+3] = 255;
  });
  ctx.putImageData(data, 0, 0);
}

const CLASS_EMOJI = {
  AnnualCrop: '🌾', Forest: '🌲', HerbaceousVegetation: '🌿', Highway: '🛣️',
  Industrial: '🏭', Pasture: '🐄', PermanentCrop: '🍇', Residential: '🏘️',
  River: '🏞️', SeaLake: '🌊'
};

// ─── TAB 1: CLASSIFY ─────────────────────────────────────────────────────────
let fileClassify = null;
setupDropzone('dropzone-classify', 'file-classify', f => {
  fileClassify = f;
  document.getElementById('fname-classify').textContent = f.name;
  showImagePreview('img-classify', 'preview-classify', f);
});

document.getElementById('btn-classify').addEventListener('click', async () => {
  if (!fileClassify) { document.getElementById('msg-classify').textContent = 'Choose an image first.'; return; }
  const btn = document.getElementById('btn-classify');
  btn.disabled = true; btn.textContent = 'Classifying...';
  document.getElementById('msg-classify').textContent = 'Sending to model...';
  try {
    const fd = new FormData(); fd.append('file', fileClassify);
    const r  = await fetch('/predict', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Prediction failed');

    document.getElementById('result-classify-empty').classList.add('hidden');
    document.getElementById('result-classify').classList.remove('hidden');
    document.getElementById('predicted-class').textContent = d.predicted_class;
    document.getElementById('confidence-pct').textContent  = `${Math.round(d.confidence * 100)}%`;
    document.getElementById('class-emoji').textContent     = CLASS_EMOJI[d.predicted_class] || '🛰️';

    const barsEl = document.getElementById('prob-bars');
    barsEl.innerHTML = '';
    Object.entries(d.probabilities).sort((a,b) => b[1]-a[1]).forEach(([cls, prob]) => {
      const pct = Math.round(prob * 100);
      const isTop = cls === d.predicted_class;
      barsEl.innerHTML += `
        <div class="flex items-center gap-2 text-xs">
          <span class="w-4">${CLASS_EMOJI[cls]||'•'}</span>
          <span class="w-36 truncate ${isTop ? 'font-semibold text-forest' : 'text-slate-600'}">${cls}</span>
          <div class="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
            <div class="prob-bar h-full rounded-full ${isTop ? 'bg-forest' : 'bg-slate-300'}" style="width:${pct}%"></div>
          </div>
          <span class="w-8 text-right ${isTop ? 'font-bold text-forest' : 'text-slate-500'}">${pct}%</span>
        </div>`;
    });
    document.getElementById('msg-classify').textContent = '✓ Classification complete';
  } catch(e) {
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

document.getElementById('btn-anomaly').addEventListener('click', async () => {
  if (!fileAnomaly) { document.getElementById('msg-anomaly').textContent = 'Choose an image first.'; return; }
  const btn = document.getElementById('btn-anomaly');
  btn.disabled = true; btn.textContent = 'Analysing...';
  try {
    const fd = new FormData(); fd.append('file', fileAnomaly);
    const r  = await fetch('/anomaly', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Anomaly detection failed');

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
    document.getElementById('msg-anomaly').textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false; btn.textContent = 'Detect Anomaly';
  }
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

document.getElementById('btn-change').addEventListener('click', async () => {
  if (!fileBefore || !fileAfter) { document.getElementById('msg-change').textContent = 'Upload both images first.'; return; }
  const btn = document.getElementById('btn-change');
  btn.disabled = true; btn.textContent = 'Analysing...';
  try {
    const fd = new FormData(); fd.append('before', fileBefore); fd.append('after', fileAfter);
    const r  = await fetch('/change-detect', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Change detection failed');

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
    document.getElementById('msg-change').textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false; btn.textContent = 'Detect Changes';
  }
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

document.getElementById('btn-segment').addEventListener('click', async () => {
  if (!fileSegment) { document.getElementById('msg-segment').textContent = 'Choose an image first.'; return; }
  const btn = document.getElementById('btn-segment');
  btn.disabled = true; btn.textContent = 'Detecting...';
  try {
    const fd = new FormData(); fd.append('file', fileSegment);
    const thresh = parseFloat(confSlider.value);
    const r  = await fetch(`/segment?confidence_threshold=${thresh}`, { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Segmentation failed');

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
    document.getElementById('msg-segment').textContent = `✓ Found ${d.num_trees} tree crown${d.num_trees !== 1 ? 's' : ''}`;
  } catch(e) {
    document.getElementById('msg-segment').textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false; btn.textContent = 'Detect Tree Crowns';
  }
});

// ─── TAB 5: LIDAR ────────────────────────────────────────────────────────────
let fileLidar = null;
setupDropzone('dropzone-lidar', 'file-lidar', f => {
  fileLidar = f;
  const prev = document.getElementById('preview-lidar');
  prev.textContent = `📁 ${f.name}  (${(f.size / 1024 / 1024).toFixed(1)} MB)`;
  prev.classList.remove('hidden');
});

document.getElementById('btn-lidar').addEventListener('click', async () => {
  if (!fileLidar) { document.getElementById('msg-lidar').textContent = 'Choose a LAS/LAZ file first.'; return; }
  const btn = document.getElementById('btn-lidar');
  btn.disabled = true; btn.textContent = 'Processing...';
  document.getElementById('msg-lidar').textContent = 'Analysing point cloud (this may take 30s)...';
  try {
    const fd = new FormData(); fd.append('file', fileLidar);
    const r  = await fetch('/pointcloud', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Point cloud processing failed');

    document.getElementById('result-lidar-empty').classList.add('hidden');
    document.getElementById('result-lidar').classList.remove('hidden');

    const statsGrid = document.getElementById('lidar-stats');
    const s = d.stats;
    const statItems = [
      { icon: '🌲', label: 'Trees detected',       value: d.num_trees_detected },
      { icon: '📏', label: 'Mean canopy height',    value: `${s.mean_canopy_height_m.toFixed(1)} m` },
      { icon: '⬆️', label: 'Max canopy height',     value: `${s.max_canopy_height_m.toFixed(1)} m` },
      { icon: '🌿', label: 'Canopy cover',          value: `${(s.canopy_cover_fraction * 100).toFixed(1)}%` },
      { icon: '🔢', label: 'Stem density',          value: `${s.stem_density_per_ha.toFixed(0)}/ha` },
      { icon: '📍', label: 'Total points',          value: s.num_points.toLocaleString() },
    ];
    statsGrid.innerHTML = statItems.map(item => `
      <div class="bg-orange-50 rounded-lg p-3 flex items-start gap-2">
        <span class="text-lg">${item.icon}</span>
        <div>
          <div class="text-xs text-slate-500">${item.label}</div>
          <div class="font-bold text-slate-800">${item.value}</div>
        </div>
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
    if (d.trees.length > 20) {
      treeList.innerHTML += `<div class="text-xs text-slate-400 text-center py-1">... and ${d.trees.length - 20} more trees</div>`;
    }
    document.getElementById('msg-lidar').textContent = '✓ Analysis complete';
  } catch(e) {
    document.getElementById('msg-lidar').textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false; btn.textContent = 'Analyse Point Cloud';
  }
});
