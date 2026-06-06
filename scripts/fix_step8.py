import json

with open(r'C:\Users\sriva\Desktop\Projects\Personal Repositories\geospatial\.claude\worktrees\beautiful-shaw-ecc6e7\train_on_colab.ipynb') as f:
    nb = json.load(f)

new_source = """\
import shutil, json, subprocess, os, datetime
from pathlib import Path
from google.colab import auth, userdata
from googleapiclient.discovery import build

# ── Config ────────────────────────────────────────────────────────────────
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
REPO         = 'iamvisheshsrivastava/geospatial'
BRANCH       = 'main'
RUN_TS       = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')

# ── 1. Copy checkpoints to Google Drive ───────────────────────────────────
FILES = ['best_model.pt', 'autoencoder_best.pt', 'segmentation_best.pt']
print('Copying checkpoints to Google Drive...')
for fname in FILES:
    src = Path('checkpoints') / fname
    dst = Path(DRIVE_OUTPUT) / fname
    if not src.exists():
        raise FileNotFoundError(f'{fname} not found — run training cells first.')
    shutil.copy2(src, dst)
    print(f'  OK  {fname}  ({dst.stat().st_size / 1_048_576:.1f} MB)')

# ── 2. Get Drive file IDs & make publicly readable ────────────────────────
print('\\nGetting Google Drive file IDs...')
auth.authenticate_user()
service = build('drive', 'v3', cache_discovery=False)

folder_query = "name='geospatial_checkpoints' and mimeType='application/vnd.google-apps.folder' and trashed=false"
res = service.files().list(q=folder_query, fields='files(id, name)').execute()
if not res['files']:
    raise RuntimeError("Folder 'geospatial_checkpoints' not found in Drive.")
folder_id = res['files'][0]['id']

KEY_MAP = {
    'best_model.pt':        'best_model',
    'autoencoder_best.pt':  'autoencoder',
    'segmentation_best.pt': 'segmentation',
}

config = {}
for fname, key in KEY_MAP.items():
    file_query = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=file_query, fields='files(id, name)').execute()
    if not results['files']:
        raise RuntimeError(f"'{fname}' not found in Drive folder.")
    file_id = results['files'][0]['id']
    service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    config[key] = file_id
    print(f'  {fname}: {file_id}')

# Versioning: timestamp changes every run so git always has something to commit
config['_trained_at'] = RUN_TS
config['_note'] = 'Auto-updated by train_on_colab.ipynb — do not edit manually.'

# ── 3. Clone repo, update model_config.json, force push ──────────────────
print('\\nPushing model_config.json to GitHub...')

repo_dir = Path('/content/repo_push')
if repo_dir.exists():
    shutil.rmtree(repo_dir)

subprocess.run([
    'git', 'clone', '--depth', '1',
    f'https://{GITHUB_TOKEN}@github.com/{REPO}.git',
    str(repo_dir)
], check=True)

config_path = repo_dir / 'model_config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

subprocess.run(['git', '-C', str(repo_dir), 'config', 'user.email', 'colab-training@auto.bot'], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'config', 'user.name', 'Colab Training Bot'], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'add', 'model_config.json'], check=True)
subprocess.run([
    'git', '-C', str(repo_dir), 'commit', '--allow-empty', '-m',
    f'chore(models): training run {RUN_TS} — force redeploy to Heroku'
], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'push', 'origin', BRANCH], check=True)

print('\\n' + '='*60)
print(f'  DONE — run {RUN_TS}')
print('='*60)
print('model_config.json pushed to GitHub.')
print('GitHub Actions → Docker build → Heroku deploy (~12 min)')
print('='*60)
"""

nb['cells'][18]['source'] = new_source

with open(r'C:\Users\sriva\Desktop\Projects\Personal Repositories\geospatial\.claude\worktrees\beautiful-shaw-ecc6e7\train_on_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('Updated Step 8.')
