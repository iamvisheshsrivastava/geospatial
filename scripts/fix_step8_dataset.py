import json

with open(r'C:\Users\sriva\Desktop\Projects\Personal Repositories\geospatial\.claude\worktrees\beautiful-shaw-ecc6e7\train_on_colab.ipynb') as f:
    nb = json.load(f)

# Update Step 8 markdown
nb['cells'][17]['source'] = (
    '## Step 8 — Upload checkpoints to Kaggle Dataset → triggers Heroku deploy\n\n'
    'This cell:\n'
    '1. Uploads the 3 model checkpoints as a **Kaggle Dataset** (`geospatial-checkpoints`)\n'
    '2. Pushes a commit to GitHub → triggers GitHub Actions\n'
    '3. GitHub Actions downloads from the dataset → builds Docker → deploys to Heroku\n\n'
    '**No Google Drive needed. Works every time without committing the notebook.**\n\n'
    '**Before running:** add `GITHUB_TOKEN` to Kaggle Secrets (Add-on → Secrets).'
)

# Update Step 8 code
nb['cells'][18]['source'] = '''\
import subprocess, json, datetime, shutil, os
from pathlib import Path
from kaggle_secrets import UserSecretsClient

GITHUB_TOKEN = UserSecretsClient().get_secret('GITHUB_TOKEN')
KAGGLE_TOKEN = UserSecretsClient().get_secret('KAGGLE_API_TOKEN') if 'KAGGLE_API_TOKEN' in [s.label for s in UserSecretsClient().get_secrets()] else None
REPO   = 'iamvisheshsrivastava/geospatial'
BRANCH = 'main'
RUN_TS = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')

# Set up Kaggle API credentials
os.environ['KAGGLE_API_TOKEN'] = KAGGLE_TOKEN or ''

# ── 1. Verify checkpoints exist ───────────────────────────────────────────
FILES = ['best_model.pt', 'autoencoder_best.pt', 'segmentation_best.pt']
print('Verifying checkpoints...')
for fname in FILES:
    p = Path('checkpoints') / fname
    if not p.exists():
        raise FileNotFoundError(f'{fname} not found — run training cells first.')
    print(f'  OK  {fname}  ({p.stat().st_size / 1_048_576:.1f} MB)')

# ── 2. Upload to Kaggle Dataset ───────────────────────────────────────────
dataset_dir = Path('/kaggle/working/dataset_upload')
dataset_dir.mkdir(exist_ok=True)

# Copy checkpoints into upload folder
for fname in FILES:
    shutil.copy2(Path('checkpoints') / fname, dataset_dir / fname)

# Write dataset metadata
metadata = {
    'title': 'Geospatial ML Checkpoints',
    'id': 'visheshsrivastava/geospatial-checkpoints',
    'licenses': [{'name': 'CC0-1.0'}]
}
with open(dataset_dir / 'dataset-metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'\\nUploading checkpoints to Kaggle Dataset...')

# Try to create new dataset; if it exists, add a new version instead
result = subprocess.run(
    ['kaggle', 'datasets', 'create', '-p', str(dataset_dir), '--dir-mode', 'zip'],
    capture_output=True, text=True
)
if 'already exists' in result.stdout + result.stderr or result.returncode != 0:
    print('Dataset exists — uploading new version...')
    result = subprocess.run(
        ['kaggle', 'datasets', 'version', '-p', str(dataset_dir),
         '-m', f'Training run {RUN_TS}', '--dir-mode', 'zip'],
        capture_output=True, text=True
    )
print(result.stdout or result.stderr)

# ── 3. Push trigger commit to GitHub ─────────────────────────────────────
print('Pushing trigger commit to GitHub...')

repo_dir = Path('/kaggle/working/repo_push')
if repo_dir.exists():
    shutil.rmtree(repo_dir)

subprocess.run([
    'git', 'clone', '--depth', '1',
    f'https://{GITHUB_TOKEN}@github.com/{REPO}.git', str(repo_dir)
], check=True)

trigger = {
    '_trained_at': RUN_TS,
    '_dataset': 'visheshsrivastava/geospatial-checkpoints',
    '_note': 'Auto-updated by Kaggle notebook — do not edit manually.'
}
with open(repo_dir / 'model_config.json', 'w') as f:
    json.dump(trigger, f, indent=2)

subprocess.run(['git', '-C', str(repo_dir), 'config', 'user.email', 'kaggle-training@auto.bot'], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'config', 'user.name', 'Kaggle Training Bot'], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'add', 'model_config.json'], check=True)
subprocess.run([
    'git', '-C', str(repo_dir), 'commit', '--allow-empty', '-m',
    f'chore(models): training run {RUN_TS} — deploy from Kaggle dataset'
], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'push', 'origin', BRANCH], check=True)

print('\\n' + '='*60)
print(f'  DONE — run {RUN_TS}')
print('='*60)
print('Checkpoints uploaded to: kaggle.com/visheshsrivastava/geospatial-checkpoints')
print('GitHub Actions will now:')
print('  1. Download checkpoints from Kaggle Dataset')
print('  2. Build Docker image')
print('  3. Deploy to Heroku (~12 min)')
print('Watch: https://github.com/iamvisheshsrivastava/geospatial/actions')
print('='*60)
'''

with open(r'C:\Users\sriva\Desktop\Projects\Personal Repositories\geospatial\.claude\worktrees\beautiful-shaw-ecc6e7\train_on_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('Notebook updated.')
