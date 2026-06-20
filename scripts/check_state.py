from pathlib import Path

kj = Path.home() / '.kaggle' / 'kaggle.json'
print(f'kaggle.json:  {kj.exists()}')

img_dir = Path('data/raw/aptos2019/train_images/train_images')
csv     = Path('data/raw/aptos2019/train_1.csv')
meta    = Path('data/metadata/clinical_metadata.csv')

n = len(list(img_dir.glob('*.png'))) if img_dir.exists() else 0
print(f'Train images: {n}')
print(f'train_1.csv:  {csv.exists()}')
if meta.exists():
    import pandas as pd
    df = pd.read_csv(meta)
    print(f'metadata:     {len(df)} patients, cols={list(df.columns)}')
else:
    print('metadata:     MISSING')

checks = [
    'outputs/network/patient_network_community.png',
    'outputs/network/patient_network_drgrade.png',
    'outputs/network/patient_graph.gexf',
    'outputs/models',
    'outputs/figures',
    'notebooks/01_EDA.ipynb',
    'app/streamlit_app.py',
]
print()
for p in checks:
    exists = Path(p).exists()
    print(f'  {"OK" if exists else "!!"} {p}')
