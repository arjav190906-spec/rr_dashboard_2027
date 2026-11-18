"""Runs the whole pipeline.  Usage:  python run_all.py
Expects the unzipped 'Data Pack' folder at  data/Data Pack/"""
import subprocess, sys, pathlib
root = pathlib.Path(__file__).parent
data = root / 'data' / 'Data Pack'
if not (data / 'ball_by_ball.csv').exists():
    sys.exit(f'ball_by_ball.csv not found in {data}. Unzip the data pack into data/ first.')
steps = [
    ['src/01_ball_models.py', str(data), 'work'],
    ['src/02_scout.py', str(data), 'work', 'outputs'],
    ['src/03_validate.py', 'work'],
]
for s in steps:
    print('\n=== running', s[0], '===')
    r = subprocess.run([sys.executable, *s], cwd=root)
    if r.returncode: sys.exit(f'{s[0]} failed')
print('\nDone. See outputs/ . Dashboard:  streamlit run app.py')
