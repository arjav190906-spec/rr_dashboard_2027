# RR Scout - laptop setup

## 1. Install (one time)
- Install Python 3.10+ from python.org (Windows: tick "Add Python to PATH").
- Open a terminal in this folder (rr_scout) and run:
    python -m venv venv
    Windows:  venv\Scripts\activate        Mac/Linux:  source venv/bin/activate
    pip install -r requirements.txt

## 2. Put the data
Unzip the data pack so that this file exists:  data/Data Pack/ball_by_ball.csv

## 3. Run
    python run_all.py          (about 2-3 minutes)
Results appear in outputs/ : shortlist_batters.csv, shortlist_bowlers.csv, scout_all_*.csv

## 4. Dashboard (optional)
    streamlit run app.py

## 5. Files
src/01_ball_models.py  ball table + 2 ML models (outcome, win-probability) -> leverage/pressure, value over expected
src/02_scout.py        shrinkage, multi-factor score, uncapped/limelight filters, RR need, shortlists
src/03_validate.py     backtest season s -> s+1
Edit KNOWN_CAPPED in src/02_scout.py (data starts 2022; verify capped status yourself).
