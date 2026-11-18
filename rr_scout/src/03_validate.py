"""
STEP 3 - Does the method actually predict the FUTURE?  (backtest on season s -> season s+1)
Usage: python 03_validate.py <work_dir>
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
W = sys.argv[1].rstrip('/') + '/'
L = pd.read_pickle(W + 'balls.pkl')
def eb_k(df, col, min_n):
    g = df.groupby('pid')[col].agg(['mean', 'count']); s2 = df[col].var(); g = g[g['count'] >= min_n]
    return s2 / max(g['mean'].var() - (s2 / g['count']).mean(), 1e-5)
rows = []
for role, pid, val in (('bat', 'striker_id', 'bat_val'), ('bowl', 'bowler_id', 'bowl_val')):
    X = L.rename(columns={pid: 'pid', val: 'v'})
    k = eb_k(X, 'v', 100); kp = eb_k(X[X.pressure == 1], 'v', 40)
    for s in (2023, 2024, 2025):
        past, fut = X[X.year <= s], X[X.year == s + 1]
        a = past.groupby('pid').v.agg(['mean', 'count']); a.columns = ['raw', 'n']
        pp = past[past.pressure == 1].groupby('pid').v.agg(['mean', 'count']); pp.columns = ['praw', 'pn']
        f = fut.groupby('pid').v.agg(['mean', 'count']); f.columns = ['fut', 'fn']
        d = a.join(pp).join(f, how='inner').fillna({'praw': 0, 'pn': 0}); d = d[(d.n >= 60) & (d.fn >= 60)]
        d['shr'] = d.raw * d.n / (d.n + k); d['pshr'] = d.praw * d.pn / (d.pn + kp); d['pair'] = f'{s}->{s+1}'; d['role'] = role
        rows.append(d)
R = pd.concat(rows)
print('Predicting NEXT-season value/ball from data up to this season (players with >=60 balls in both)\n')
out = []
for role in ('bat', 'bowl'):
    d = R[R.role == role]
    r = lambda c: round(spearmanr(d[c], d.fut)[0], 3)
    out.append((role, len(d), r('raw'), r('shr'), r('pshr')))
    # does pressure value add info beyond overall? (OLS on standardised vars)
    Z = (d[['shr', 'pshr']] - d[['shr', 'pshr']].mean()) / d[['shr', 'pshr']].std(); y = (d.fut - d.fut.mean()) / d.fut.std()
    b = np.linalg.lstsq(np.c_[Z.values, np.ones(len(Z))], y.values, rcond=None)[0]
    print(f'{role}: n={len(d)}  rank-corr with next season:  raw={r("raw")}  shrunk-overall={r("shr")}  shrunk-pressure={r("pshr")}   | OLS beta overall={b[0]:.2f}, pressure={b[1]:.2f}')
print('\n-> if shrunk > raw, shrinkage helps. If pressure beta ~ 0, "pressure skill" adds little beyond overall quality (consistent with literature).')

# hidden-gem backtest: pick uncapped, low-visibility players using ONLY data up to 2024, check 2025-26
print('\nHIDDEN-GEM BACKTEST (selected with data <=2024, judged on 2025-26)')
for role, pid, val in (('bat', 'striker_id', 'bat_val'), ('bowl', 'bowler_id', 'bowl_val')):
    X = L.rename(columns={pid: 'pid', val: 'v'}); k = eb_k(X, 'v', 100); kp = eb_k(X[X.pressure == 1], 'v', 40)
    past, fut = X[X.year <= 2024], X[X.year >= 2025]
    ipl_games = past[past.is_ipl == 1].groupby('pid').match_id.nunique()
    t20i_any = set(X[(X.is_ipl == 0) & ~X.comp_name.str.contains('Emerging|Rising')].pid)   # played a non-IPL, non-emerging (i.e. international) match
    a = past.groupby('pid').v.agg(['mean', 'count']); pp = past[past.pressure == 1].groupby('pid').v.agg(['mean', 'count'])
    a['score'] = a['mean'] * a['count'] / (a['count'] + k) + pp['mean'].reindex(a.index).fillna(0) * pp['count'].reindex(a.index).fillna(0) / (pp['count'].reindex(a.index).fillna(0) + kp)
    a['ipl'] = ipl_games.reindex(a.index).fillna(0); a['tt'] = a.index.isin(t20i_any)
    cand = a[(a['count'] >= 60) & (a.ipl <= 40) & (~a.tt)]
    f = fut.groupby('pid').v.agg(['mean', 'count']); f = f[f['count'] >= 30]
    cand = cand.join(f, rsuffix='_f', how='inner')
    top = cand.sort_values('score', ascending=False).head(12); rest = cand.drop(top.index)
    print(f'{role}: candidates with future data={len(cand)} | top-12 mean future value/ball={top["mean_f"].mean():+.3f}  vs rest={rest["mean_f"].mean():+.3f}  (0 = league avg)  | rank-corr score vs future={spearmanr(cand.score, cand.mean_f)[0]:.2f}')
