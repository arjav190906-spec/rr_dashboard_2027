"""
STEP 1 - Build ball table + 2 ML models (outcome model, win-probability model)
Output: work/balls.pkl  (one row per ball, with expected value, leverage, pressure flags)

Usage:  python 01_ball_models.py "<path to Data Pack folder>" <work_dir>
"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss

D = sys.argv[1].rstrip('/') + '/'
W = sys.argv[2].rstrip('/') + '/'
RNG = 7

# ---------------------------------------------------------------- 1. load + clean
b = pd.read_csv(D + 'ball_by_ball.csv', low_memory=False)
m = pd.read_csv(D + 'matches.csv')
c = pd.read_csv(D + 'competitions.csv')
m = m.merge(c[['comp_id', 'comp_name', 'match_format']], on='comp_id')
m['year'] = pd.to_datetime(m.match_date).dt.year
m['is_ipl'] = m.comp_name.str.startswith('IPL').astype(int)
# drop matches with rain-reduced overs / revised targets and matches with no winner (ties/NR)
bad = m.revised_target.notna() | m.first_inn_revised_overs.notna() | m.second_inn_revised_overs.notna() | m.win_team_id.isna()
m = m[~bad]
b = b[b.match_id.isin(m.match_id) & b.innings.isin([1, 2])].copy()
b = b.merge(m[['match_id', 'year', 'is_ipl', 'comp_name', 'win_team_id']], on='match_id')
b = b.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)

b['legal'] = (~b.extras_type.isin(['wide', 'Overstep'])).astype(int)
b['team_runs'] = b.runs + b.extras1_runs + b.extras_2_runs
b['is_w'] = (b.is_wicket & (b.dismissal_type != 'Retire Out')).astype(int)
b['bowler_w'] = (b.is_w == 1) & ~b.dismissal_type.isin(['RunOut', 'Obstracting the field', 'Obstructing the field'])
b['bowler_w'] = b.bowler_w.astype(int)
b['striker_out'] = ((b.is_w == 1) & (b.dismissed_player_id == b.striker_id)).astype(int)
# runs charged to bowler on illegal balls (wides / no-balls); attach to the next legal ball
b['illegal_charge'] = np.where(b.legal == 0, b.team_runs, 0)

g = b.groupby(['match_id', 'innings'])
b['runs_before'] = g.team_runs.cumsum() - b.team_runs
b['wkts_before'] = g.is_w.cumsum() - b.is_w
b['legal_before'] = g.legal.cumsum() - b.legal
b['balls_left'] = 120 - b.legal_before
b['extra_attached'] = g.illegal_charge.cumsum()
# extras charged BEFORE-and-including illegal balls since last legal ball -> assigned to next legal ball
b['_grp'] = g.legal.cumsum()               # increments at each legal ball
ex = b.groupby(['match_id', 'innings', '_grp']).illegal_charge.transform('sum')
b['extras_for_legal'] = 0.0
last_in_grp = b.groupby(['match_id', 'innings', '_grp']).cumcount(ascending=False) == 0
# illegal balls come BEFORE their legal ball in a group only if grp counter logic holds; simple approach:
# charge each illegal ball's runs to the first legal ball that follows (or precedes) inside the same over
b['over_key'] = b.match_id.astype(str) + '_' + b.innings.astype(str) + '_' + b.overid.astype(str)
b['over_ill'] = b.groupby('over_key').illegal_charge.transform('sum')
b['over_legal_n'] = b.groupby('over_key').legal.transform('sum').clip(lower=1)
b['extras_for_legal'] = np.where(b.legal == 1, b.over_ill / b.over_legal_n, 0.0)  # spread over the over's legal balls

# target / chase features
tot1 = b[b.innings == 1].groupby('match_id').team_runs.sum()
b['target'] = np.where(b.innings == 2, b.match_id.map(tot1) + 1, np.nan)
b['need'] = b.target - b.runs_before
b['crr'] = np.where(b.legal_before > 0, b.runs_before * 6 / b.legal_before.clip(lower=1), 0.0)
b['rrr'] = np.where(b.innings == 2, b.need * 6 / b.balls_left.clip(lower=1), np.nan)
b['rr_gap'] = b.rrr - b.crr
b['bat_wins'] = (b.batting_team_id == b.win_team_id).astype(int)
b['over0'] = b.overid - 1
b['phase'] = pd.cut(b.over0, [-1, 5, 14, 19], labels=['PP', 'MID', 'DEATH']).astype(str)

L = b[b.legal == 1].copy()          # modelling table = legal deliveries only
print('legal balls', len(L), 'matches', L.match_id.nunique())

# ---------------------------------------------------------------- 2. MODEL A: ball-outcome model (multiclass)
CLS = ['W', '0', '1', '2', '3', '4', '6']
CLS_RUNS = np.array([0, 0, 1, 2, 3, 4, 6])
def outcome_class(r):
    if r.is_w: return 0
    k = min(int(r.runs), 6)
    return {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}.get(k, 6) if k < 5 else 6
y_out = np.where(L.is_w == 1, 0, np.select([L.runs == 0, L.runs == 1, L.runs == 2, L.runs == 3, L.runs == 4], [1, 2, 3, 4, 5], 6))
FEAT_O = ['innings', 'over0', 'balls_left', 'wkts_before', 'runs_before', 'crr', 'rrr', 'rr_gap', 'need', 'is_ipl']
L['pace'] = (L.pace_or_spin == 'Pace').astype(int)
L['bat_lh'] = (L.batting_type.astype(str).str.upper().str.startswith('L')).astype(int)
FEAT_O += ['pace', 'bat_lh', 'batting_order']

gkf = GroupKFold(n_splits=5)
P = np.zeros((len(L), 7))
for tr, te in gkf.split(L, y_out, L.match_id):
    mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.07, max_leaf_nodes=24, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=RNG)
    mdl.fit(L.iloc[tr][FEAT_O], y_out[tr])
    P[te] = mdl.predict_proba(L.iloc[te][FEAT_O])
print('Outcome model OOF logloss', round(log_loss(y_out, P), 4), ' (naive prior:', round(log_loss(y_out, np.tile(np.bincount(y_out) / len(y_out), (len(y_out), 1))), 4), ')')
for i, k in enumerate(CLS): L['p_' + k] = P[:, i]

# ---------------------------------------------------------------- 3. MODEL B: win-probability model (batting team wins)
FEAT_W = ['innings', 'balls_left', 'wkts_before', 'runs_before', 'need', 'rrr', 'crr', 'rr_gap', 'is_ipl']
WP = np.zeros(len(L))
models_w = []
for tr, te in gkf.split(L, L.bat_wins, L.match_id):
    # monotonic constraints = domain knowledge: more wickets lost / more runs needed / higher RRR can never RAISE win prob
    mono = {'wkts_before': -1, 'need': -1, 'rrr': -1, 'rr_gap': -1}
    mw = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=16, min_samples_leaf=600,
                                        l2_regularization=2.0, monotonic_cst=mono, random_state=RNG)
    mw.fit(L.iloc[tr][FEAT_W], L.bat_wins.iloc[tr])
    WP[te] = mw.predict_proba(L.iloc[te][FEAT_W])[:, 1]
    models_w.append(mw)
L['wp_oof'] = WP
L['wp'] = np.mean([mw.predict_proba(L[FEAT_W])[:, 1] for mw in models_w], axis=0)   # same ensemble as 'after' states -> no model-noise in leverage
for inn in (1, 2):
    s = L[L.innings == inn]
    print(f'Win-prob OOF AUC innings {inn}:', round(roc_auc_score(s.bat_wins, s.wp_oof), 3))
# calibration check (chase)
s = L[L.innings == 2].copy(); s['bin'] = pd.cut(s.wp_oof, [0, .1, .3, .5, .7, .9, 1]); print(s.groupby('bin', observed=True).agg(pred=('wp_oof', 'mean'), actual=('bat_wins', 'mean'), n=('wp_oof', 'size')).round(3))

# ---------------------------------------------------------------- 4. LEVERAGE = expected |change in win prob| on the next ball
def wp_after(Lrow, r_add, w_add):
    X = Lrow[FEAT_W].copy()
    X['runs_before'] = Lrow.runs_before + r_add; X['wkts_before'] = Lrow.wkts_before + w_add; X['balls_left'] = (Lrow.balls_left - 1).clip(lower=0)
    X['need'] = Lrow.need - r_add
    X['crr'] = X.runs_before * 6 / (121 - Lrow.balls_left).clip(lower=1)
    X['rrr'] = np.where(Lrow.innings == 2, X.need * 6 / X.balls_left.clip(lower=1), np.nan)
    X['rr_gap'] = X.rrr - X.crr
    p = np.mean([mw.predict_proba(X[FEAT_W])[:, 1] for mw in models_w], axis=0)
    chase_done = (Lrow.innings == 2) & (X.need <= 0)
    all_out = (X.wkts_before >= 10) | ((X.balls_left <= 0) & (Lrow.innings == 2))
    p = np.where(chase_done, 1.0, np.where((Lrow.innings == 2) & all_out, 0.0, p))
    return p
lev = np.zeros(len(L)); WPA_exp = np.zeros(len(L))
wp_next = {}
for i, k in enumerate(CLS):
    r = CLS_RUNS[i]; w = 1 if k == 'W' else 0
    wp_next[k] = wp_after(L, r, w)
    lev += L['p_' + k].values * np.abs(wp_next[k] - L.wp.values)
L['leverage'] = lev
# realised win-prob change for the actual ball outcome (batting-team perspective) -> used only to estimate wicket cost in runs
act_r = np.minimum(L.runs.values, 6)
L['wp_after_actual'] = wp_after(L, act_r, L.is_w.values)
L['dWP'] = L.wp_after_actual - L.wp

# ---------------------------------------------------------------- 5. wicket cost lambda (runs-equivalent of a wicket), per phase, from chase balls
lam = {}
for ph in ['PP', 'MID', 'DEATH']:
    s = L[(L.innings == 2) & (L.phase == ph)]
    A = np.c_[s.runs.clip(upper=6).values, s.is_w.values, np.ones(len(s))]
    coef = np.linalg.lstsq(A, s.dWP.values, rcond=None)[0]
    lam[ph] = float(-coef[1] / coef[0])
print('wicket cost in runs (lambda):', {k: round(v, 2) for k, v in lam.items()})
L['lam'] = L.phase.map(lam)

# ---------------------------------------------------------------- 6. expected value per ball & residual ("value over expected")
exp_r = sum(L['p_' + k].values * CLS_RUNS[i] for i, k in enumerate(CLS))
exp_w = L['p_W'].values
frac_bowler = L.bowler_w.sum() / L.is_w.sum()
frac_striker = L.striker_out.sum() / L.is_w.sum()
# batter view: runs - lam * out     bowler view: -(runs conceded + extras) + lam * wicket credited
L['bat_val'] = (L.runs - L.lam * L.striker_out) - (exp_r - L.lam * exp_w * frac_striker)
base_extra = L.groupby('over0').extras_for_legal.transform('mean')
L['bowl_val'] = -((L.runs + L.extras_for_legal) - L.lam * L.bowler_w) + ((exp_r + base_extra) - L.lam * exp_w * frac_bowler)

# ---------------------------------------------------------------- 7. pressure flags
q75, q93 = L.leverage.quantile([.75, .95])
L['pressure'] = (L.leverage >= q75).astype(int)
L['critical'] = (L.leverage >= q93).astype(int)
print('leverage cut-offs  pressure(top25%%)>=%.4f  critical(top5%%)>=%.4f' % (q75, q93))
print(L.groupby(['innings', 'phase']).agg(n=('leverage', 'size'), lev=('leverage', 'mean'), pressure_share=('pressure', 'mean')).round(3))
print(L.groupby('pressure').agg(rr_gap=('rr_gap', 'mean'), wkts=('wkts_before', 'mean'), balls_left=('balls_left', 'mean'), runs_per_ball=('runs', 'mean'), wkt_rate=('is_w', 'mean')).round(3))

keep = ['match_id', 'year', 'is_ipl', 'comp_name', 'innings', 'over0', 'phase', 'balls_left', 'wkts_before', 'rrr', 'crr', 'rr_gap', 'wp', 'leverage', 'pressure', 'critical',
        'striker_id', 'striker_name', 'bowler_id', 'bowler_name', 'batting_order', 'pace_or_spin', 'bowling_type', 'batting_type', 'runs', 'is_w', 'bat_val', 'bowl_val', 'bowler_w', 'striker_out',
        'batting_team_id', 'bowling_team_id', 'bat_wins']
L[keep].to_pickle(W + 'balls.pkl')
print('saved', W + 'balls.pkl')
