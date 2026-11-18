"""
STEP 2 - Player scouting table (Empirical-Bayes shrinkage + multi-factor score + 'under the radar' flags)
Usage: python 02_scout.py "<Data Pack folder>" <work_dir> <out_dir>
Output: scout_bat.csv, scout_bowl.csv (all players)  +  shortlist_bat.csv, shortlist_bowl.csv (uncapped & under-the-radar)
"""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
D = sys.argv[1].rstrip('/') + '/'; W = sys.argv[2].rstrip('/') + '/'; O = sys.argv[3].rstrip('/') + '/'
L = pd.read_pickle(W + 'balls.pkl')

# ------------------------------------------------------------------ player metadata (limelight / capped / franchise)
mp = pd.read_csv(D + 'competition_team_player_mapping.csv'); comps = pd.read_csv(D + 'competitions.csv')
mp = mp.merge(comps[['comp_id', 'match_format']], on='comp_id')
pe = pd.read_csv(D + 'playing_eleven.csv'); pe = pe[pe.is_played.astype(str).str.lower() == 'true']
mt = pd.read_csv(D + 'matches.csv')[['match_id', 'comp_id']].merge(comps[['comp_id', 'comp_name', 'match_format']], on='comp_id')
pe = pe.merge(mt, on='match_id')
t20i_m = pe[pe.match_format == 'T20I'].groupby('player_id').size()
ipl = pe[pe.comp_name.str.startswith('IPL')]
ipl_m = ipl.groupby('player_id').size()
ipl26 = ipl[ipl.comp_name == 'IPL 2026']
team_m26 = ipl26.groupby('team_id').match_id.nunique()
u = ipl26.groupby(['player_id', 'team_id']).size().reset_index(name='m26'); u['usage26'] = u.m26 / u.team_id.map(team_m26)
usage26 = u.sort_values('usage26', ascending=False).drop_duplicates('player_id').set_index('player_id').usage26
squad26 = mp[mp.comp_id == 3298].drop_duplicates('player_id').set_index('player_id').team_name
emer = mp[mp.comp_name.str.contains('Emerging|Rising')].groupby('player_id').team_name.agg(lambda x: x.value_counts().index[0])
names = mp.drop_duplicates('player_id').set_index('player_id').player_name
RR = set(mp[(mp.comp_id == 3298) & (mp.team_id == 3)].player_id)

# Players known (from public knowledge) to have international caps BEFORE 2022 or in non-T20I formats -> NOT uncapped.
# The data pack only starts in 2022, so this list MUST be verified/extended by you. Put player names here.
KNOWN_CAPPED = {'Jaydev Unadkat', 'Mohit Sharma', 'Vijay Shankar', 'Navdeep Saini', 'T Natarajan', 'Krunal Pandya', 'Sunil Narine',
                'Faf du Plessis', 'Mahendra Singh Dhoni', 'Shivam Dube', 'Mohammed Shami', 'Piyush Chawla', 'Amit Mishra',
                'Karn Sharma', 'Shahbaz Ahmed', 'Washington Sundar', 'Karun Nair'}

# ------------------------------------------------------------------ empirical-Bayes helper
def eb_k(df, key, col, min_n):
    """k = sigma^2 / tau^2  (how many 'pseudo-balls' of league-average evidence to blend in). Method of moments."""
    g = df.groupby(key)[col].agg(['mean', 'count'])
    sig2 = df[col].var()
    g = g[g['count'] >= min_n]
    tau2 = max(g['mean'].var() - (sig2 / g['count']).mean(), 1e-5)
    return sig2 / tau2, sig2

def build(role):
    if role == 'bat':
        pid, val, nm = 'striker_id', 'bat_val', 'striker_name'
        X = L.copy()
    else:
        pid, val, nm = 'bowler_id', 'bowl_val', 'bowler_name'
        X = L.copy()
    X = X.rename(columns={pid: 'pid', val: 'v'})
    k_all, s2 = eb_k(X, 'pid', 'v', 100)
    P = X[X.pressure == 1]
    k_p, s2p = eb_k(P, 'pid', 'v', 40)
    print(role, ': shrink k_all=%.0f balls, k_pressure=%.0f pressure-balls' % (k_all, k_p))
    g = X.groupby('pid').agg(balls=('v', 'size'), ov=('v', 'mean'), last_year=('year', 'max'), first_year=('year', 'min'), matches=('match_id', 'nunique'))
    gp = P.groupby('pid').agg(p_balls=('v', 'size'), pv=('v', 'mean'))
    gc = X[X.critical == 1].groupby('pid').agg(c_balls=('v', 'size'), cv=('v', 'mean'))
    gn = X[X.pressure == 0].groupby('pid').agg(n_balls=('v', 'size'), nv=('v', 'mean'))
    g = g.join([gp, gc, gn]).fillna({'p_balls': 0, 'pv': 0, 'c_balls': 0, 'cv': 0, 'n_balls': 0, 'nv': 0})
    g['ov_s'] = g.ov * g.balls / (g.balls + k_all)                       # shrunk overall value / ball (runs above expected)
    g['pv_s'] = g.pv * g.p_balls / (g.p_balls + k_p)                     # shrunk pressure value / ball
    g['sd_post'] = np.sqrt(s2 / (g.balls + k_all))
    g['p_above_avg'] = norm.cdf(g.ov_s / g.sd_post)                      # probability true skill > league average
    # consistency: share of matches where player beat expectation (shrunk to league share)
    pm = X.groupby(['pid', 'match_id']).v.agg(['sum', 'size']).reset_index(); pm = pm[pm['size'] >= 6]
    pm['pos'] = (pm['sum'] > 0).astype(int)
    p0 = pm.pos.mean(); gm = pm.groupby('pid').pos.agg(['sum', 'count'])
    g['hit_rate'] = ((gm['sum'] + 8 * p0) / (gm['count'] + 8)).reindex(g.index).fillna(p0)
    g['n_good_games'] = gm['count'].reindex(g.index).fillna(0)
    # recent form (2025-26 only), shrunk
    R = X[X.year >= 2025]; k_r = k_all
    gr = R.groupby('pid').v.agg(['mean', 'count'])
    g['recent_balls'] = gr['count'].reindex(g.index).fillna(0)
    g['recent_s'] = (gr['mean'] * gr['count'] / (gr['count'] + k_r)).reindex(g.index).fillna(0)
    # role descriptors
    if role == 'bat':
        d = X.groupby('pid').agg(avg_order=('batting_order', 'mean'), death_share=('phase', lambda s: (s == 'DEATH').mean()), pp_share=('phase', lambda s: (s == 'PP').mean()))
        g = g.join(d)
        g['slot'] = pd.cut(g.avg_order, [0, 2.5, 4.5, 6.5, 12], labels=['Opener', 'Top3-4', 'Middle5-6', 'Finisher7+']).astype(str)
        raw = X.groupby('pid').agg(runs=('runs', 'sum'), outs=('striker_out', 'sum'))
        g['sr'] = (raw.runs / g.balls * 100).round(1); g['balls_per_out'] = (g.balls / raw.outs.clip(lower=1)).round(1)
        pr = P.groupby('pid').agg(pruns=('runs', 'sum'), pouts=('striker_out', 'sum'), pb=('runs', 'size'))
        g['sr_pressure'] = (pr.pruns / pr.pb * 100).reindex(g.index).round(1)
    else:
        d = X.groupby('pid').agg(pace=('pace_or_spin', lambda s: (s == 'Pace').mean()), death_share=('phase', lambda s: (s == 'DEATH').mean()), pp_share=('phase', lambda s: (s == 'PP').mean()),
                                 btype=('bowling_type', lambda s: s.value_counts().index[0]))
        g = g.join(d)
        g['slot'] = np.where(g.pace > .5, 'Pace', 'Spin')
        raw = X.groupby('pid').agg(runs=('runs', 'sum'), w=('bowler_w', 'sum'))
        g['econ'] = (raw.runs / g.balls * 6).round(2); g['balls_per_wkt'] = (g.balls / raw.w.clip(lower=1)).round(1)
        pr = P.groupby('pid').agg(pruns=('runs', 'sum'), pb=('runs', 'size'))
        g['econ_pressure'] = (pr.pruns / pr.pb * 6).reindex(g.index).round(2)
    # reference pool for z-scores = established players with decent sample
    ref = g[g.balls >= 120]
    z = lambda c: (g[c] - ref[c].mean()) / ref[c].std()
    g['S_raw'] = .35 * z('ov_s') + .30 * z('pv_s') + .20 * z('hit_rate') + .15 * z('recent_s')
    g['scout'] = 50 + 10 * (g.S_raw - g.S_raw[ref.index].mean()) / g.S_raw[ref.index].std()
    g['role'] = role; g = g.reset_index().rename(columns={'index': 'pid'})
    g['name'] = g.pid.map(names)
    return g, X

def add_meta(g):
    g['t20i_m'] = g.pid.map(t20i_m).fillna(0).astype(int); g['ipl_m'] = g.pid.map(ipl_m).fillna(0).astype(int)
    g['usage26'] = g.pid.map(usage26); g['squad26'] = g.pid.map(squad26); g['emer_team'] = g.pid.map(emer)
    g['rr'] = g.pid.isin(RR)
    g['uncapped_in_data'] = g.t20i_m == 0
    g['known_capped'] = g['name'].isin(KNOWN_CAPPED)
    g['limelight'] = np.select([g.ipl_m <= 15, g.ipl_m <= 40], ['Hidden (<=15 IPL games)', 'Fringe (16-40)'], 'Established (40+)')
    # availability hint (NOT a fact): who could plausibly hit the auction
    g['pool_hint'] = np.select([g.squad26.isna(), g.usage26.fillna(0) < .35], ['Not in an IPL-2026 squad -> likely in pool', 'In squad but rarely played -> possible release'], 'Regular starter -> likely retained')
    g['confidence'] = pd.cut(g.balls, [0, 60, 120, 250, 1e5], labels=['Very low', 'Low', 'Medium', 'High']).astype(str)
    return g

bat, Xb = build('bat'); bowl, Xw = build('bowl')
bat = add_meta(bat); bowl = add_meta(bowl)

# ------------------------------------------------------------------ RR need (where RR leaks vs league, IPL 2025-26)
need = {}
for role, X, g in (('bat', Xb, bat), ('bowl', Xw, bowl)):
    seg = g.set_index('pid').slot
    Y = X[(X.is_ipl == 1) & (X.year >= 2025)].copy(); Y['slot'] = Y.pid.map(seg)
    Y['rr'] = Y.batting_team_id.eq(3) if role == 'bat' else Y.bowling_team_id.eq(3)
    t = Y[Y.pressure == 1].groupby(['slot', 'rr']).v.agg(['mean', 'count', 'std']).reset_index()
    for s_ in t.slot.unique():
        a = t[(t.slot == s_) & (t.rr)]; l = t[(t.slot == s_) & (~t.rr)]
        if len(a) and len(l): need[(role, s_)] = float((a['mean'].iloc[0] - l['mean'].iloc[0]) / (a['std'].iloc[0] / np.sqrt(a['count'].iloc[0])))
print('RR gap z (negative = RR weak vs league, pressure balls IPL 25-26):'); [print('  ', k, round(v, 2)) for k, v in need.items()]
for role, g in (('bat', bat), ('bowl', bowl)):
    g['rr_need_z'] = [-need.get((role, s_), 0.0) for s_ in g.slot]       # positive = RR is weak in this segment -> player's segment is a need

# ------------------------------------------------------------------ shortlist rules
def shortlist(g, min_balls=60):
    s = g[(g.uncapped_in_data) & (~g.known_capped) & (g.last_year >= 2025) & (g.balls >= min_balls)
          & (~g.emer_team.fillna('').str.contains('Pakistan')) & (g.limelight != 'Established (40+)')].copy()
    s['visibility_pct'] = s.ipl_m.rank(pct=True)
    s['hidden_gem'] = (s.scout - 50) - 5 * s.visibility_pct             # same quality but less visible -> higher
    s['final'] = s.scout + 1.5 * s.rr_need_z.clip(-2, 2)                # small RR-fit nudge
    return s.sort_values('final', ascending=False)

cols = ['name', 'slot', 'squad26', 'emer_team', 'pool_hint', 'limelight', 'ipl_m', 'usage26', 'balls', 'p_balls', 'confidence', 'scout', 'final', 'hidden_gem', 'ov_s', 'pv_s', 'hit_rate', 'recent_s', 'p_above_avg', 'rr_need_z']
extra_bat = ['sr', 'sr_pressure', 'balls_per_out', 'avg_order', 'death_share']; extra_bowl = ['econ', 'econ_pressure', 'balls_per_wkt', 'btype', 'death_share', 'pp_share']
sb, sw = shortlist(bat), shortlist(bowl)
bat.to_csv(O + 'scout_all_batters.csv', index=False); bowl.to_csv(O + 'scout_all_bowlers.csv', index=False)
sb[cols + extra_bat].round(3).to_csv(O + 'shortlist_batters.csv', index=False); sw[cols + extra_bowl].round(3).to_csv(O + 'shortlist_bowlers.csv', index=False)
bat.to_pickle(W + 'bat.pkl'); bowl.to_pickle(W + 'bowl.pkl')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('\nBATTERS', len(sb)); print(sb[['name', 'slot', 'squad26', 'emer_team', 'ipl_m', 'balls', 'p_balls', 'scout', 'final', 'ov_s', 'pv_s', 'hit_rate', 'p_above_avg', 'sr', 'sr_pressure']].head(20).round(3).to_string(index=False))
print('\nBOWLERS', len(sw)); print(sw[['name', 'slot', 'squad26', 'emer_team', 'ipl_m', 'balls', 'p_balls', 'scout', 'final', 'ov_s', 'pv_s', 'hit_rate', 'p_above_avg', 'econ', 'econ_pressure']].head(20).round(3).to_string(index=False))
