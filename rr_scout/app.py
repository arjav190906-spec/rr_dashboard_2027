# Minimal dashboard:  pip install streamlit pandas plotly   then   streamlit run app.py
import pandas as pd, streamlit as st
st.set_page_config(page_title='RR Hidden-Gem Scout', layout='wide')
side = st.sidebar.radio('Role', ['Batters', 'Bowlers'])
df = pd.read_csv('outputs/shortlist_batters.csv' if side == 'Batters' else 'outputs/shortlist_bowlers.csv')
st.title('Uncapped under-the-radar scouting board')
c1, c2, c3 = st.columns(3)
slot = c1.multiselect('Slot', sorted(df.slot.unique()))
hint = c2.multiselect('Pool hint', sorted(df.pool_hint.unique()))
minb = c3.slider('Min balls', 60, int(df.balls.max()), 60)
if slot: df = df[df.slot.isin(slot)]
if hint: df = df[df.pool_hint.isin(hint)]
df = df[df.balls >= minb]
sort = st.selectbox('Sort by', ['final', 'scout', 'hidden_gem', 'p_above_avg', 'pv_s', 'ov_s'])
st.dataframe(df.sort_values(sort, ascending=False), use_container_width=True, hide_index=True)
st.scatter_chart(df, x='ipl_m', y='scout', size='balls', color='slot')   # quality vs visibility
st.caption('ov_s / pv_s = shrunk runs-above-expected per ball (all / pressure balls). Verify capped status & availability manually.')
