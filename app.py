import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from processing.prepare_dataset import prepare_dataset
from model.train import train_models
from model.predict import run_monte_carlo, predict_match, get_match_lambda, simulate_group_stage, simulate_knockout_stage

# 1. Page Configuration & CSS Injection
st.set_page_config(
    page_title="Preditor da Copa do Mundo 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Background and Style */
    .stApp {
        background: radial-gradient(circle at top right, #171E30 0%, #0B0F19 100%);
    }
    
    /* Header Gradient styling */
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        text-shadow: 0px 4px 20px rgba(0, 242, 254, 0.15);
    }
    
    /* Custom card container */
    .premium-card {
        background: rgba(23, 30, 48, 0.65);
        border: 1px solid rgba(0, 242, 254, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .premium-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.25);
    }
    
    /* Highlights */
    .cyan-text {
        color: #00F2FE;
        font-weight: 600;
    }
    
    .green-text {
        color: #10B981;
        font-weight: 600;
    }
    
    .red-text {
        color: #EF4444;
        font-weight: 600;
    }
    
    /* Stage Pill/Header */
    .stage-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #94A3B8;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        padding-bottom: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
</style>
""", unsafe_allow_html=True)

# 2. Cached Pipeline Functions
@st.cache_data
def get_clean_features():
    """Checks and prepares the final dataset."""
    if not os.path.exists(config.FINAL_DATASET_PATH):
        prepare_dataset()
    return pd.read_csv(config.FINAL_DATASET_PATH)

@st.cache_resource
def check_and_train_models():
    """Trains ML models if not present."""
    rf_path = os.path.join(config.DATA_DIR, "rf_model.pkl")
    if not os.path.exists(rf_path):
        train_models()

@st.cache_data
def get_simulation_results(n_sims):
    """Runs Monte Carlo simulations and caches results."""
    # Ensure models are trained
    check_and_train_models()
    return run_monte_carlo(n_simulations=n_sims)

# Load data
df_teams = get_clean_features()
team_features = df_teams.set_index('team').to_dict(orient='index')

# 3. Sidebar setup
st.sidebar.markdown("<h2 style='color: #00F2FE; font-weight: 800;'>⚙️ Configurações</h2>", unsafe_allow_html=True)
st.sidebar.write("Configure os parâmetros da simulação Monte Carlo:")
n_simulations = st.sidebar.slider("Número de Simulações", min_value=100, max_value=2000, value=500, step=100)

if st.sidebar.button("🔄 Recalcular Simulações"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("💡 **Como o modelo prevê?**")
st.sidebar.write("O modelo utiliza uma **Floresta Aleatória** para inferir importâncias e regressores de **Poisson** para computar a taxa de gols de cada equipe em campo neutro.")

# 4. Main Header
st.markdown("<p class='header-title'>⚽ Simulador Inteligente - Copa do Mundo 2026</p>", unsafe_allow_html=True)
st.markdown("<p style='color: #94A3B8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px;'>Previsões geradas por Machine Learning baseado no Ranking FIFA, Valor do Elenco, Forma Recente, EA FC26 e Desempenho em Eliminatórias.</p>", unsafe_allow_html=True)

# Fetch simulation outcomes
with st.spinner("Processando simulações Monte Carlo..."):
    df_sim_results = get_simulation_results(n_simulations)

# Setup Tabs
tab_favs, tab_bracket, tab_match, tab_data = st.tabs([
    "🏆 Favoritos & Grupos", 
    "🌿 Simulador do Torneio", 
    "⚔️ Simulador de Confrontos", 
    "📁 Banco de Dados"
])

# ----------------- TAB 1: FAVORITOS & GRUPOS -----------------
with tab_favs:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0;'>Top 15 Favoritos ao Título</h3>", unsafe_allow_html=True)
        st.write("Probabilidade média acumulada de se consagrar campeão mundial:")
        
        # Sort and filter top 15
        top_15 = df_sim_results.sort_values(by="Winner", ascending=False).head(15)
        
        # Altair chart for styling
        import altair as alt
        chart_df = top_15["Winner"].reset_index().rename(columns={"index": "Seleção", "Winner": "Chance"})
        chart_df["Chance (%)"] = round(chart_df["Chance"] * 100, 1)
        
        chart = alt.Chart(chart_df).mark_bar(
            cornerRadiusBottomRight=4,
            cornerRadiusTopRight=4,
            color='#00F2FE'
        ).encode(
            x=alt.X('Chance (%)', title='Probabilidade de Título (%)'),
            y=alt.Y('Seleção', sort='-x', title=''),
            tooltip=['Seleção', 'Chance (%)']
        ).properties(height=450)
        
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_right:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0;'>Probabilidade de Classificação nos Grupos</h3>", unsafe_allow_html=True)
        
        selected_group = st.selectbox("Selecione o Grupo:", sorted(df_teams['group'].unique()))
        
        # Merge simulation results back with group names
        df_group_res = df_teams[['team', 'group']].set_index('team').join(df_sim_results)
        g_df = df_group_res[df_group_res['group'] == selected_group].sort_values(by="Qualified Group Stage", ascending=False)
        
        # Plot Group Standings Probabilities
        g_chart_df = g_df["Qualified Group Stage"].reset_index().rename(columns={"team": "Seleção", "Qualified Group Stage": "Chance"})
        g_chart_df["Chance (%)"] = round(g_chart_df["Chance"] * 100, 1)
        
        g_chart = alt.Chart(g_chart_df).mark_bar(
            cornerRadiusBottomRight=4,
            cornerRadiusTopRight=4,
            color='#10B981'
        ).encode(
            x=alt.X('Chance (%)', title='Chance de Avançar (%)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('Seleção', sort='-x', title=''),
            tooltip=['Seleção', 'Chance (%)']
        ).properties(height=200)
        
        st.altair_chart(g_chart, use_container_width=True)
        
        # Display small info table
        st.write("**Chances Detalhadas no Grupo:**")
        disp_df = g_df[["Qualified Group Stage", "Winner"]].copy()
        disp_df["Qualified Group Stage"] = disp_df["Qualified Group Stage"].apply(lambda x: f"{x*100:.1f}%")
        disp_df["Winner"] = disp_df["Winner"].apply(lambda x: f"{x*100:.1f}%")
        disp_df.columns = ["Avançar da Fase de Grupos", "Chance de Título"]
        st.table(disp_df)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: BRACKET SIMULATOR -----------------
with tab_bracket:
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Simulador do Chaveamento da Copa</h3>", unsafe_allow_html=True)
    st.write("Clique no botão abaixo para rodar uma simulação dinâmica **única** do torneio e ver o mata-mata se desenrolar:")
    
    if st.button("🏁 Rodar Nova Simulação da Copa"):
        # Run one full tournament simulation
        with open(config.GROUPS_JSON_PATH, 'r') as f:
            groups = json.load(f)
            
        winners, runners, thirds = simulate_group_stage(groups)
        stages = simulate_knockout_stage(winners, runners, thirds)
        
        # Group Advanced teams
        adv_teams = set(winners + runners + thirds)
        
        # Filter stages
        r32 = [t for t, stg in stages.items() if stg == "Round of 32"]
        r16 = [t for t, stg in stages.items() if stg == "Round of 16"]
        qf = [t for t, stg in stages.items() if stg == "Quarter-Finals"]
        sf = [t for t, stg in stages.items() if stg == "Semi-Finals"]
        runner_up = [t for t, stg in stages.items() if stg == "Runner-Up"][0]
        champion = [t for t, stg in stages.items() if stg == "Champion"][0]
        
        # Visual columns
        col_group, col_r32, col_r16, col_qf, col_sf, col_final = st.columns(6)
        
        with col_group:
            st.markdown("<p class='stage-header'>Classificados (32)</p>", unsafe_allow_html=True)
            for t in sorted(adv_teams):
                st.write(f"🟢 {t}")
                
        with col_r32:
            st.markdown("<p class='stage-header'>Oitavas (R16) (16)</p>", unsafe_allow_html=True)
            # The teams that survived R32 (advanced to R16)
            survivors_r32 = list(adv_teams - set(r32))
            for t in sorted(survivors_r32):
                st.write(f"🏃‍♂️ {t}")
                
        with col_r16:
            st.markdown("<p class='stage-header'>Quartas (QF) (8)</p>", unsafe_allow_html=True)
            survivors_r16 = list(adv_teams - set(r32) - set(r16))
            for t in sorted(survivors_r16):
                st.write(f"🏅 {t}")
                
        with col_qf:
            st.markdown("<p class='stage-header'>Semifinais (SF) (4)</p>", unsafe_allow_html=True)
            survivors_qf = list(adv_teams - set(r32) - set(r16) - set(qf))
            for t in sorted(survivors_qf):
                st.write(f"⚡ {t}")
                
        with col_sf:
            st.markdown("<p class='stage-header'>Finalistas (2)</p>", unsafe_allow_html=True)
            st.write(f"🥈 {runner_up}")
            st.write(f"🥇 {champion}")
            
        with col_final:
            st.markdown("<p class='stage-header'>🏆 Campeão</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #00F2FE;'>👑 {champion}</h3>", unsafe_allow_html=True)
            st.balloons()
            
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: MATCH SIMULATOR -----------------
with tab_match:
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Simulador de Partida Personalizada</h3>", unsafe_allow_html=True)
    st.write("Escolha duas seleções para se enfrentarem com base nos modelos de Machine Learning:")
    
    col_sel_A, col_vs, col_sel_B = st.columns([2, 1, 2])
    
    with col_sel_A:
        team_a = st.selectbox("Seleção A:", sorted(config.QUALIFIED_TEAMS), index=8) # Brazil default
    with col_vs:
        st.markdown("<h2 style='text-align: center; margin-top: 15px;'>VS</h2>", unsafe_allow_html=True)
    with col_sel_B:
        team_b = st.selectbox("Seleção B:", sorted(config.QUALIFIED_TEAMS), index=32) # France default
        
    if team_a == team_b:
        st.warning("Selecione duas equipes diferentes!")
    else:
        # Display team metrics side-by-side
        col_stats_A, col_labels, col_stats_B = st.columns([2, 1, 2])
        
        feat_a = team_features[team_a]
        feat_b = team_features[team_b]
        
        with col_stats_A:
            st.markdown(f"<h4 style='text-align: right; color: #00F2FE;'>{team_a}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: right;'>Rank FIFA: <b>#{feat_a['fifa_rank']}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: right;'>Valor Elenco: <b>€{feat_a['squad_value_m']:.1f}M</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: right;'>FC 26 Rating: <b>{feat_a['fc26_rating']}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: right;'>Eliminatórias PPG: <b>{feat_a['qualifiers_ppg']:.2f}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: right;'>Forma Recente: <b>{feat_a['recent_results']}</b></p>", unsafe_allow_html=True)
            
        with col_labels:
            st.markdown("<p style='text-align: center; color:#94A3B8; font-weight:600; margin-top: 25px;'>PARÂMETRO</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color:#94A3B8;'>Rank FIFA</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color:#94A3B8;'>Valor de Mercado</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color:#94A3B8;'>Rating EA FC 26</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color:#94A3B8;'>Pontos p/ Jogo</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color:#94A3B8;'>Últimos 5 jogos</p>", unsafe_allow_html=True)
            
        with col_stats_B:
            st.markdown(f"<h4 style='color: #4FACFE;'>{team_b}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>Rank FIFA: <b>#{feat_b['fifa_rank']}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Valor Elenco: <b>€{feat_b['squad_value_m']:.1f}M</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p>FC 26 Rating: <b>{feat_b['fc26_rating']}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Eliminatórias PPG: <b>{feat_b['qualifiers_ppg']:.2f}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Forma Recente: <b>{feat_b['recent_results']}</b></p>", unsafe_allow_html=True)

        # Expected Goals (Lambdas)
        lambda_a, lambda_b = get_match_lambda(team_a, team_b)
        
        st.markdown("<p class='stage-header'>Métrica Estimada de Gols</p>", unsafe_allow_html=True)
        st.write(f"Média esperada de gols (neutralizado): **{team_a}** {lambda_a:.2f} vs {lambda_b:.2f} **{team_b}**")
        
        # Match Outcome Distributions using a local RNG to avoid polluting the global random state
        rng = np.random.default_rng(42)
        n_match_runs = 2000
        goals_a_sim = rng.poisson(lambda_a, n_match_runs)
        goals_b_sim = rng.poisson(lambda_b, n_match_runs)
        
        wins_a = np.sum(goals_a_sim > goals_b_sim)
        draws = np.sum(goals_a_sim == goals_b_sim)
        wins_b = np.sum(goals_b_sim > goals_a_sim)
        
        # Plot outcomes
        col_prob_A, col_prob_D, col_prob_B = st.columns(3)
        with col_prob_A:
            st.metric(f"Vitória de {team_a}", f"{wins_a / n_match_runs * 100:.1f}%")
        with col_prob_D:
            st.metric("Empate", f"{draws / n_match_runs * 100:.1f}%")
        with col_prob_B:
            st.metric(f"Vitória de {team_b}", f"{wins_b / n_match_runs * 100:.1f}%")
            
        # Score distribution
        scores = {}
        for i in range(n_match_runs):
            score = f"{goals_a_sim[i]}-{goals_b_sim[i]}"
            scores[score] = scores.get(score, 0) + 1
            
        score_df = pd.DataFrame.from_dict(scores, orient='index', columns=['Ocorrências']).reset_index().rename(columns={"index": "Placar"})
        score_df['Frequência (%)'] = round(score_df['Ocorrências'] / n_match_runs * 100, 1)
        score_df = score_df.sort_values(by='Ocorrências', ascending=False).head(10)
        
        st.markdown("<p class='stage-header'>Placares mais Prováveis (%)</p>", unsafe_allow_html=True)
        score_chart = alt.Chart(score_df).mark_bar(color='#4FACFE').encode(
            x=alt.X('Frequência (%)', title='Frequência (%)'),
            y=alt.Y('Placar', sort='-x', title='Placar (A-B)'),
            tooltip=['Placar', 'Frequência (%)']
        ).properties(height=300)
        
        st.altair_chart(score_chart, use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 4: DATABASE -----------------
with tab_data:
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Explorar Base de Dados da Copa 2026</h3>", unsafe_allow_html=True)
    st.write("Veja e ordene todas as seleções classificadas com suas respectivas métricas reunidas:")
    
    # Format and display DataFrame
    formatted_df = df_teams.copy()
    formatted_df = formatted_df.rename(columns={
        "team": "Seleção",
        "group": "Grupo",
        "fifa_rank": "Rank FIFA",
        "fifa_points": "Pontos FIFA",
        "squad_value_m": "Valor Elenco (€M)",
        "recent_results": "Resultados Recentes",
        "recent_form_index": "Índice Forma",
        "fc26_rating": "Rating FC 26",
        "qualifiers_ppg": "PPG Eliminatórias"
    })
    
    st.dataframe(formatted_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
