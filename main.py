import os
import pandas as pd
import sys

from processing.prepare_dataset import prepare_dataset
from model.train import train_models
from model.predict import run_monte_carlo
import config

def main():
    print("=====================================================")
    print("      FIFA WORLD CUP 2026 PREDICTOR & SIMULATOR      ")
    print("=====================================================")
    
    # 1. Run dataset preparation (Fase 1: Data Gathering)
    df_features = prepare_dataset()
    
    # 2. Train/Update the models (Fase 2: Model Training)
    train_models()
    
    # 3. Run Monte Carlo Simulation of the Tournament (Fase 2: Prediction)
    sim_runs = 10000
    df_res = run_monte_carlo(n_simulations=sim_runs)
    
    print("\n" + "=" * 80)
    print(f"               TOP 15 SELEÇÕES FAVORITAS AO TÍTULO (Em {sim_runs} Simulações)               ")
    print("=" * 80)
    print(f"{'Seleção':<22} | {'Classif. Grupo':<15} | {'Oitavas (R16)':<14} | {'Finais reached':<14} | {'Campeão (%)':<10}")
    print("-" * 80)
    
    top_favorites = df_res.sort_values(by="Winner", ascending=False).head(15)
    for team, row in top_favorites.iterrows():
        print(f"{team:<22} | {row['Qualified Group Stage']*100:>13.1f}% | {row['Reached Round of 16']*100:>12.1f}% | {row['Reached Final']*100:>12.1f}% | {row['Winner']*100:>9.1f}%")
    print("=" * 80)
    
    # Group-by-group qualification probability
    print("\n" + "=" * 80)
    print("                  PROBABILIDADE DE CLASSIFICAÇÃO POR GRUPO                  ")
    print("=" * 80)
    
    # Merge simulation results back with group names
    df_group_res = df_features[['team', 'group']].set_index('team').join(df_res)
    
    for group_name in sorted(df_group_res['group'].unique()):
        print(f"\n--- {group_name} ---")
        g_df = df_group_res[df_group_res['group'] == group_name].sort_values(by="Qualified Group Stage", ascending=False)
        print(f"{'País':<25} | {'Chance de Avançar (%)':<22} | {'Chance de Título (%)':<18}")
        print("-" * 75)
        for team, row in g_df.iterrows():
            print(f"{team:<25} | {row['Qualified Group Stage']*100:>19.1f}% | {row['Winner']*100:>16.1f}%")
            
    print("\n=========================================================================")
    print("Simulação concluída com sucesso! Os resultados foram gerados com base em:")
    print("1. Rankings oficiais da FIFA atualizados (Junho 2026)")
    print("2. Valor de mercado dos elencos extraídos do Transfermarkt")
    print("3. Índice de forma recente dos últimos 5 jogos oficiais das seleções")
    print("=========================================================================")

if __name__ == "__main__":
    main()

