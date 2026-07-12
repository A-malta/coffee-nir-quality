import pandas as pd
import os

def filter_results():
    bayesian_file = 'resultados_bayesian_search_treinamento.csv'
    validation_file = 'resultados_validacao_final.csv'
    
    if not os.path.exists(bayesian_file) or not os.path.exists(validation_file):
        print(f"Error: One or both files not found: {bayesian_file}, {validation_file}")
        return
    try:
        df_bayesian = pd.read_csv(bayesian_file)
        df_validation = pd.read_csv(validation_file)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    print(f"Original Bayesian rows: {len(df_bayesian)}")
    print(f"Validation rows: {len(df_validation)}")
    valid_models = set(df_validation['model'].tolist())
    
    df_filtered = df_bayesian[df_bayesian['model_file'].isin(valid_models)]
    
    print(f"Filtered Bayesian rows: {len(df_filtered)}")
    
    df_filtered.to_csv(bayesian_file, index=False)
    print(f"Successfully saved filtered results to {bayesian_file}")

if __name__ == "__main__":
    filter_results()
