import sqlite3
import pandas as pd
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_DIR = os.path.join(CURRENT_DIR, "sql")
APP_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DB_NAME = os.path.join(APP_DIR, "food.db")

def connect_db():
    return sqlite3.connect(DB_NAME)

def execute_sql_from_file(filepath):
    conn = connect_db()
    cursor = conn.cursor()
    with open(filepath, 'r') as f:
        sql_script = f.read()
    cursor.executescript(sql_script)
    conn.commit()
    conn.close()

def create_database_and_tables():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    
    sql_files = [
        os.path.join(SQL_DIR, "sor_food.sql"),
        os.path.join(SQL_DIR, "sot_food.sql"),
        os.path.join(SQL_DIR, "spec_food.sql"),
    ]

    for filepath in sql_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo SQL não encontrado: {filepath}")
        execute_sql_from_file(filepath)
    print("Banco de dados e tabelas criados com sucesso.")

def insert_csv_to_sor(df):
    conn = connect_db()

    df = df.loc[:, ~df.columns.duplicated()]

    target_col = "nutrition_score_fr_100g"
    if target_col not in df.columns:
        raise KeyError(f"A coluna '{target_col}' não existe. Colunas disponíveis: {df.columns.tolist()}")

    df_train = df[df[target_col].notna()]

    df_train.to_sql("sor_food", conn, if_exists="replace", index=False)
    conn.close()
    print("Dados de treino inseridos na tabela SOR.")


def run_etl_sor_to_sot():
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM sor_food", conn)
    
    df_sot = df.drop(columns=['code', 'product_name', 'collagen_meat_protein_ratio_100g', 'cocoa_100g', 'chlorophyl_100g', 'carbon_footprint_100g', 
                              'glycemic_index_100g', 'water_hardness_100g'], errors='ignore')
    df_sot.to_sql("sot_food", conn, if_exists="replace", index=False)
    conn.close()
    print("ETL de SOR para SOT (treino) concluído.")

def run_etl_sot_to_spec_train():
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM sot_food", conn)
    df.to_sql("spec_food", conn, if_exists="replace", index=False)
    conn.close()
    print("ETL de SOT para SPEC (treino) concluído.")

def run_etl_for_test_data(df_test):
    conn = connect_db()
    
    df_spec = df_test[['nutrition_score_fr_100g'] + [col for col in df_test.columns if col not in ['nutrition_score_fr_100g', 'code', 'product_name']]]

    df_spec.to_sql("spec_food", conn, if_exists="replace", index=False)
    conn.close()
    print("ETL para dados de teste concluído e salvo na SPEC (previsão).")

def load_data(table_name: str):
    conn = connect_db()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def drop_database():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"Banco de dados '{DB_NAME}' removido.")