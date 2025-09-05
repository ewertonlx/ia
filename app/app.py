import sys, os, pickle
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data.readcsv import read_csv
from core.data.database import (
    create_database_and_tables,
    insert_csv_to_sor,
    run_etl_sor_to_sot,
    run_etl_sot_to_spec_train,
    run_etl_for_test_data,
    load_data,
    drop_database
)
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_regressor, train_classifier
from core.models.predict import evaluate_classifier, evaluate_regressor
from core.explain.coefficients import extract_logit_importances, extract_linear_importances
from core.chatbot.rules import answer_from_metrics

st.set_page_config(page_title="🍔 OpenFoodFacts - CHATBOT", layout="wide")

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = False
if "prediction_df" not in st.session_state:
    st.session_state.prediction_df = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Oi, eu sou o bot do OpenFoodFacts. Envie seus dados para começarmos!"}
    ]
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "importances" not in st.session_state:
    st.session_state.importances = None

MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "off_model.pickle")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

st.title("🍔 OpenFoodFacts - CHATBOT")

with st.sidebar:
    st.header("1. Upload dos Dados")
    uploaded_files = st.file_uploader(
        "Envie 'Train.csv/TSV' e/ou 'Test.csv/TSV'",
        type=["csv", "tsv"],
        accept_multiple_files=True
    )

    st.header("2. Ações do Pipeline")

    st.subheader("Treinar Novo Modelo")
    task = st.selectbox("Escolha a tarefa", ["Classificação", "Regressão"])
    test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.4, 0.2, 0.05)

    if st.button("Executar Treinamento"):
        df_train = None
        for file in uploaded_files:
            if file.name.endswith((".csv", ".tsv")) and "test" not in file.name.lower():
                df_train = read_csv(file)
                print("Columns after reading CSV:", df_train.columns)

        if df_train is not None:
            with st.spinner("Treinando o modelo..."):
                create_database_and_tables()
                insert_csv_to_sor(df_train)
                run_etl_sor_to_sot()
                run_etl_sot_to_spec_train()

                df_spec_train = load_data("spec_food")
                target = "nutrition_score_fr_100g"
                if target not in df_spec_train.columns:
                    st.error(f"A coluna alvo '{target}' não foi encontrada.")
                    st.stop()

                if task == "Classificação":
                    y = (df_spec_train[target] > 10).astype(int)
                    X = df_spec_train.drop(columns=[target])
                    pre = make_preprocess_pipeline(X)
                    model, X_test, y_test = train_classifier(X, y, pre, test_size=test_size)
                    st.session_state.metrics, cm = evaluate_classifier(model, X_test, y_test)
                    st.session_state.importances = extract_logit_importances(model, X)
                else:
                    y = df_spec_train[target]
                    X = df_spec_train.drop(columns=[target])
                    pre = make_preprocess_pipeline(X)
                    model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)
                    st.session_state.metrics = evaluate_regressor(model, X_test, y_test)
                    st.session_state.importances = extract_linear_importances(model, X.columns, pre)

                with open(MODEL_PATH, "wb") as f:
                    pickle.dump(model, f)

                st.session_state.model_trained = True
                st.session_state.predictions_made = False
            st.success("✅ Modelo treinado e salvo com sucesso!")
        else:
            st.warning("Arquivo 'Train.csv' não encontrado.")

    st.subheader("Usar Modelo Existente")
    if st.button("Carregar Modelo e Fazer Previsões"):
        if not os.path.exists(MODEL_PATH):
            st.error("Nenhum modelo salvo encontrado. Treine um modelo primeiro.")
        else:
            df_test = None
            for file in uploaded_files:
                if file.name.endswith((".csv", ".tsv")) and "test" not in file.name.lower():
                    df_test = read_csv(file)

            if df_test is not None:
                with st.spinner("Carregando modelo e fazendo previsões..."):
                    run_etl_for_test_data(df_test)
                    df_spec_predict = load_data("spec_food")

                    with open(MODEL_PATH, "rb") as f:
                        model = pickle.load(f)

                    ids = df_spec_predict[["code"]] if "code" in df_spec_predict.columns else None
                    X_predict = df_spec_predict.drop(columns=["code"], errors="ignore")

                    predictions = model.predict(X_predict)

                    result_df = pd.DataFrame({"prediction": predictions})
                    if ids is not None:
                        result_df.insert(0, "code", ids)

                    st.session_state.prediction_df = result_df
                    st.session_state.predictions_made = True
                st.success("Previsões geradas com sucesso!")
            else:
                st.warning("Arquivo 'Test.csv' não encontrado.")

    st.header("3. Manutenção")
    if st.button("Limpar Tudo"):
        drop_database()
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.session_state.clear()
        st.info("Banco, modelo e sessão resetados.")
        st.rerun()

tab_train, tab_predict, tab_chat = st.tabs(["📊 Resultados do Treino", "🚀 Previsões", "💬 Chat"])

with tab_train:
    st.header("Resultados do Modelo")
    if not st.session_state.model_trained or st.session_state.metrics is None:
        st.info("Treine um modelo para ver resultados.")
    else:
        st.subheader("📈 Métricas")
        st.json(st.session_state.metrics)
        st.subheader("🔎 Importâncias")
        st.dataframe(st.session_state.importances.head(20), use_container_width=True)

with tab_predict:
    st.header("Previsões em Dados de Teste")
    if not st.session_state.predictions_made:
        st.info("Carregue um modelo e faça previsões na barra lateral.")
    else:
        st.dataframe(st.session_state.prediction_df)
        csv_data = convert_df_to_csv(st.session_state.prediction_df)
        st.download_button(
            label="⬇️ Baixar Previsões em CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
        )

with tab_chat:
    st.header("Converse com o Modelo")
    if not st.session_state.model_trained:
        st.info("Treine um modelo primeiro para poder conversar.")
    else:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Pergunte sobre métricas ou importâncias..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            response = answer_from_metrics(
                question=prompt,
                task=task,
                metrics_df_or_dict=st.session_state.metrics,
                importances_df=st.session_state.importances
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()
