import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np

from core.data.readcsv import read_csv
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_regressor, train_classifier
from core.models.predict import evaluate_classifier, evaluate_regressor
from core.explain.coefficients import extract_logit_importances, extract_linear_importances
from core.chatbot.rules import answer_from_metrics

st.set_page_config(page_title="OpenFoodFacts - CHATBOT", layout="wide")

for key in ["last_task", "last_metrics", "last_importances"]:
    st.session_state.setdefault(key, None)

st.title("OpenFoodFacts - CHATBOT 🍔")
st.info("Para começar, envie um arquivo CSV ou TSV.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Oi, eu sou o bot do OpenFoodFacts. Como posso ajudar?"}
    ]

with st.sidebar:
    st.header("Configurações ⚙️")
    task = st.selectbox("Escolha uma tarefa", ["Classificação", "Regressão"])
    test_size = st.slider("Escolha o tamanho do teste", 0.1, 0.9, 0.2, 0.1)
    uploaded_file = st.sidebar.file_uploader("Envie o CSV/TSV do Open Food Facts", type=["csv", "tsv"])
    
st.subheader("Chat")
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.text_input("Faça uma pergunta sobre o OpenFoodFacts", placeholder="Ex: Qual é o produto mais saudável?")
tab_train, tab_chat = st.tabs(["📊 Treino & Métricas", "💬 Chat"])
with tab_train:
    if uploaded_file:
        df = read_csv(uploaded_file)
        st.sidebar.success("Arquivo enviado com sucesso!")
        st.write("Prévia dos dados", df.head(1000))

        if task.startswith("Classificação"):
            target_column = "nutrition-score-fr_100g"
            if target_column not in df.columns:
                print(df.columns)
                st.error(f"A coluna alvo '{target_column}' não está presente no arquivo.")
                st.stop()

            y_binary = (df[target_column] > 10).astype(int)
            y = y_binary
            X = df.drop(columns=[target_column])

            st.write("Features: ", X.columns)
            st.write("Target Preview: ", y.head())
            pre = make_preprocess_pipeline(X)
            model, X_test, y_test = train_classifier(X, y, pre, test_size=test_size)

            metrics, cm = evaluate_classifier(model, X_test, y_test)
            st.subheader("Métricas de Classificação")
            st.json(metrics)

            st.subheader("🧮 Matriz de Confusão")
            cm_arr = np.array(cm)
            idx = ["Verdadeiro 0", "Verdadeiro 1"][:cm_arr.shape[0]]
            cols = ["Predito 0", "Predito 1"][:cm_arr.shape[1]]
            df_cm = pd.DataFrame(cm_arr, index=idx, columns=cols)
            st.dataframe(df_cm, use_container_width=True)

            importances = extract_logit_importances(model, X.columns, pre)
            st.subheader("🔎 Importâncias (Logistic Coef / Odds Ratio)")
            st.dataframe(importances.head(20), use_container_width=True)

            st.session_state.last_task = task
            st.session_state.last_metrics = metrics
            st.session_state.last_importances = importances

            if question:
                ans = answer_from_metrics(question, task, metrics, importances)
                st.session_state["messages"].append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                st.session_state["messages"].append({"role": "assistant", "content": ans})
                with st.chat_message("assistant"):
                    st.markdown(ans)

        else:
            target = "nutrition-score-fr_100g"
            if target not in df.columns:
                st.error(f"Coluna alvo '{target}' não encontrada no CSV.")
                st.stop()

            y = df[target]
            X = df.drop(columns=[target])

            pre = make_preprocess_pipeline(X)
            model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)

            metrics = evaluate_regressor(model, X_test, y_test)
            st.subheader("📈 Métricas (Regressão)")
            st.json(metrics)

            importances = extract_linear_importances(model, X.columns, pre)
            st.subheader("🔎 Importâncias (Coeficientes normalizados)")
            st.dataframe(importances.head(20), use_container_width=True)

            st.session_state.last_task = task
            st.session_state.last_metrics = metrics
            st.session_state.last_importances = importances

            if question:
                ans = answer_from_metrics(question, task, metrics, importances)
                st.info(ans)
                st.session_state["messages"].append({"role": "user", "content": question})
                st.session_state["messages"].append({"role": "assistant", "content": ans})
    else:
        st.info("⬆️ Envie um CSV do OpenFoodFacts na barra lateral para começar.")

with tab_chat:
    st.caption("Converse com o assistente sobre as métricas e importâncias do último treino.")
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Faça sua pergunta (ex.: Quais variáveis mais importam?)")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        task_ctx = st.session_state.get("last_task")
        metrics_ctx = st.session_state.get("last_metrics")
        importances_ctx = st.session_state.get("last_importances")

        if task_ctx and metrics_ctx is not None and importances_ctx is not None:
            ans = answer_from_metrics(prompt, task_ctx, metrics_ctx, importances_ctx)
        else:
            ans = "Ainda não há um modelo treinado nesta sessão. Vá em **📊 Treino & Métricas**, envie o CSV e treine o modelo primeiro."

        st.session_state["messages"].append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)