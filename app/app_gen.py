# app_off_rag.py — OpenFoodFacts + RAG (resumo estatístico como contexto)
import os
import sys
import pickle
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
# Ajuste do path caso necessário (mantive seu estilo)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data.readcsv import read_csv
from core.data.database import (
    create_database_and_tables,
    insert_csv_to_sor,
    run_etl_sor_to_sot,
    run_etl_sot_to_spec_train,
    run_etl_for_test_data,
    load_data,
    drop_database,
)
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_regressor, train_classifier
from core.models.predict import evaluate_classifier, evaluate_regressor
from core.explain.coefficients import extract_logit_importances, extract_linear_importances
from core.chatbot.rules import answer_from_metrics

st.set_page_config(page_title="🍔 OpenFoodFacts — RAG Chatbot", layout="wide")

# ----------------------
# Session state defaults
# ----------------------
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = False
if "prediction_df" not in st.session_state:
    st.session_state.prediction_df = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Oi, eu sou o bot do OpenFoodFacts. Envie sua pergunta!"}
    ]
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "importances" not in st.session_state:
    st.session_state.importances = None
if "messages" not in st.session_state:
    # messages for OpenAI chat (system + history). We'll keep a separate chat history used for API calls.
    st.session_state.messages = []

# ----------------------
# Paths etc.
# ----------------------
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "off_model.pickle")

# ----------------------
# Helpers: API client
# ----------------------
def get_api_key():
    # Prioritize env var, fallback to Streamlit secrets
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets["openai_api_key"]
        except Exception:
            key = None
    return key

def get_client():
    k = get_api_key()
    if not k:
        st.error("Defina OPENAI_API_KEY (variável de ambiente) ou coloque openai_api_key em .streamlit/secrets.toml.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = k
    return OpenAI()

# ----------------------
# Helpers: data -> contexto
# ----------------------
def numeric_summary(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        return "(Sem colunas numéricas)"
    desc = df[num_cols].describe().T
    desc["median"] = df[num_cols].median()
    cols = ["count", "mean", "median", "std", "min", "max"]
    # limitar a visualização a 20 colunas para não inflar o contexto
    return desc[cols].head(20).to_string()

def categorical_summary(df: pd.DataFrame, top_k: int = 5) -> str:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) == 0:
        return "(Sem colunas categóricas)"
    lines = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(top_k)
        lines.append(f"Coluna: {c}\n{vc.to_string()}\n")
    return "\n".join(lines)

def correlation_with_target(df: pd.DataFrame, target_col: str = "nutrition_score_fr_100g", top_n: int = 10) -> str:
    if target_col not in df.columns:
        return f"(Coluna alvo '{target_col}' não encontrada nesta tabela.)"
    try:
        t = pd.to_numeric(df[target_col], errors="coerce")
        num_cols = df.select_dtypes(include="number").columns
        corrs = []
        for c in num_cols:
            if c == target_col:
                continue
            corr = t.corr(pd.to_numeric(df[c], errors="coerce"))
            if pd.notna(corr):
                corrs.append((c, corr))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        lines = [f"{c}: {v:.3f}" for c, v in corrs[:top_n]]
        return "\n".join(lines) if lines else "(Sem correlações calculáveis)"
    except Exception as e:
        return f"(Erro ao calcular correlações: {e})"

def build_context(df: pd.DataFrame, max_chars: int = 4000, target_col: str = "nutrition_score_fr_100g") -> str:
    parts = []
    parts.append(f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas")
    parts.append("\n[Resumo numérico]\n" + numeric_summary(df))
    parts.append("\n[Resumo categórico]\n" + categorical_summary(df))
    parts.append(f"\n[Correlação com '{target_col}']\n" + correlation_with_target(df, target_col))
    # Amostra: primeiras 5 linhas (texto)
    try:
        sample_text = df.head(5).to_string()
        parts.append("\n[Exemplo - 5 primeiras linhas]\n" + sample_text)
    except Exception:
        pass
    ctx = "\n\n".join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n... (contexto truncado)"
    return ctx

# ----------------------
# Util: converter df para CSV (download)
# ----------------------
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")

# ----------------------
# Sidebar: upload / ações / RAG config
# ----------------------
with st.sidebar:
    st.header("1. Upload da Base de Dados")
    uploaded_files = st.file_uploader(
        "Envie en.openfoodfacts.org.products.tsv",
        type=["csv", "tsv"],
        accept_multiple_files=True,
    )

    st.header("2. Configurações Chat")
    max_ctx = st.slider("Limite do contexto (caracteres)", 500, 12000, 4000, step=500)
    show_ctx = st.checkbox("Mostrar contexto gerado", value=False)
    model_api = st.selectbox("Modelo (API)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    sys_prompt = st.text_area("System prompt (para o assistente)", value="Você é um analista de dados. Use o contexto para responder.", height=120)
    st.markdown("---")

    st.header("3. Ações")
    st.subheader("Treinar Novo Modelo")
    task = "Classificação"
    test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.4, 0.2, 0.05)

    if st.button("Treinar Modelo"):
        df_train = None
        for file in uploaded_files:
            if file.name.endswith((".csv", ".tsv")) and "test" not in file.name.lower():
                try:
                    df_train = read_csv(file)
                    break
                except Exception:
                    pass

        if df_train is None:
            st.warning("Arquivo en.openfoodfacts.org.products.tsv não encontrado.")
        else:
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

    st.subheader("Usar Modelo Existente")
    if st.button("Carregar Modelo e Fazer Previsões"):
        if not os.path.exists(MODEL_PATH):
            st.error("Nenhum modelo salvo encontrado. Treine um modelo primeiro.")
        else:
            df_test = None
            for file in uploaded_files:
                if file.name.endswith((".csv", ".tsv")) and "test" not in file.name.lower():
                    try:
                        df_test = read_csv(file)
                        break
                    except Exception:
                        pass

            if df_test is None:
                st.warning("Arquivo en.openfoodfacts.org.products.tsv não encontrado.")
            else:
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

    st.header("Excluir Dados")
    if st.button("Excluir Banco e Modelo e Resetar Sessão"):
        drop_database()
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.session_state.clear()
        st.info("Banco, modelo e sessão resetados.")
        st.rerun()

# ----------------------
# Layout: tabs
# ----------------------
tab_train, tab_predict, tab_chat = st.tabs(["📊 Resultados do Treino", "🚀 Previsões", "💬 Chat Food"])

with tab_train:
    st.header("Resultados do Modelo")
    if not st.session_state.model_trained or st.session_state.metrics is None:
        st.info("Treine um modelo para ver resultados.")
    else:
        st.subheader("📈 Métricas")
        st.json(st.session_state.metrics)
        st.subheader("🔎 Importâncias (top 20)")
        if st.session_state.importances is not None:
            try:
                st.dataframe(st.session_state.importances.head(20), use_container_width=True)
            except Exception:
                st.dataframe(st.session_state.importances, use_container_width=True)
        else:
            st.info("Sem importâncias disponíveis.")

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

# ----------------------
# Chat tab: RAG
# ----------------------
with tab_chat:
    st.header("Converse com o Modelo (RAG)")

    # Carrega a tabela final gerada no pipeline (spec_food) para usar como contexto
    df_context = None
    try:
        # se a tabela existir no DB, load_data("spec_food") trará a tabela usada em treino/predição
        df_context = load_data("spec_food")
    except Exception:
        df_context = None

    if df_context is None or df_context.empty:
        st.warning("Tabela 'spec_food' não encontrada. Treine/execute o pipeline ou faça upload/ETL para gerar a tabela final.")
        st.info("O chat ficará indisponível até que a tabela final exista.")
    else:
        # Constrói texto de contexto
        context_text = build_context(df_context, max_chars=max_ctx)

        if show_ctx:
            with st.expander("Ver contexto (resumo da base)"):
                st.text(context_text)

        # Inicializa mensagens do asistente (usadas apenas para display / API)
        # Mantemos um histórico local em st.session_state.chat_messages (exibição)
        if "messages" not in st.session_state or not st.session_state.get("messages"):
            st.session_state.messages = [{"role": "system", "content": sys_prompt}]

        # Render histórico (ignora system para exibição)
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # Input do usuário
        prompt = st.chat_input("Pergunte sobre métricas, importâncias ou a base (contexto incluído automaticamente)")
        if prompt:
            # adiciona ao histórico de exibição
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # Preparar mensagens para a API: system -> contexto (user) -> restante do histórico -> pergunta
            msgs = []
            msgs.append({"role": "system", "content": sys_prompt})
            # injetar contexto como mensagem do usuário logo após system
            msgs.append({"role": "user", "content": f"Contexto da base (resumo):\n{context_text}"})

            # opcionalmente, reenvia histórico de assistant/user (sem system) para manter coesão
            for m in st.session_state.chat_messages:
                # pulamos o primeiro system-style message que já injetamos
                if m["role"] in ("user", "assistant"):
                    msgs.append({"role": m["role"], "content": m["content"]})

            # faz chamada para API
            try:
                client = get_client()
                resp = client.chat.completions.create(
                    model=model_api,
                    messages=msgs,
                    temperature=0.2,
                )
                reply = resp.choices[0].message.content
            except Exception as e:
                reply = f"Erro na chamada à API: {e}"

            # exibir e armazenar resposta
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            # render da última mensagem assistant
            with st.chat_message("assistant"):
                st.markdown(reply)
            # opcional: persistir mensagens para próxima chamada
            st.session_state.messages = msgs

            # manter interface responsiva
            st.rerun()
