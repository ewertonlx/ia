def answer_from_metrics(question: str, task: str, metrics_df_or_dict, importances_df):
    q = (question or "").lower()

    if "importan" in q or "importân" in q or "variáve" in q or "features" in q:
        top = importances_df.head(5)[["feature"]].to_dict("records")
        top_str = ", ".join([t["feature"] for t in top])
        return f"As variáveis mais influentes são: {top_str}. (Baseado em coeficientes/odds ratio)"

    if "métric" in q or "score" in q or "acur" in q or "rmse" in q:
        return f"Métricas da tarefa {task}: {metrics_df_or_dict}"

    if "como foi treinado" in q or "pipeline" in q:
        return "O pipeline aplica imputação, one-hot e padronização; depois treina Logistic Regression (class.) ou Linear Regression (regr.)."

    if "code" in q or "lgpd" in q:
        return "No OpenFoodFacts - CHATBOT 🍔, evitamos dados sensíveis, anonimização por padrão e não persistimos dados pessoais. Para produção: consentimento expresso, minimização e auditoria."

    return "Posso falar sobre variáveis importantes, métricas do modelo e como o pipeline funciona. Pergunte algo como 'Quais variáveis mais importam?'."