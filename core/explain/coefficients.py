import numpy as np
import pandas as pd

def _feature_names_from_preprocess(pre, original_cols):
    pre_step = pre.named_steps["pre"]
    num_cols = pre_step.transformers_[0][2]
    cat_cols = pre_step.transformers_[1][2]
    out = list(num_cols)
    ohe = pre_step.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    return out + cat_feature_names

def extract_logit_importances(model_pipe, X_train):
    pre = model_pipe.named_steps["pre"]
    X_transformed = pre.transform(X_train)
    
    if hasattr(pre, "get_feature_names_out"):
        feature_names = pre.get_feature_names_out(X_train.columns)
    else:
        feature_names = []
        for name, transformer, cols in pre.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
    
    clf = model_pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()
    odds = np.exp(coefs)
    
    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "odds_ratio": odds
    })
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")


def extract_linear_importances(model_pipe, original_cols, pre):
    reg = model_pipe.named_steps["reg"]
    feature_names = _feature_names_from_preprocess(model_pipe.named_steps["pre"], original_cols)
    coefs = reg.coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)})
    return df.sort_values("abs_coef", ascending=False)