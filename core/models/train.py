from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline

def split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

def train_regressor(X, y, pre, test_size=0.2):
    X_train, X_test, y_train, y_test = split(X, y, test_size=test_size)
    reg = LinearRegression()
    model = Pipeline([("pre", pre), ("reg", reg)])
    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_classifier(X, y, pre, test_size=0.2):
    X_train, X_test, y_train, y_test = split(X, y, test_size=test_size)
    clf = LogisticRegression(max_iter=1000)
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)
    return model, X_test, y_test