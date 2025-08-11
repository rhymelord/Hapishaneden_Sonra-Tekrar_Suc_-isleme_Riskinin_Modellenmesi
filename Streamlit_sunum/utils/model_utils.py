# utils/model_utils.py
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier, Pool

TARGET_COL = "Recidivism_Within_3years"
DROP_COLS  = ["ID", "Training_Sample", TARGET_COL]

def _find_csv_path() -> str:
    # 1) Proje içi data klasörü, 2) Colab/Notebook vs. senaryoları için /mnt/data fallback
    candidates = [
        os.path.join("data", "NIJ_s_Recidivism_Full_Dataset.csv"),
        "/mnt/data/NIJ_s_Recidivism_Full_Dataset.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("NIJ_s_Recidivism_Full_Dataset.csv bulunamadı. 'data/' klasörünü kontrol et.")

def load_data() -> pd.DataFrame:
    path = _find_csv_path()
    df = pd.read_csv(path)
    # Bazı datasetlerde bool kolonlar int/0-1 gelebilir; dtype’ları normalize edelim
    # Kategorik gibi yorumlanacak bool ve object dtypes'ı toparlayacağız.
    return df

def _prep_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # bool -> str (CatBoost kategorik olarak ele alabilsin)
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(str)

    # object kolonlar kategorik sayılacak
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    return X, y, cat_features

def train_or_load_model(df: pd.DataFrame):
    """
    Basit: her çalıştırmada hızlıca eğit (küçük datasetlerde yeterli).
    İstersen modeli .cbm olarak kaydedip yüklemeye çevirebilirsin.
    """
    X, y, cat_features = _prep_features(df)
    if y is None:
        raise ValueError(f"'{TARGET_COL}' kolonu dataset içinde yok. Tahmin için gerekir.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = CatBoostClassifier(
        cat_features=cat_features,
        verbose=0,
        random_seed=42,
        loss_function="Logloss",
        eval_metric="AUC"
    )
    model.fit(X_train, y_train)
    # Kısa bir rapor döndürelim
    proba = model.predict_proba(X_test)[:, 1]
    y_hat = (proba >= 0.5).astype(int)
    report = classification_report(y_test, y_hat, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    return {
        "model": model,
        "cat_features": cat_features,
        "X_columns": X.columns.tolist(),
        "train_report": report,
        "auc": float(auc),
    }

def predict_single(model, cat_features: List[str], X_columns: List[str], row_dict: Dict) -> Dict:
    # Kullanıcının girdiği satırı DataFrame’e çevir
    X = pd.DataFrame([row_dict], columns=X_columns)
    # Eksik kolonları doldur
    for col in X_columns:
        if col not in X.columns:
            X[col] = np.nan
    # bool-string normalizasyonu
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(str)

    pool = Pool(X, cat_features=cat_features)
    proba = model.predict_proba(pool)[:, 1][0]
    pred = int(proba >= 0.5)

    # SHAP katkısı (opsiyonel; CatBoost destekliyor)
    try:
        shap_vals = model.get_feature_importance(pool, type="ShapValues")
        # Son sütun bias; ilk satır: 1 örnek
        contrib = pd.Series(shap_vals[0, :-1], index=X_columns).sort_values(key=lambda s: s.abs(), ascending=False)
        top_contrib = contrib.head(8).to_dict()
    except Exception:
        top_contrib = {}

    return {"proba": float(proba), "pred": pred, "top_contrib": top_contrib}

def cohort_describe(df: pd.DataFrame) -> pd.DataFrame:
    # Hızlı özet istatistik (sayısal)
    num = df.select_dtypes(include=[np.number])
    return num.describe().T
