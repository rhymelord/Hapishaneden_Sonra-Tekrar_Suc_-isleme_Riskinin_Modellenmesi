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
# utils/var_help.py
from __future__ import annotations
from pathlib import Path
import re

# (Kısa ve sade) Türkçe açıklamalar — bildiklerimizi doldurduk,
# kalanları altta kurallarla tahmin edeceğiz.
VAR_TR: dict[str, str] = {
    "ID": "Kayıt kimliği (benzersiz).",
    "Training_Sample": "Model eğitimi için işaret (1=Eğitim, 0=Diğer/Doğrulama).",
    "Recidivism_Within_3years": "Tahliyeden sonraki 3 yıl içinde yeniden suç işleme (1=Evet, 0=Hayır).",
    "Age_at_Release": "Tahliye anındaki yaş (yıl).",
    "Sentence_Length_Months": "Verilen ceza süresi (ay).",
    "Time_Served_Months": "Cezaevinde geçirilen süre (ay).",
    "Education_Level": "En yüksek tamamlanan eğitim seviyesi.",
    "Employment_Status": "İstihdam durumu (tahliye öncesi/sonrası).",
    "Gender": "Cinsiyet.",
    "Prison_Offense": "Cezaevine girişe temel suç türü.",
    "Prior_Arrests": "Önceki gözaltı/tutuklama sayısı.",
    "Prior_Convictions": "Önceki mahkûmiyet sayısı.",
    "Prior_Commitments": "Önceki tutukluluk/kapalı kurum sayısı.",
    "Release_Type": "Tahliye türü (ör. koşullu, denetimli).",
    "Parole_Supervision": "Denetimli serbestlik/gözetim durumu.",
    "Supervision_Level": "Denetim yoğunluğu/seviyesi.",
    "Marital_Status": "Medeni durum.",
    "Num_Children": "Çocuk sayısı.",
    "Housing_Status": "Barınma durumu (kalıcı/kalıcı değil).",
    "Homeless": "Evsizlik göstergesi (1=Evet, 0=Hayır).",
    "Substance_Abuse_History": "Madde kullanım geçmişi (1/0 veya kategorik).",
    "Alcohol_Abuse_History": "Alkol kullanım/bağımlılık geçmişi.",
    "Mental_Health_Diagnosis": "Ruh sağlığı tanısı bilgisi (var/yok).",
    "Program_Participation": "Program/rehabilitasyon katılımı.",
    "Program_Completion": "Katıldığı programı tamamlama durumu.",
    "Risk_Score": "Önceden hesaplanmış risk skoru (varsa).",
    "County": "İlçe/County bilgisi.",
    "State": "Eyalet/il bilgisi.",
    "Zip": "Posta kodu.",
    # Gerekirse ekleyelim...
}

# Sözlükte yoksa kullanılacak KISA açıklama kuralları
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"recid", re.I), "Yeniden suç işleme ile ilgili hedef/sinyal."),
    (re.compile(r"\bage\b|\bya[sş]\b", re.I), "Yaş bilgisi (yıl)."),
    (re.compile(r"month|ay|_mo\b", re.I), "Ay cinsinden süre/değer."),
    (re.compile(r"prior|previous|onceki", re.I), "Geçmiş/önceki kayıt sayısı."),
    (re.compile(r"arrest|g[oö]zalt", re.I), "Gözaltı/tutuklama bilgisi."),
    (re.compile(r"convict|mahk[uû]m", re.I), "Mahkûmiyet bilgisi."),
    (re.compile(r"offen|su[cç]", re.I), "Suç türü/kategorisi."),
    (re.compile(r"educ|okul|school", re.I), "Eğitim seviyesi/bilgisi."),
    (re.compile(r"employ|job|work|istihdam", re.I), "İstihdam/çalışma durumu."),
    (re.compile(r"gender|sex|cinsiyet", re.I), "Cinsiyet bilgisi."),
    (re.compile(r"parole|probation|supervis", re.I), "Denetim/koşullu serbestlik bilgisi."),
    (re.compile(r"housing|home|evsiz", re.I), "Barınma/konut durumu."),
    (re.compile(r"alcohol|substance|drug|madde", re.I), "Madde/alkol kullanımı bilgisi."),
    (re.compile(r"mental|ruh", re.I), "Ruh sağlığına ilişkin bilgi."),
    (re.compile(r"risk", re.I), "Risk göstergesi/puanı."),
    (re.compile(r"state|county|city|il|ilce|zip|posta", re.I), "Coğrafi/konum bilgisi."),
]

def _guess(col: str) -> str:
    for pat, desc in _RULES:
        if pat.search(col):
            return desc
    if col.lower() in {"id", "_id"}:
        return "Kayıt kimliği (benzersiz)."
    return "Bu değişken için kısa açıklama henüz eklenmedi."

def help_of(col: str) -> str:
    """Widget'larda help=help_of(c) kullan."""
    return VAR_TR.get(col, _guess(col))

def export_dict_for(df_columns: list[str]) -> dict[str, str]:
    """İstersen df.columns içinden otomatik sözlük taslağı üretir."""
    out = {}
    for c in df_columns:
        out[c] = help_of(c)
    return out

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

