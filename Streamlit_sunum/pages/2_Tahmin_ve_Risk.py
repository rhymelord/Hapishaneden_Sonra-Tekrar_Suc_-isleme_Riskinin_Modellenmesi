# pages/2_Tahmin_ve_Risk.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import load_data, train_or_load_model, predict_single

st.set_page_config(page_title="Tahmin & Risk", page_icon="🎯", layout="wide")
st.title("🎯 Tahmin & Risk")
# Varsayılan "Pages" menüsünü gizle
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}  /* default nav kapalı */
</style>
""", unsafe_allow_html=True)

# Kendi menünü oluştur
with st.sidebar:
    st.header("Menü")
    st.page_link("main.py", label="🏠 Ana Sayfa")  # ana dosyan hâlâ main.py ise
    st.page_link("pages/1_Profil_Analizi.py", label="🔎 Profil Analizi")
    st.page_link("pages/2_Tahmin_ve_Risk.py", label="🎯 Tahmin & Risk")
    st.page_link("pages/3_Tavsiye_Sistemi.py", label="🧩 Tavsiye Sistemi")
    st.page_link("pages/4_Rehabilitasyon_Senaryo_Simulatoru.py", label="🛠️ Senaryo Simülatörü")

# --- Sabit Eşik (override: secrets/ENV) ---
DEFAULT_THRESHOLD = 0.50  # değişmez eşik

def get_threshold():
    # Öncelik: Streamlit secrets → ENV → DEFAULT
    try:
        import streamlit as st
        if "THRESHOLD" in st.secrets:
            return float(st.secrets["THRESHOLD"])
    except Exception:
        pass
    import os
    return float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

threshold = get_threshold()

# --- Sayı kutularındaki +/- spin butonlarını gizle ---
st.markdown("""
<style>
/* Chrome, Safari, Edge, Opera */
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
/* Firefox */
input[type=number] { -moz-appearance: textfield; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def _load_df():
    return load_data()

@st.cache_resource(show_spinner=True)
def _load_model(_df):
    return train_or_load_model(_df)

# -------------------------
# Veri ve Model
# -------------------------
df = _load_df()
if df is None or df.empty:
    st.error("Veri yüklenemedi. Lütfen ana sayfada veri kaynağını doğrulayın.")
    st.stop()

bundle = _load_model(df)
model = bundle["model"]
cat_features = bundle.get("cat_features", [])
X_columns = bundle["X_columns"]

# --- Yardım ikonu ve tooltip fonksiyonu ---
def help_icon(text):
    return f"""
    <span style='position: relative; cursor: help; display:inline-block;'>
        ℹ️
        <span style="
            visibility: hidden;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
            width: 120px;
            font-size: 12px;
            ">
            {text}
        </span>
    </span>
    <style>
    span:hover span {{
        visibility: visible !important;
        opacity: 1 !important;
    }}
    </style>
    """

# Her bir kolon için tooltip açıklaması (örnek, kendi verine göre düzenleyebilirsin)
COLUMN_DESCRIPTIONS = {
    # Örnek kolon açıklamaları, kendi verine göre değiştir
    "feature1": "Bu özellik kişinin yaşını belirtir.",
    "feature2": "Bu özellik eğitim seviyesini gösterir.",
    "feature3": "Gelir durumu (bin TL cinsinden).",
    # Aşağıdaki boş bırakılanları kendine göre doldurabilirsin
    # ...
}

st.divider()
st.subheader("Tekil Tahmin")

mode = st.radio("Girdi yöntemi", ["ID Seç", "Manuel Giriş"], horizontal=True)

# -------------------------
# Yardımcılar
# -------------------------
def _is_binary(s: pd.Series) -> bool:
    try:
        return s.dropna().nunique() <= 2
    except Exception:
        return False

def _is_cat(s: pd.Series) -> bool:
    return s.dtype == "object" or str(s.dtype).startswith("category")

def _present_cols(cols, frame):
    return [c for c in cols if c in frame.columns]

# -------------------------
# Girdi Toplama
# -------------------------
input_row = {}

if mode == "ID Seç" and "ID" in df.columns:
    pick_id = st.selectbox("ID seç", options=df["ID"].tolist())
    base = df[df["ID"] == pick_id].drop(
        columns=[c for c in ["ID", "Training_Sample", "Recidivism_Within_3years"] if c in df.columns],
        errors="ignore"
    ).iloc[0]
    for c in X_columns:
        input_row[c] = base.get(c, np.nan)

else:
    st.info("İkili (0/1 veya 2 sınıf) alanlar aşağıda *radyo* ile, çok değerli sayısallar *tam sayı* olarak girilir.")

    present = _present_cols(X_columns, df)

    # Sınıflara göre ayır
    bin_num  = [c for c in present if pd.api.types.is_numeric_dtype(df[c]) and _is_binary(df[c])]
    bin_cat  = [c for c in present if _is_cat(df[c]) and _is_binary(df[c])]
    multi_num = [c for c in present if pd.api.types.is_numeric_dtype(df[c]) and not _is_binary(df[c])]
    multi_cat = [c for c in present if _is_cat(df[c]) and df[c].dropna().nunique() > 2]
    others    = [c for c in X_columns if c not in present]  # veride olmayan kolonlar

    # --- İkili Sayısal ---
    if bin_num:
        st.markdown("*İkili Sayısal (0/1) Alanlar*")
        cols = st.columns(4)
        for i, c in enumerate(bin_num):
            s = pd.to_numeric(df[c], errors="coerce")
            uniq = sorted(set([int(x) for x in s.dropna().unique().tolist() if x in [0, 1]])) or [0, 1]
            default = int(s.mode(dropna=True).iloc[0]) if not s.mode(dropna=True).empty else 0
            with cols[i % 4]:
                st.markdown(f"{c} {help_icon(COLUMN_DESCRIPTIONS.get(c, '0 veya 1 değeri seçiniz.'))}", unsafe_allow_html=True)
                choice = st.radio("", options=uniq, index=uniq.index(default) if default in uniq else 0, horizontal=True)
                input_row[c] = int(choice)

    # --- İkili Kategorik ---
    if bin_cat:
        st.markdown("*İkili Kategorik Alanlar*")
        cols = st.columns(4)
        for i, c in enumerate(bin_cat):
            s = df[c].dropna().astype(str)
            uniq_vals = sorted(s.unique().tolist())[:2]
            default = s.mode().iloc[0] if not s.mode().empty else (uniq_vals[0] if uniq_vals else "")
            with cols[i % 4]:
                st.markdown(f"{c} {help_icon(COLUMN_DESCRIPTIONS.get(c, 'İki kategoriden birini seçiniz.'))}", unsafe_allow_html=True)
                choice = st.radio("", options=uniq_vals if uniq_vals else [""],
                                  index=(uniq_vals.index(default) if default in uniq_vals else 0),
                                  horizontal=True)
                input_row[c] = choice

    # --- Çok Değerli Sayısal (tam sayı, spin yok) ---
    if multi_num:
        st.markdown("*Sayısal Özellikler*")
        cols = st.columns(3)
        for i, c in enumerate(multi_num):
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                dflt = int(np.round(np.nanmedian(s.values)))
                minv = int(np.floor(np.nanmin(s.values)))
                maxv = int(np.ceil(np.nanmax(s.values)))
            else:
                dflt, minv, maxv = 0, -10*6, 10*6  # veri yoksa geniş aralık
            if minv >= maxv:
                minv, maxv = minv - 100, maxv + 100
            with cols[i % 3]:
                st.markdown(f"{c} {help_icon(COLUMN_DESCRIPTIONS.get(c, 'Tam sayı değeri giriniz.'))}", unsafe_allow_html=True)
                input_row[c] = st.number_input(
                    "", value=int(dflt), min_value=int(minv),
                    max_value=int(maxv), step=1, format="%d"
                )

    # --- Çok Sınıflı Kategorik ---
    if multi_cat:
        st.markdown("*Kategorik Özellikler*")
        cols = st.columns(3)
        for i, c in enumerate(multi_cat):
            s = df[c].dropna().astype(str)
            uniq = s.unique().tolist()
            default = s.mode().iloc[0] if not s.mode().empty else (uniq[0] if uniq else "")
            with cols[i % 3]:
                st.markdown(f"{c} {help_icon(COLUMN_DESCRIPTIONS.get(c, 'Listeden bir değer seçiniz.'))}", unsafe_allow_html=True)
                input_row[c] = st.selectbox("", options=uniq if uniq else [""],
                                            index=(uniq.index(default) if default in uniq else 0))

    # --- Diğer (veride olmayan kolonlar) ---
    if others:
        st.markdown("*Diğer (veride bulunmayan kolonlar)*")
        cols = st.columns(3)
        for i, c in enumerate(others):
            with cols[i % 3]:
                st.markdown(f"{c} {help_icon(COLUMN_DESCRIPTIONS.get(c, 'Değeri manuel olarak giriniz.'))}", unsafe_allow_html=True)
                val = st.text_input("", value="")
                try:
                    input_row[c] = float(val) if val != "" else np.nan
                except Exception:
                    input_row[c] = val if val != "" else np.nan

# -------------------------
# Tahmin
# -------------------------
if st.button("Tahmin Et", type="primary"):
    try:
        result = predict_single(model, cat_features, X_columns, input_row)
        proba = float(result["proba"])
        pred  = int(proba >= threshold)

        # Risk bandı
        left, right = st.columns(2)
        with left:
            st.metric("Olasılık (1 sınıfı)", f"{proba:.3f}")
            st.metric("Tahmin (0/1)", str(pred))
        with right:
            if   proba < 0.33: band = "Düşük Risk"
            elif proba < 0.66: band = "Orta Risk"
            else:              band = "Yüksek Risk"
            st.metric("Risk Bandı", band)

        st.markdown("##### En Etkili Özellikler (örnek bazlı, SHAP)")
        contrib = result.get("top_contrib", {})
        if contrib:
            top_df = pd.DataFrame({"Özellik": list(contrib.keys()), "Katkı": list(contrib.values())})
            st.dataframe(top_df, use_container_width=True)
        else:
            st.caption("Not: SHAP katkıları hesaplanamadı; CatBoost sürümü/ortamı sınırlı olabilir.")

    except Exception as e:
        st.error(f"Tahmin sırasında hata: {e}")

st.divider()
st.subheader("Toplu Tahmin (CSV Yükle)")

file = st.file_uploader("Aynı kolon yapısında CSV yükle (başlık satırı gerekli)", type=["csv"])
if file is not None:
    try:
        batch_df = pd.read_csv(file)

        # Kolon hizalama: eksikler NaN eklenir, sıraya sokulur
        for c in X_columns:
            if c not in batch_df.columns:
                batch_df[c] = np.nan
        batch_df = batch_df[X_columns]

        # Toplu tahmin
        batch_result = model.predict_proba(batch_df)[:, 1]
        batch_pred = (batch_result >= threshold).astype(int)
        batch_df["Tahmin_Proba"] = batch_result
        batch_df["Tahmin_Sonuc"] = batch_pred

        st.success(f"Toplu tahmin başarıyla tamamlandı, {len(batch_df)} kayıt işlendi.")
        st.dataframe(batch_df.head(10))

        # İndir butonu
        st.download_button("Sonucu CSV olarak indir", batch_df.to_csv(index=False), "tahmin_sonuclari.csv")

    except Exception as e:
        st.error(f"Dosya işleme sırasında hata: {e}")
