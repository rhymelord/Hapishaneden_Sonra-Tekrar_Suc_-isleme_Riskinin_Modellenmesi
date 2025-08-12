# pages/3_Tavsiye_Sistemi.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import load_data, train_or_load_model, predict_single

st.set_page_config(page_title="Tavsiye Sistemi", page_icon="🧩", layout="wide")
st.title("🧩 Tavsiye Sistemi")
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

st.caption("""
**Uyarı (etik & adil kullanım):** Bu sayfa, eğitim/demonstrasyon amaçlı **kural tabanlı** öneriler üretir.
Gerçek hayatta karar destek sistemleri *adillik, önyargı azaltma, şeffaflık ve hukuki uygunluk* kriterleriyle değerlendirilmelidir.
""")

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
    st.error("Veri yüklenemedi. Lütfen ana sayfadaki veri kaynağını doğrulayın.")
    st.stop()

bundle = _load_model(df)
model = bundle["model"]
cat_features = bundle.get("cat_features", [])
X_columns = bundle["X_columns"]

# -------------------------
# Üst Bilgi Şeridi (AUC & Eşik)
# -------------------------
left, right = st.columns([2, 1])
with left:
    st.markdown("#### Model Özeti")
    auc_val = bundle.get("auc", None)
    st.write(f"**AUC:** {auc_val:.3f}" if auc_val is not None else "AUC: —")
with right:
    st.markdown("#### Eşik Değeri")
    threshold = st.slider("Sınıflandırma eşiği (proba ≥ eşik ⇒ 1)", 0.05, 0.95, 0.50, 0.01)

st.divider()
st.subheader("Öneri Girdileri")

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
mode = st.radio("Girdi yöntemi", ["ID Seç", "Manuel Giriş"], horizontal=True)
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
    st.info("İkili alanlar **radyo** ile, sayısallar **tam sayı** olarak; çok sınıflı kategorikler **selectbox** ile girilir.")

    present = _present_cols(X_columns, df)

    bin_num   = [c for c in present if pd.api.types.is_numeric_dtype(df[c]) and _is_binary(df[c])]
    bin_cat   = [c for c in present if _is_cat(df[c]) and _is_binary(df[c])]
    multi_num = [c for c in present if pd.api.types.is_numeric_dtype(df[c]) and not _is_binary(df[c])]
    multi_cat = [c for c in present if _is_cat(df[c]) and df[c].dropna().nunique() > 2]
    others    = [c for c in X_columns if c not in present]  # veride olmayan kolonlar

    # --- İkili Sayısal ---
    if bin_num:
        st.markdown("**İkili Sayısal (0/1) Alanlar**")
        cols = st.columns(4)
        for i, c in enumerate(bin_num):
            s = pd.to_numeric(df[c], errors="coerce")
            uniq = sorted(set([int(x) for x in s.dropna().unique().tolist() if x in [0, 1]])) or [0, 1]
            default = int(s.mode(dropna=True).iloc[0]) if not s.mode(dropna=True).empty else 0
            with cols[i % 4]:
                choice = st.radio(c, options=uniq, index=uniq.index(default) if default in uniq else 0, horizontal=True)
                input_row[c] = int(choice)

    # --- İkili Kategorik ---
    if bin_cat:
        st.markdown("**İkili Kategorik Alanlar**")
        cols = st.columns(4)
        for i, c in enumerate(bin_cat):
            s = df[c].dropna().astype(str)
            uniq_vals = sorted(s.unique().tolist())[:2]
            default = s.mode().iloc[0] if not s.mode().empty else (uniq_vals[0] if uniq_vals else "")
            with cols[i % 4]:
                choice = st.radio(
                    c,
                    options=uniq_vals if uniq_vals else [""],
                    index=(uniq_vals.index(default) if default in uniq_vals else 0),
                    horizontal=True
                )
                input_row[c] = choice

    # --- Çok Değerli Sayısal (tam sayı, spin yok) ---
    if multi_num:
        st.markdown("**Sayısal Özellikler**")
        cols = st.columns(3)
        for i, c in enumerate(multi_num):
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                dflt = int(np.round(np.nanmedian(s.values)))
                minv = int(np.floor(np.nanmin(s.values)))
                maxv = int(np.ceil(np.nanmax(s.values)))
            else:
                dflt, minv, maxv = 0, -10**6, 10**6
            if minv >= maxv:
                minv, maxv = minv - 100, maxv + 100
            with cols[i % 3]:
                input_row[c] = st.number_input(
                    c, value=int(dflt), min_value=int(minv), max_value=int(maxv), step=1, format="%d"
                )

    # --- Çok Sınıflı Kategorik ---
    if multi_cat:
        st.markdown("**Kategorik Özellikler**")
        cols = st.columns(3)
        for i, c in enumerate(multi_cat):
            s = df[c].dropna().astype(str)
            uniq = s.unique().tolist()
            default = s.mode().iloc[0] if not s.mode().empty else (uniq[0] if uniq else "")
            with cols[i % 3]:
                input_row[c] = st.selectbox(
                    c, options=uniq if uniq else [""], index=(uniq.index(default) if default in uniq else 0)
                )

    # --- Diğer (veride olmayan kolonlar) ---
    if others:
        st.markdown("**Diğer (veride bulunmayan kolonlar)**")
        cols = st.columns(3)
        for i, c in enumerate(others):
            with cols[i % 3]:
                val = st.text_input(c, value="")
                try:
                    input_row[c] = float(val) if val != "" else np.nan
                except Exception:
                    input_row[c] = val if val != "" else np.nan

# -------------------------
# Önerileri Oluştur
# -------------------------
if st.button("Önerileri Oluştur", type="primary"):
    try:
        res = predict_single(model, cat_features, X_columns, input_row)
        p = float(res["proba"])
        pred = int(p >= threshold)

        # Risk göstergeleri
        st.markdown("#### Risk Düzeyi")
        if   p < 0.33: band = "Düşük"
        elif p < 0.66: band = "Orta"
        else:          band = "Yüksek"

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Olasılık (1 sınıfı)", f"{p:.3f}")
        with c2: st.metric("Tahmin (0/1)", str(pred))
        with c3: st.metric("Risk", band)

        # Örnek bazlı katkılar (varsa)
        contrib = res.get("top_contrib", {})

        st.markdown("#### Kişiye Özel Öneriler (Kural Tabanlı)")
        tips = []

        # Veri sözlüğüne göre anahtar alanlara örnek kurallar
        key_edu = [c for c in X_columns if "Education" in c or "School" in c]
        key_job = [c for c in X_columns if "Employment" in c or "Job" in c or "Work" in c]

        if any(k in contrib for k in key_edu):
            tips.append("Eğitim programlarına yönlendirme (GED/HS tamamlama, mesleki sertifika).")
        if any(k in contrib for k in key_job):
            tips.append("İş bulma desteği: CV hazırlama, mesleki kurslar ve staj eşleştirmesi.")

        for k in contrib.keys():
            if "Age" in k:
                tips.append("Yaşa uygun mentorluk ve akran destek grupları.")
                break
        for k in contrib.keys():
            if "Prior" in k or "Arrest" in k or "Offense" in k:
                tips.append("Yoğun denetim, bilişsel-davranışçı programlar ve düzenli takip.")
                break

        if band == "Yüksek":
            tips += [
                "Daha sık temaslı denetim planı.",
                "Madde bağımlılığı değerlendirmesi ve gerekiyorsa tedavi yönlendirmesi.",
                "Bireysel ihtiyaç analizi ile kişiselleştirilmiş vaka yönetimi.",
            ]
        elif band == "Orta":
            tips += [
                "Gelişim odaklı atölyeler (iletişim, problem çözme).",
                "Topluluk temelli sosyal destek ağlarına katılım.",
            ]
        else:
            tips += ["Mevcut koruyucu faktörleri sürdürmeye yönelik hafif dokunuşlu takip."]

        st.success("Öneri listesi hazır:")
        for t in dict.fromkeys(tips):  # tekrarları temizle
            st.markdown(f"- {t}")

        st.markdown("##### En Etkili Özellikler (örnek bazlı, SHAP)")
        if contrib:
            top_df = pd.DataFrame({"Özellik": list(contrib.keys()), "Katkı": list(contrib.values())})
            st.dataframe(top_df, use_container_width=True)
        else:
            st.caption("Not: SHAP katkıları hesaplanamadı; CatBoost sürümü/ortamı sınırlı olabilir.")

    except Exception as e:
        st.error(f"Öneri oluşturma sırasında hata: {e}")

