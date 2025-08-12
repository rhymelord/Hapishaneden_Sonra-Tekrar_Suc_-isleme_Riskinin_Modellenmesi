# pages/1_Profil_Analizi.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.model_utils import load_data, cohort_describe, TARGET_COL

st.set_page_config(page_title="Profil Analizi", page_icon="🔎", layout="wide")
st.title("🔎 Profil Analizi")
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


@st.cache_data(show_spinner=False)
def _load():
    return load_data()

df = _load()

# ---------- Yardımcılar ----------
def is_id_col(name: str) -> bool:
    n = name.lower()
    # ID, _id, id_, ... gibi varyasyonları ve "identifier" benzeri adları yakala
    return (
        n == "id"
        or n.endswith("_id")
        or n.startswith("id_")
        or " id" in n
        or "id " in n
        or "identifier" in n
    )

def is_binary_series(s: pd.Series) -> bool:
    # Boşları at, benzersiz değer sayısı ≤2 ise ikili kabul et
    try:
        vals = pd.unique(s.dropna())
        return len(vals) <= 2
    except Exception:
        return False

# ---------- Üst Özet ----------
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### Özet İstatistikler (Sayısal)")
    desc = cohort_describe(df)
    st.dataframe(desc, use_container_width=True)

with col2:
    st.markdown("### Hızlı Bilgiler")
    st.metric("Toplam Kayıt", len(df))
    if TARGET_COL in df.columns:
        try:
            st.metric("Pozitif Oranı", f"{pd.to_numeric(df[TARGET_COL], errors='coerce').mean()*100:.1f}%")
        except Exception:
            pass

st.divider()
st.subheader("Filtrele & Dağılım")

# ---------- Uygun kolonları hazırla (ikili ve ID'leri çıkar) ----------
all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

eligible_num_cols = []
for c in num_cols:
    if is_id_col(c):
        continue
    if is_binary_series(df[c]):   # 0/1 vb. iki sınıflı sayısallar hariç
        continue
    eligible_num_cols.append(c)

if not eligible_num_cols and num_cols:
    st.info("İkili ve ID olmayan sayısal sütun bulunamadı. Mevcut sayısallar ikili olabilir.")
    eligible_num_cols = [c for c in num_cols if not is_id_col(c)]

sel_col = st.selectbox(
    "Histogram için değişken seç",
    options=eligible_num_cols if eligible_num_cols else all_cols
)

bins = st.slider("Bin sayısı", 5, 60, 25)

# Grafik
fig, ax = plt.subplots()
series = pd.to_numeric(df[sel_col], errors="coerce") if sel_col in df.columns else pd.Series(dtype=float)
ax.hist(series.dropna(), bins=bins)
ax.set_title(f"{sel_col} Dağılımı")
ax.set_xlabel(sel_col); ax.set_ylabel("Frekans")
st.pyplot(fig, clear_figure=True)

st.divider()
st.subheader("Birey Karşılaştırma")

# Mevcut mantığı koruyup sadece kıyaslamadan ikili ve ID sütunlarını çıkarıyoruz
id_col = "ID" if "ID" in df.columns else None
selected_id = None
if id_col:
    selected_id = st.selectbox("Kayıt (ID) seç", options=df[id_col].head(5000).tolist())

if selected_id is not None:
    row = df[df[id_col] == selected_id].iloc[0]
    st.markdown("##### Seçilen Kayıt Özeti")
    st.json(row.to_dict())

    st.markdown("##### Yüzdesel Karşılaştırma (Sayısal)")
    num = df.select_dtypes(include=[np.number]).copy()

    # ID benzeri ve ikili sütunları çıkar
    drop_cols = []
    for c in num.columns:
        if is_id_col(c) or is_binary_series(num[c]):
            drop_cols.append(c)
    num = num.drop(columns=drop_cols, errors="ignore")

    comps = {}
    for c in num.columns:
        val = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        col_vals = pd.to_numeric(num[c], errors="coerce")
        col_vals = col_vals.dropna()
        if pd.notna(val) and len(col_vals) > 0:
            comps[c] = (col_vals <= val).mean() * 100

    if comps:
        comps_df = pd.DataFrame({"Persentil(%)": comps}).sort_values("Persentil(%)")
        st.dataframe(comps_df, use_container_width=True)
    else:
        st.info("Kıyaslama için uygun (ikili/ID olmayan) sayısal sütun bulunamadı.")
else:
    st.info("ID kolonu yoksa veya seçim yapmadıysan üstteki dağılımları kullanabilirsin.")


