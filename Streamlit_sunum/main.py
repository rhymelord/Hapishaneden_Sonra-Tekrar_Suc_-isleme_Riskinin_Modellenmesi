# Ana_Sayfa.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.model_utils import load_data  # <— tekil veri yükleyici

# --------------------
# SAYFA AYARLARI
# --------------------
st.set_page_config(
    page_title="Recidivism Risk Studio | Ana Sayfa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
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


# --------------------
# CACHE'Lİ VERİ YÜKLEME
# --------------------
@st.cache_data(show_spinner=False)
def _load_df():
    # load_data(return_path=True) destekliyse yolu da al, değilse geriye dönük çalış
    try:
        df, used = load_data(return_path=True)
    except TypeError:
        df, used = load_data(), None
    return df, used

df, used_path = _load_df()

# --------------------
# YARDIMCI FONKSİYONLAR
# --------------------
def safe_mean(s: pd.Series):
    return pd.to_numeric(s, errors="coerce").dropna().mean()

def safe_unique(s: pd.Series | None):
    return int(s.nunique()) if s is not None else 0

def render_card(col, value, label, emoji, color="#0d47a1"):
    card_style = f"""
        background-color: {color}1A;
        border-radius: 14px;
        padding: 1.1rem .9rem;
        text-align: center;
        box-shadow: 0 6px 15px rgb(3 155 229 / 0.25);
        min-height: 100px; display:flex; flex-direction:column; justify-content:center;
    """
    number_style = f"font-size: 1.9rem; font-weight: 800; color: {color};"
    label_style  = f"font-size: 1.05rem; color: {color}; font-weight:700; margin-top:.15rem;"
    col.markdown(
        f"""
        <div style="{card_style}">
          <div style="{number_style}">{value}</div>
          <div style="{label_style}">{emoji} {label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------
# ÜST BAŞLIK + AÇIKLAMA
# --------------------
st.title("Hapishaneden Sonra: Tekrar Suç İşleme Riskinin Modellenmesi")
st.caption("NIJ Recidivism veri seti ile makine öğrenmesi tabanlı risk analiz stüdyosu")

st.markdown(
    """
    <div style="
        background-color: #0d1b2a; color: white; padding: 1.6rem 2rem;
        border-radius: 14px; box-shadow: 0 6px 15px rgba(0,0,0,0.35);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h2 style="margin: 0 0 .4rem 0;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h2>
        <h3 style="margin: .2rem 0; color:#90caf9;">Proje Amacı</h3>
        <p style="line-height:1.6; font-size:1.05rem;">
            Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme riskini (recidivism)
            veri bilimi ve makine öğrenmesi teknikleri ile tahmin etmeyi amaçlar.
            Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve risk analizi
            yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.
        </p>
        <h3 style="margin-top: 1.2rem; color:#90caf9;">Veri Seti Hakkında</h3>
        <p style="line-height:1.6; font-size:1.05rem;">
            Veri seti; demografik bilgiler, ceza süreleri, geçmiş suç kayıtları ve yeniden suç işleme
            bilgilerini içerir. Bu bilgiler risk faktörlerini analiz etmek ve modeller geliştirmek
            için zengin bir kaynak sunar.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Veri kaynağı yolu (varsa) göster
if used_path:
    st.caption(f"Kaynak: {used_path}")

st.divider()

# --------------------
# VERİ KONTROLÜ
# --------------------
if df is None or df.empty:
    st.error(
        "Veri yüklenemedi. Lütfen veriyi repo içindeki `data/` klasörüne koyun "
        "(örn. `data/NIJ_s_Recidivism_Full_Dataset.csv`) veya secrets/ENV ile `DATA_PATH` tanımlayın."
    )
    st.stop()

# --------------------
# KPI ve BİLGİ KARTLARI
# --------------------
# 'Hedef' metninin tam görünmesi için geniş 3. sütun
k1, k2, k3 = st.columns([1, 1, 2.6])

with k1:
    st.metric("Kayıt Sayısı", f"{df.shape[0]:,}")

with k2:
    st.metric("Özellik Sayısı", f"{df.shape[1]}")

with k3:
    st.markdown(
        """
        <div style="
            border: 1px solid #e5e7eb; border-radius: 12px; padding: .8rem 1rem;
            background: #f9fafb; box-shadow: 0 2px 8px rgba(0,0,0,.05);
        ">
          <div style="font-size:.95rem; color:#6b7280; margin-bottom:.2rem;">Hedef</div>
          <div style="font-size:1.1rem; font-weight:700; color:#111827; text-align:left;">
            3 yıl içinde yeniden suç işleme (0/1)
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("")

# Bilgi kartları (grid)
cards = []
cards.append(("Sütun Sayısı", df.shape[1], "📋", "#1976d2"))

if "Prison_Offense" in df.columns:
    n_offense = safe_unique(df["Prison_Offense"])
    if n_offense > 0:
        cards.append(("Farklı Suç Tipi", n_offense, "📌", "#0288d1"))

if "Sentence_Length_Months" in df.columns:
    avg_sentence = safe_mean(df["Sentence_Length_Months"])
    if pd.notna(avg_sentence):
        cards.append(("Ortalama Ceza Süresi (Ay)", f"{avg_sentence:.1f}", "⏳", "#388e3c"))

recid_col = next((c for c in df.columns if "recid" in c.lower()), None)
if recid_col:
    recid_rate = safe_mean(pd.to_numeric(df[recid_col], errors="coerce"))
    if pd.notna(recid_rate):
        cards.append(("Yeniden Suç İşleme Oranı", f"%{recid_rate*100:.1f}", "⚠️", "#d32f2f"))

if "Age_at_Release" in df.columns:
    avg_age = safe_mean(df["Age_at_Release"])
    if pd.notna(avg_age):
        cards.append(("Ortalama Tahliye Yaşı", f"{avg_age:.1f}", "👤", "#00695c"))

if "Education_Level" in df.columns:
    n_edu = safe_unique(df["Education_Level"])
    if n_edu > 0:
        cards.append(("Eğitim Seviyesi Sayısı", n_edu, "🎓", "#6a1b9a"))

if "Gender" in df.columns:
    n_gender = safe_unique(df["Gender"])
    if n_gender > 0:
        cards.append(("Cinsiyet Sayısı", n_gender, "🚻", "#5d4037"))

if cards:
    rows = (len(cards) + 3) // 4
    for r in range(rows):
        cols = st.columns(4, gap="small")
        for i in range(4):
            idx = r * 4 + i
            if idx >= len(cards):
                break
            label, val, emoji, color = cards[idx]
            render_card(cols[i], val, label, emoji, color)

st.divider()

# --------------------
# PASTA GRAFİĞİ – Recidivism oranı
# --------------------
st.subheader("🎯 Yeniden Suç İşleme Oranı (Pasta Grafiği)")
c1, c2 = st.columns([3, 1])
with c1:
    if recid_col and recid_col in df.columns:
        counts = pd.to_numeric(df[recid_col], errors="coerce").dropna().astype(int).value_counts().sort_index()
        labels = ["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"]
        values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
        fig = px.pie(
            names=labels, values=values,
            title="3 Yıl İçinde Yeniden Suç İşleme Oranı",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0, 0.1])
        fig.update_layout(title_x=0.5, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Yeniden suç işleme verisi bulunamadı (hedef kolon adında 'recid' geçmeli).")
with c2:
    st.markdown(
        "ℹ️ Bu pasta grafik, tahliye sonrası yeniden suç işleme durumunu yüzdesel olarak gösterir. "
        "'Tekrar Suç İşledi' dilimi öne çıkarılmıştır."
    )







