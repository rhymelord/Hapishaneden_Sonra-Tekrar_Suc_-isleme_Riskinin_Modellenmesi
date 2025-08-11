# pages/4_Rehabilitasyon_Senaryo_Simulatoru.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import load_data, train_or_load_model, predict_single

st.set_page_config(page_title="Rehabilitasyon Senaryo Simülatörü", page_icon="🛠️", layout="wide")
st.title("🛠️ Rehabilitasyon Senaryo Simülatörü")

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

# --- Cache'li yükleme ---
@st.cache_data(show_spinner=False)
def _load_df():
    return load_data()

@st.cache_resource(show_spinner=True)
def _load_bundle(df):
    return train_or_load_model(df)

df = _load_df()
if df is None or df.empty:
    st.error("Veri yüklenemedi. Lütfen ana sayfadaki veri kaynağını doğrulayın.")
    st.stop()

bundle = _load_bundle(df)
model = bundle["model"]
cat_features = bundle.get("cat_features", [])
X_columns = bundle["X_columns"]

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

# Kolon türleri (genel bilgi)
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_all = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

st.caption("Temel profili seç ve aşağıdaki müdahaleleri ayarla. ‘Simülasyonu Çalıştır’ ile risk farkını gör.")
base_mode = st.radio("Girdi yöntemi", ["ID Seç", "Manuel Giriş"], horizontal=True)

# --- Temel profil (baseline) ---
input_row = {}
if base_mode == "ID Seç" and "ID" in df.columns:
    pick_id = st.selectbox("ID seç", options=df["ID"].tolist())
    base = df[df["ID"] == pick_id].drop(
        columns=[c for c in ["ID", "Training_Sample", "Recidivism_Within_3years"] if c in df.columns],
        errors="ignore"
    ).iloc[0]
    for c in X_columns:
        input_row[c] = base.get(c, np.nan)

else:
    st.info("İkili (0/1 veya 2 sınıf) alanlar **radyo**, çok değerli sayısallar **tam sayı**; çok sınıflı kategorikler **selectbox** ile girilir.", icon="📝")

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

    # --- Çok Değerli Sayısal (tam sayı) ---
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

# --- Baseline tahmin ---
base_res = predict_single(model, cat_features, X_columns, input_row)
p0 = float(base_res["proba"])
band0 = "Düşük" if p0 < 0.33 else ("Orta" if p0 < 0.66 else "Yüksek")

b1, b2, b3 = st.columns(3)
with b1: st.metric("Başlangıç Olasılığı (1)", f"{p0:.3f}")
with b2: st.metric("Başlangıç Risk", band0)
with b3: st.metric("Özellik Sayısı", str(len(X_columns)))

st.divider()
st.subheader("Müdahaleleri Tasarla")

st.caption("Aşağıda en fazla **3 müdahale** tanımlayabilirsin. Sayısalda: yüzde değişim veya sabit ekleme; kategorikte: yeni değer.")

def scenario_row_ui(idx: int):
    col = st.selectbox(f"[{idx}] Kolon", options=X_columns, key=f"col_{idx}")

    # Kolon tipi
    is_num = (col in num_cols_all)
    if is_num:
        mode = st.radio(f"[{idx}] İşlem", ["Yüzde Değişim (%)", "Sabit Ekle/Çıkar"], horizontal=True, key=f"mode_{idx}")
        if mode == "Yüzde Değişim (%)":
            pct = st.number_input(f"[{idx}] % değişim (örn. +20 veya -15)", value=0, step=1, format="%d", key=f"pct_{idx}")
            return {"col": col, "type": "num_pct", "val": int(pct)}
        else:
            add = st.number_input(f"[{idx}] Sabit ekle/çıkar (örn. +2 veya -1)", value=0, step=1, format="%d", key=f"add_{idx}")
            return {"col": col, "type": "num_add", "val": int(add)}
    else:
        # Kategorikler için mevcut değerlerden öneri
        vals = []
        if col in df.columns:
            try:
                vals = df[col].astype(str).dropna().value_counts().head(25).index.tolist()
            except Exception:
                vals = []
        new_val = st.selectbox(f"[{idx}] Yeni değer", options=(["(elle yaz)"] + vals) if vals else ["(elle yaz)"], key=f"cat_{idx}")
        if new_val == "(elle yaz)":
            new_val = st.text_input(f"[{idx}] Elle değer gir", key=f"cat_manual_{idx}")
        return {"col": col, "type": "cat_set", "val": new_val}

# 3 müdahale slotu
m1 = scenario_row_ui(1)
m2 = scenario_row_ui(2)
m3 = scenario_row_ui(3)
mods = [m for m in [m1, m2, m3] if m and m["col"]]

def apply_mods(base_row: dict, mods: list) -> dict:
    row = dict(base_row)  # kopya
    for m in mods:
        c, t, v = m["col"], m["type"], m["val"]
        if t == "num_pct":
            try:
                cur = float(row.get(c, np.nan))
                if not np.isnan(cur):
                    row[c] = cur * (1.0 + float(v)/100.0)
            except Exception:
                pass
        elif t == "num_add":
            try:
                cur = float(row.get(c, np.nan))
                if not np.isnan(cur):
                    row[c] = cur + float(v)
            except Exception:
                pass
        elif t == "cat_set":
            row[c] = str(v) if v is not None else row.get(c, np.nan)
    return row

if st.button("Simülasyonu Çalıştır", type="primary"):
    scen_row = apply_mods(input_row, mods)
    scen_res = predict_single(model, cat_features, X_columns, scen_row)
    p1 = float(scen_res["proba"])
    band1 = "Düşük" if p1 < 0.33 else ("Orta" if p1 < 0.66 else "Yüksek")

    # Farklar
    delta_pp = (p1 - p0) * 100.0               # yüzde puan farkı
    rel_change = ( (p1 - p0) / p0 * 100.0 ) if p0 > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Senaryo Olasılığı (1)", f"{p1:.3f}")
    with c2: st.metric("Senaryo Risk", band1)
    with c3:
        sign = "↓" if delta_pp < 0 else "↑" if delta_pp > 0 else "•"
        st.metric("Değişim (puan)", f"{delta_pp:+.2f} pp", delta=f"{sign} {abs(delta_pp):.2f} pp")

    # Policy tarzı kısa cümle
    bullets = []
    for m in mods:
        c, t, v = m["col"], m["type"], m["val"]
        if t == "num_pct":
            bullets.append(f"**{c}** %{v:+.0f} → risk {delta_pp:+.2f} puan")
        elif t == "num_add":
            bullets.append(f"**{c}** {v:+d} → risk {delta_pp:+.2f} puan")
        else:
            bullets.append(f"**{c}** = \"{v}\" → risk {delta_pp:+.2f} puan")

    st.subheader("Politika/Program Önerisi (Otomatik Özet)")
    if delta_pp < 0:
        st.success(" • " + " | ".join(bullets) + f"  ⇒ **göreli değişim**: {rel_change:+.1f}% (negatifse iyidir).")
    else:
        st.warning(" • " + " | ".join(bullets) + f"  ⇒ **göreli değişim**: {rel_change:+.1f}% (Pozitif artış risk yükseltiyor olabilir.)")

    # İsteğe bağlı: SHAP katkılarındaki değişim (bilgilendirici)
    base_contrib = base_res.get("top_contrib", {})
    scen_contrib = scen_res.get("top_contrib", {})
    if base_contrib and scen_contrib:
        st.markdown("##### En Etkili Özelliklerde Değişim (SHAP, ilk 8)")
        df_shap = pd.DataFrame({
            "Özellik": list(base_contrib.keys()),
            "Başlangıç Katkı": list(base_contrib.values())
        }).merge(
            pd.DataFrame({"Özellik": list(scen_contrib.keys()), "Senaryo Katkı": list(scen_contrib.values())}),
            on="Özellik", how="outer"
        ).fillna(0.0)
        df_shap["Fark (Senaryo - Başlangıç)"] = df_shap["Senaryo Katkı"] - df_shap["Başlangıç Katkı"]
        st.dataframe(
            df_shap.sort_values("Fark (Senaryo - Başlangıç)", key=lambda s: s.abs(), ascending=False),
            use_container_width=True
        )

    # Rapor indir (Markdown)
    md = [
        "# Rehabilitasyon Senaryo Raporu",
        f"- Başlangıç olasılığı: **{p0:.3f}** ({band0})",
        f"- Senaryo olasılığı: **{p1:.3f}** ({band1})",
        f"- Değişim: **{delta_pp:+.2f}** yüzde puan | Göreli: **{rel_change:+.1f}%**",
        "## Müdahaleler:",
    ] + [f"- {b}" for b in bullets]
    st.download_button(
        "Raporu indir (Markdown)",
        data="\n".join(md).encode("utf-8"),
        file_name="rehabilitasyon_senaryo_raporu.md",
        mime="text/markdown"
    )

st.caption("Not: Bu simülatör eğitim amaçlıdır. Gerçek dünyada müdahaleler etik, hukuki ve operasyonel gerekliliklerle birlikte değerlendirilmelidir.")
