import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Analisis Perikanan Gurame",
    layout="wide"
)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("dataset_invest_juta.csv")

df = load_data()

# =====================
# SIDEBAR
# =====================
st.sidebar.title("🐟 Menu Aplikasi")

menu = st.sidebar.radio(
    "Navigasi",
    ["Dashboard", "EDA", "Model", "Prediksi", "Insight"]
)

st.sidebar.markdown("---")

kecamatan = st.sidebar.selectbox(
    "Filter Kecamatan",
    ["Semua"] + sorted(df["kemendagri_nama_kecamatan"].dropna().unique())
)

if kecamatan != "Semua":
    df = df[df["kemendagri_nama_kecamatan"] == kecamatan]

# =====================
# DATA & MODEL (GLOBAL)
# =====================
X = df[
    [
        "jumlah_pembudidaya",
        "invest_juta",
        "jumlah_proyek_perikanan",
        "jumlah_tenaga_kerja_perikanan",
    ]
]

y = df["jumlah_produksi_ikan_gurame"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

# KOEFISIEN (DIBUAT SEKALI)
coef_df = pd.DataFrame({
    "Variabel": X.columns,
    "Koefisien": model.coef_
})

# METRIK MODEL
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# =====================
# DASHBOARD
# =====================
if menu == "Dashboard":
    st.title("📊 Dashboard Produksi Ikan Gurame")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Produksi", f"{y.sum():,.0f}")
    col2.metric("Rata-rata Produksi", f"{y.mean():,.2f}")
    col3.metric("Total Investasi (Juta)", f"{df['invest_juta'].sum():,.0f}")

    st.subheader("Dataset")
    st.dataframe(df)

# =====================
# EDA
# =====================
elif menu == "EDA":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("Statistik Deskriptif")
    st.dataframe(X.describe())

    st.subheader("Distribusi Produksi Ikan Gurame")
    fig, ax = plt.subplots()
    ax.hist(y, bins=20)
    ax.set_xlabel("Produksi")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    st.subheader("Investasi vs Produksi")
    fig, ax = plt.subplots()
    ax.scatter(df["invest_juta"], y)
    ax.set_xlabel("Investasi (Juta)")
    ax.set_ylabel("Produksi")
    st.pyplot(fig)

    st.subheader("Produksi per Kecamatan")
    st.bar_chart(
        df.groupby("kemendagri_nama_kecamatan")["jumlah_produksi_ikan_gurame"].sum()
    )

# =====================
# MODEL
# =====================
elif menu == "Model":
    st.title("📐 Model Regresi Linier Berganda")

    st.subheader("Koefisien Model")
    st.table(coef_df)

    st.write(f"**Intercept:** {model.intercept_:.2f}")

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.2f}")

    st.subheader("Aktual vs Prediksi")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()])
    ax.set_xlabel("Produksi Aktual")
    ax.set_ylabel("Produksi Prediksi")
    st.pyplot(fig)

# =====================
# PREDIKSI
# =====================
elif menu == "Prediksi":
    st.title("🔮 Prediksi Produksi Ikan Gurame")

    pembudidaya = st.number_input("Jumlah Pembudidaya", min_value=0)
    invest = st.number_input("Investasi (Juta Rupiah)", min_value=0.0)
    proyek = st.number_input("Jumlah Proyek Perikanan", min_value=0)
    tenaga_kerja = st.number_input("Jumlah Tenaga Kerja Perikanan", min_value=0)

    if st.button("Prediksi Produksi"):
        pred = model.predict([[pembudidaya, invest, proyek, tenaga_kerja]])
        hasil = pred[0]

        st.success(f"Perkiraan Produksi: **{hasil:.2f}**")

        if hasil > y.mean():
            st.info("📈 Produksi diprediksi lebih tinggi dari rata-rata.")
        else:
            st.warning("📉 Produksi diprediksi lebih rendah dari rata-rata.")

# =====================
# INSIGHT (SUDAH FIX)
# =====================
elif menu == "Insight":
    st.title("📝 Insight & Kesimpulan")

    variabel_utama = coef_df.loc[
        coef_df["Koefisien"].abs().idxmax(), "Variabel"
    ]

    st.write(f"""
    Berdasarkan hasil analisis regresi linier berganda,
    variabel yang **paling berpengaruh** terhadap produksi ikan gurame adalah
    **{variabel_utama}**.

    Nilai **R² sebesar {r2:.3f}** menunjukkan bahwa model mampu
    menjelaskan sebagian besar variasi produksi ikan gurame.

    Model ini dapat dimanfaatkan sebagai alat bantu pengambilan keputusan
    dalam perencanaan investasi dan pengembangan sektor perikanan.
    """)

# =====================
# FOOTER
# =====================
st.markdown("---")
st.caption("📘 Aplikasi Analisis Produksi Ikan Gurame | Streamlit & Machine Learning")
