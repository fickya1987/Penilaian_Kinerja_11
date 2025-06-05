import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, skew
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("Penilaian_Kinerja.csv")

df = load_data()
jabatan_col = 'Nama_Posisi'

skor_korporasi = df['Skor_KPI_Final'].mean()
mean_kpi = skor_korporasi
std_kpi = df['Skor_KPI_Final'].std()
skewness = skew(df['Skor_KPI_Final'])

def kategori_kpi(percentile):
    if percentile >= 0.9:
        return 'Istimewa'
    elif percentile >= 0.75:
        return 'Sangat Baik'
    elif percentile >= 0.25:
        return 'Baik'
    elif percentile >= 0.10:
        return 'Cukup'
    else:
        return 'Kurang'

# Kategori seluruh pekerja
hasil_komparasi = []
for idx, row in df.iterrows():
    nipp = row['NIPP_Pekerja']
    jabatan = row[jabatan_col] if jabatan_col in row else ""
    nipp_atasan = row['NIPP_Atasan']
    skor = row['Skor_KPI_Final']
    gap_vs_korporasi = (skor - skor_korporasi) / skor_korporasi
    if nipp_atasan in df['NIPP_Pekerja'].values:
        skor_atasan = df[df['NIPP_Pekerja'] == nipp_atasan]['Skor_KPI_Final'].values[0]
        gap_vs_atasan = (skor - skor_atasan) / skor_atasan
    else:
        skor_atasan = np.nan
        gap_vs_atasan = np.nan
    percentile = norm.cdf(skor, loc=mean_kpi, scale=std_kpi)
    kategori = kategori_kpi(percentile)
    hasil_komparasi.append({
        'NIPP': nipp,
        'Nama_Posisi': jabatan,
        'NIPP_Atasan': nipp_atasan,
        'Skor_KPI_Final': skor,
        'Skor_KPI_Atasan': skor_atasan,
        'Gap_vs_Atasan(%)': round(100*gap_vs_atasan, 2) if not np.isnan(gap_vs_atasan) else "",
        'Gap_vs_Korporasi(%)': round(100*gap_vs_korporasi, 2),
        'Kategori_Distribusi': kategori
    })

df_komparasi = pd.DataFrame(hasil_komparasi)

# --- BAR CHART GLOBAL (KORPORASI) ---
st.header("Distribusi Pegawai Berdasarkan Kategori KPI (Seluruh Korporasi)")

kategori_order = ['Kurang', 'Cukup', 'Baik', 'Sangat Baik', 'Istimewa']
distribusi_count = df_komparasi['Kategori_Distribusi'].value_counts().reindex(kategori_order, fill_value=0)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(kategori_order, distribusi_count.values, color=['red', 'orange', 'skyblue', 'green', 'gold'])
ax.set_xlabel('Kategori')
ax.set_ylabel('Jumlah Pegawai')
ax.set_title('Distribusi Pegawai Berdasarkan Kategori KPI')
ax.bar_label(bars, label_type='edge', fontsize=12)
st.pyplot(fig)

# --- TABEL PEGAWAI PER KATEGORI ---
st.header("Daftar Pegawai per Kategori Distribusi Normal KPI (Seluruh Korporasi)")
for kategori in kategori_order:
    st.subheader(f"Kategori: {kategori}")
    df_kat = df_komparasi[df_komparasi['Kategori_Distribusi'] == kategori][['NIPP', 'Nama_Posisi', 'Skor_KPI_Final']]
    if df_kat.empty:
        st.write("Tidak ada.")
    else:
        st.dataframe(df_kat, hide_index=True)

# --- Statistik ringkas & persentase ---
st.header("Statistik Distribusi Seluruh Pegawai")
st.markdown(f"- **Rata-rata (Korporasi/Pelindo):** {mean_kpi:.2f}")
st.markdown(f"- **Standard Deviasi:** {std_kpi:.2f}")
st.markdown(f"- **Skewness:** {skewness:.2f}")

st.markdown("### Sebaran Persentase Kategori Distribusi Normal")
distribusi_persen = df_komparasi['Kategori_Distribusi'].value_counts(normalize=True).reindex(
    kategori_order
).fillna(0) * 100
st.table(distribusi_persen.reset_index().rename(columns={'index':'Kategori','Kategori_Distribusi':'Persentase (%)'}))
