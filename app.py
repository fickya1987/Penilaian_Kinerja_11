import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, skew
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Penilaian_Kinerja.csv")

df = load_data()
jabatan_col = 'Nama_Posisi'

skor_korporasi = df['Skor_KPI_Final'].mean()
mean_kpi = skor_korporasi
std_kpi = df['Skor_KPI_Final'].std()
skewness = skew(df['Skor_KPI_Final'])

kategori_labels = [
    ('Kurang', 0.0, 0.10, 'red'),
    ('Cukup', 0.10, 0.25, 'orange'),
    ('Baik', 0.25, 0.75, 'deepskyblue'),
    ('Sangat Baik', 0.75, 0.90, 'green'),
    ('Istimewa', 0.90, 1.0, 'gold')
]
label_props = {
    'Kurang':  {'color': 'red',        'fontsize': 12, 'weight': 'bold'},
    'Cukup':   {'color': 'orange',     'fontsize': 12, 'weight': 'bold'},
    'Baik':    {'color': 'deepskyblue','fontsize': 12, 'weight': 'bold'},
    'Sangat Baik': {'color': 'green',  'fontsize': 12, 'weight': 'bold'},
    'Istimewa':{'color': 'gold',       'fontsize': 12, 'weight': 'bold'},
}

def kategori_kpi(percentile):
    for label, low, high, _ in kategori_labels:
        if low <= percentile < high or (label == "Istimewa" and percentile == 1.0):
            return label
    return "Kurang"

# Penentuan kategori untuk semua pekerja
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

st.title("Kurva Distribusi Normal KPI Seluruh Pegawai (Korporasi/Pelindo)")
st.header("Kurva Distribusi Normal Skor KPI Pegawai Pelindo")

fig, ax = plt.subplots(figsize=(12, 4))
x = np.linspace(90, 110, 1000)
y = norm.pdf(x, mean_kpi, std_kpi)
ax.plot(x, y, color='black', linewidth=2, label='Kurva Normal')

# Area warna dan label kategori di bawah kurva (X axis)
for label, low, high, color in kategori_labels:
    x_fill = norm.ppf([low, high], mean_kpi, std_kpi)
    mask = (x >= x_fill[0]) & (x <= x_fill[1])
    ax.fill_between(x[mask], y[mask], alpha=0.25, color=color, label=label)
    # Letakkan label di bawah titik tengah area
    xpos = np.clip((x_fill[0] + x_fill[1]) / 2, 90, 110)
    ymin = y.min()
    props = label_props[label]
    ax.text(xpos, ymin - 0.01, label,
            ha='center', va='top', fontsize=props['fontsize'],
            color=props['color'], fontweight=props['weight'],
            zorder=10)

# Titik-titik NIPP pegawai
ax.scatter(df_komparasi['Skor_KPI_Final'], norm.pdf(df_komparasi['Skor_KPI_Final'], mean_kpi, std_kpi),
           color='grey', s=25, alpha=0.7, label="NIPP Pekerja (seluruh korporasi)")
for i, row in df_komparasi.iterrows():
    ax.text(row['Skor_KPI_Final'], norm.pdf(row['Skor_KPI_Final'], mean_kpi, std_kpi)+0.002,
            str(row['NIPP']), fontsize=7, ha='center', color='grey', alpha=0.7, rotation=90)

ax.set_xlim(90, 110)
ax.set_ylim(y.min() - 0.025, y.max() + 0.02)
ax.set_xlabel('Skor KPI')
ax.set_ylabel('Densitas')
ax.set_title('Kurva Distribusi Normal Skor KPI Pegawai Pelindo')
ax.legend(fontsize=9, loc='upper left')
plt.subplots_adjust(bottom=0.22)
st.pyplot(fig)

# (Opsional) Tabel kategori
st.header("Daftar Pekerja per Kategori Distribusi Normal KPI (Seluruh Korporasi)")
for label, _, _, _ in kategori_labels[::-1]:
    st.subheader(f"Kategori: {label}")
    df_kat = df_komparasi[df_komparasi['Kategori_Distribusi'] == label][['NIPP', 'Nama_Posisi', 'Skor_KPI_Final']]
    if df_kat.empty:
        st.write("Tidak ada.")
    else:
        st.dataframe(df_kat, hide_index=True)
