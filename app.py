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

kategori_labels = [
    ('Kurang', 0.0, 0.10, 'red'),
    ('Cukup', 0.10, 0.25, 'orange'),
    ('Baik', 0.25, 0.75, 'skyblue'),
    ('Sangat Baik', 0.75, 0.90, 'green'),
    ('Istimewa', 0.90, 1.0, 'gold')
]

def kategori_kpi(percentile):
    for label, low, high, _ in kategori_labels:
        if low <= percentile < high or (label == "Istimewa" and percentile == 1.0):
            return label
    return "Kurang"

# Kategori dan gap seluruh pekerja
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

# --- KURVA GLOBAL (SELURUH KORPORASI) DENGAN LABEL KATEGORI DI X-AXIS ---
st.header("Kurva Distribusi Normal KPI Seluruh Pegawai (Korporasi/Pelindo)")
fig, ax = plt.subplots(figsize=(12, 4))
x = np.linspace(90, 110, 1000)
y = norm.pdf(x, mean_kpi, std_kpi)
ax.plot(x, y, color='black', linewidth=2, label='Kurva Normal')

# Area + label kategori
for label, low, high, color in kategori_labels:
    x_fill = norm.ppf([low, high], mean_kpi, std_kpi)
    mask = (x >= x_fill[0]) & (x <= x_fill[1])
    ax.fill_between(x[mask], y[mask], alpha=0.25, color=color, label=label)
    # Letakkan label di bawah batas bawah area
    xpos = (x_fill[0] + x_fill[1]) / 2
    if xpos < 90: xpos = 90
    if xpos > 110: xpos = 110
    ax.annotate(label, xy=(xpos, 0), xytext=(xpos, -0.002),
                textcoords="data", ha='center', va='top', fontsize=11,
                color=color, fontweight='bold', annotation_clip=False)

ax.scatter(df_komparasi['Skor_KPI_Final'], norm.pdf(df_komparasi['Skor_KPI_Final'], mean_kpi, std_kpi),
           color='grey', s=35, alpha=0.7, label="NIPP Pekerja (seluruh korporasi)")
for i, row in df_komparasi.iterrows():
    ax.text(row['Skor_KPI_Final'], norm.pdf(row['Skor_KPI_Final'], mean_kpi, std_kpi)+0.0005,
            str(row['NIPP']), fontsize=7, ha='center', color='grey', alpha=0.8, rotation=90)
ax.set_xlim(90, 110)
ax.set_xlabel('Skor KPI')
ax.set_ylabel('Densitas')
ax.set_title('Kurva Distribusi Normal Skor KPI Pegawai Pelindo')
ax.legend(fontsize=8)
ax.margins(y=0.15)
st.pyplot(fig)

# --- KURVA & TABEL PER ATASAN LANGSUNG (GROUP/DEPT) ---
st.header("Analisis Tiap Atasan Langsung (Group/Dept) - Kurva dan Tabel Kategori")
for nipp_atasan in df['NIPP_Atasan'].dropna().unique():
    if pd.isna(nipp_atasan) or nipp_atasan == '' or nipp_atasan not in df['NIPP_Pekerja'].values:
        continue
    jabatan_atasan = df[df['NIPP_Pekerja'] == nipp_atasan][jabatan_col].iloc[0]
    df_bawahan = df[df['NIPP_Atasan'] == nipp_atasan][['NIPP_Pekerja', jabatan_col, 'Skor_KPI_Final']]
    if df_bawahan.empty:
        continue
    mean_local = df_bawahan['Skor_KPI_Final'].mean()
    std_local = df_bawahan['Skor_KPI_Final'].std()
    # --- KURVA DISTRIBUSI + SCATTER NIPP + LABEL KATEGORI DI X ---
    fig, ax = plt.subplots(figsize=(12, 3))
    x = np.linspace(90, 110, 1000)
    y = norm.pdf(x, mean_local, std_local)
    ax.plot(x, y, color='black', linewidth=2, label='Kurva Normal (bawahan)')
    for label, low, high, color in kategori_labels:
        x_fill = norm.ppf([low, high], mean_local, std_local)
        mask = (x >= x_fill[0]) & (x <= x_fill[1])
        ax.fill_between(x[mask], y[mask], alpha=0.25, color=color, label=label)
        xpos = (x_fill[0] + x_fill[1]) / 2
        if xpos < 90: xpos = 90
        if xpos > 110: xpos = 110
        ax.annotate(label, xy=(xpos, 0), xytext=(xpos, -0.002),
                    textcoords="data", ha='center', va='top', fontsize=11,
                    color=color, fontweight='bold', annotation_clip=False)
    ax.scatter(df_bawahan['Skor_KPI_Final'], norm.pdf(df_bawahan['Skor_KPI_Final'], mean_local, std_local),
               color='grey', s=35, alpha=0.7, label="NIPP Pekerja")
    for i, row in df_bawahan.iterrows():
        ax.text(row['Skor_KPI_Final'], norm.pdf(row['Skor_KPI_Final'], mean_local, std_local)+0.0005,
                str(row['NIPP_Pekerja']), fontsize=7, ha='center', color='grey', alpha=0.8, rotation=90)
    ax.set_xlim(90, 110)
    ax.set_xlabel('Skor KPI')
    ax.set_ylabel('Densitas')
    ax.set_title(f"Bawahan dari Atasan: {jabatan_atasan}")
    ax.legend(fontsize=8)
    ax.margins(y=0.15)
    st.pyplot(fig)
    # --- TABEL KATEGORI BAWAHAN GROUP/DEPT TERSEBUT ---
    st.markdown(f"**Tabel Pekerja per Kategori untuk Bawahan dari Atasan: {jabatan_atasan}**")
    for kategori in [l[0] for l in kategori_labels[::-1]]:
        df_bawah_cat = df_komparasi[(df_komparasi['NIPP_Atasan'] == nipp_atasan) &
                                    (df_komparasi['Kategori_Distribusi'] == kategori)][['NIPP', 'Nama_Posisi', 'Skor_KPI_Final']]
        st.markdown(f"*Kategori: {kategori}*")
        if df_bawah_cat.empty:
            st.write("Tidak ada.")
        else:
            st.dataframe(df_bawah_cat, hide_index=True)
