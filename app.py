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
    'Kurang':  {'color': 'red',        'fontsize': 10, 'weight': 'bold'},
    'Cukup':   {'color': 'orange',     'fontsize': 10, 'weight': 'bold'},
    'Baik':    {'color': 'deepskyblue','fontsize': 10, 'weight': 'bold'},
    'Sangat Baik': {'color': 'green',  'fontsize': 10, 'weight': 'bold'},
    'Istimewa':{'color': 'gold',       'fontsize': 10, 'weight': 'bold'},
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

def plot_kurva(mean_kpi, std_kpi, df_poin, title, legend_label):
    used_std = std_kpi if (not np.isnan(std_kpi) and std_kpi > 0) else 0.2
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.linspace(90, 110, 1000)
    y = norm.pdf(x, mean_kpi, used_std)
    ax.plot(x, y, color='black', linewidth=2, label=legend_label)
    for label, low, high, color in kategori_labels:
        x_fill = norm.ppf([low, high], mean_kpi, used_std)
        mask = (x >= x_fill[0]) & (x <= x_fill[1])
        ax.fill_between(x[mask], y[mask], alpha=0.25, color=color, label=label)
        xpos = np.clip((x_fill[0] + x_fill[1]) / 2, 90.5, 109.5)
        props = label_props[label]
        # --- RAPIKAN LABEL: selalu letakkan DI BAWAH kurva, pakai y offset -0.03, kecil, tidak tumpuk sumbu
        ax.annotate(label, (xpos, y.min()-0.03), ha='center', va='top',
                    fontsize=props['fontsize'], color=props['color'],
                    fontweight=props['weight'], annotation_clip=False)

    # Titik NIPP pekerja
    ax.scatter(df_poin['Skor_KPI_Final'], norm.pdf(df_poin['Skor_KPI_Final'], mean_kpi, used_std),
               color='grey', s=20, alpha=0.6, label="NIPP Pekerja (seluruh korporasi)")
    for i, row in df_poin.iterrows():
        ax.text(row['Skor_KPI_Final'], norm.pdf(row['Skor_KPI_Final'], mean_kpi, used_std)+0.004,
                str(row['NIPP']), fontsize=6, ha='center', color='grey', alpha=0.7, rotation=90)

    ax.set_xlim(90, 110)
    ax.set_ylim(y.min() - 0.045, y.max() + 0.025)
    ax.set_xlabel('Skor KPI')
    ax.set_ylabel('Densitas')
    ax.set_title(title)
    ax.legend(fontsize=9, loc='upper left')
    plt.subplots_adjust(bottom=0.18)
    st.pyplot(fig)

# ---------- KURVA & TABEL SELURUH PEGAWAI ----------
st.title("Kurva Distribusi Normal KPI Seluruh Pegawai (Korporasi/Pelindo)")
st.header("Kurva Distribusi Normal Skor KPI Pegawai Pelindo")
plot_kurva(mean_kpi, std_kpi, df_komparasi, 'Kurva Distribusi Normal Skor KPI Pegawai Pelindo', 'Kurva Normal')

# Tabel kategori seluruh pegawai
st.header("Daftar Pekerja per Kategori Distribusi Normal KPI (Seluruh Korporasi)")
for label, _, _, _ in kategori_labels[::-1]:
    st.subheader(f"Kategori: {label}")
    df_kat = df_komparasi[df_komparasi['Kategori_Distribusi'] == label][['NIPP', 'Nama_Posisi', 'Skor_KPI_Final']]
    if df_kat.empty:
        st.write("Tidak ada.")
    else:
        st.dataframe(df_kat, hide_index=True)

# ---------- KURVA & TABEL PER ATASAN LANGSUNG (GROUP/DEPT) ----------
st.header("Kurva Distribusi Normal KPI untuk Tiap Atasan Langsung (Group/Dept)")
for nipp_atasan in df['NIPP_Atasan'].dropna().unique():
    if pd.isna(nipp_atasan) or nipp_atasan == '' or nipp_atasan not in df['NIPP_Pekerja'].values:
        continue
    jabatan_atasan = df[df['NIPP_Pekerja'] == nipp_atasan][jabatan_col].iloc[0]
    df_bawahan = df[df['NIPP_Atasan'] == nipp_atasan][['NIPP_Pekerja', jabatan_col, 'Skor_KPI_Final']]
    if df_bawahan.empty:
        continue
    mean_local = df_bawahan['Skor_KPI_Final'].mean()
    std_local = df_bawahan['Skor_KPI_Final'].std()
    # --- PENTING: kategori per group harus pakai norm group
    df_bawah_komp = df_komparasi[df_komparasi['NIPP'].isin(df_bawahan['NIPP_Pekerja'])].copy()
    plot_kurva(mean_local, std_local, df_bawah_komp,
               f"Bawahan dari Atasan: {jabatan_atasan} (NIPP {nipp_atasan})",
               'Kurva Normal (bawahan)')
    # Tabel kategori per atasan
    st.markdown(f"**Tabel Pekerja per Kategori untuk Bawahan dari Atasan: {jabatan_atasan} (NIPP {nipp_atasan})**")
    for kategori in [l[0] for l in kategori_labels[::-1]]:
        df_bawah_cat = df_bawah_komp[df_bawah_komp['Kategori_Distribusi'] == kategori][['NIPP', 'Nama_Posisi', 'Skor_KPI_Final']]
        st.markdown(f"*Kategori: {kategori}*")
        if df_bawah_cat.empty:
            st.write("Tidak ada.")
        else:
            st.dataframe(df_bawah_cat, hide_index=True)
