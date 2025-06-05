import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Konfigurasi kategori & warna ---
kategori_labels = [
    ("Kurang", 0.00, 0.05, "#FFB3B3"),
    ("Cukup", 0.05, 0.10, "#FFE3A1"),
    ("Baik", 0.10, 0.85, "#B3E0FF"),
    ("Sangat Baik", 0.85, 0.95, "#B3FFB3"),
    ("Istimewa", 0.95, 1.00, "#FFFACD"),
]
kategori_colors = dict((k[0], k[3]) for k in kategori_labels)

# --- Fungsi pembagian kategori berdasarkan percentile ---
def kategori_from_score(score, scores):
    percentiles = np.percentile(scores, [5, 10, 85, 95])
    if score < percentiles[0]:
        return "Kurang"
    elif score < percentiles[1]:
        return "Cukup"
    elif score < percentiles[2]:
        return "Baik"
    elif score < percentiles[3]:
        return "Sangat Baik"
    else:
        return "Istimewa"

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Penilaian_Kinerja.csv")
    df = df.dropna(subset=['Skor_KPI_Final', 'NIPP_Pekerja', 'Nama_Posisi'])
    return df

df = load_data()

# Hitung kategori untuk seluruh pegawai
all_scores = df['Skor_KPI_Final'].values
df['Kategori_KPI'] = df['Skor_KPI_Final'].apply(lambda x: kategori_from_score(x, all_scores))

# --- Fungsi plot kurva distribusi normal dengan kategori sebagai x-axis ---
def plot_kurva(scores, nipp_list=None, title=None):
    mean_kpi = np.mean(scores)
    std_kpi = np.std(scores)
    x = np.linspace(90, 110, 1000)
    y = norm.pdf(x, mean_kpi, std_kpi)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12,4))
    # Warnai area kategori
    xticks = []
    labels = []
    for label, pmin, pmax, color in kategori_labels:
        xmin = np.clip(norm.ppf(pmin, mean_kpi, std_kpi), 90, 110)
        xmax = np.clip(norm.ppf(pmax, mean_kpi, std_kpi), 90, 110)
        ax.fill_between(x, 0, y, where=(x >= xmin) & (x <= xmax), color=color, alpha=0.4)
        # Posisi label di tengah area
        xmid = (xmin + xmax) / 2
        xticks.append(xmid)
        labels.append(label)

    # Kurva normal
    ax.plot(x, y, color='black', linewidth=3, label="Kurva Normal")
    # Titik pegawai
    scatter = ax.scatter(scores, np.zeros_like(scores), color='gray', s=40, alpha=0.6, label="NIPP Pekerja")
    if nipp_list is not None:
        # Tambah label NIPP di bawah titik
        for xi, nipp in zip(scores, nipp_list):
            ax.annotate(f"{int(nipp)}", (xi, 0), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7, color='gray', alpha=0.7)

    ax.set_xlim(90, 110)
    ax.set_xlabel("Kategori Penilaian", fontsize=13, fontweight='bold')
    ax.set_ylabel("Densitas", fontsize=13, fontweight='bold')
    if title: ax.set_title(title, fontsize=18, weight='bold')

    # Set x-ticks ke kategori & warna, fontsize kecil & rata
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    # Hilangkan ticks minor angka
    ax.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False)
    ax.legend()
    st.pyplot(fig)

# --- APP STREAMLIT ---
st.title("Kurva Distribusi Normal KPI Seluruh Pegawai (Korporasi/Pelindo)")

# Kurva seluruh pegawai
plot_kurva(df['Skor_KPI_Final'].values, nipp_list=df['NIPP_Pekerja'].values, 
           title="Kurva Distribusi Normal Skor KPI Pegawai Pelindo")

# Tabel Kategori Seluruh Pegawai
st.subheader("Tabel Pegawai berdasarkan Kategori Penilaian")
df_tabel = df[['NIPP_Pekerja','Nama_Posisi','Skor_KPI_Final','Kategori_KPI']].sort_values(by='Skor_KPI_Final', ascending=False)
st.dataframe(df_tabel, hide_index=True)

# --- Distribusi per Atasan Langsung (Group/Dept) ---
st.header("Kurva Distribusi Normal KPI untuk Tiap Atasan Langsung (Group/Dept)")
atasans = df['NIPP_Atasan'].dropna().unique()
selected_atasan = st.selectbox("Pilih NIPP Atasan untuk Melihat Distribusi Kelompok:", sorted(atasans))
df_group = df[df['NIPP_Atasan']==selected_atasan]

if len(df_group) > 2:
    plot_kurva(df_group['Skor_KPI_Final'].values, nipp_list=df_group['NIPP_Pekerja'].values, 
               title=f"Bawahan dari Atasan NIPP {selected_atasan} ({df_group.iloc[0]['Nama_Posisi']})")
    # Tabel kategori untuk group
    st.subheader("Tabel Pegawai dalam Group/Dept ini")
    df_group_tabel = df_group[['NIPP_Pekerja','Nama_Posisi','Skor_KPI_Final','Kategori_KPI']].sort_values(by='Skor_KPI_Final', ascending=False)
    st.dataframe(df_group_tabel, hide_index=True)
else:
    st.warning("Tidak cukup data untuk membuat distribusi normal di group ini.")

# Statistik ringkas
st.header("Statistik Ringkas Penilaian Seluruh Pegawai")
st.write(df['Kategori_KPI'].value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%"))

