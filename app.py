import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------------------- Load Data -------------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("Penilaian_Kinerja.csv")
    df = df.dropna(subset=['Skor_KPI_Final', 'NIPP_Pekerja', 'Nama_Posisi'])
    return df

df = load_data()
df['Skor_KPI_Final'] = df['Skor_KPI_Final'].clip(lower=90, upper=110)

# -------------------- Fungsi Kategori -------------------- #
def kategori_from_score(x, all_scores):
    p = np.percentile(all_scores, [10, 25, 75, 90])
    if x <= p[0]: return 'Kurang'
    elif x <= p[1]: return 'Cukup'
    elif x <= p[2]: return 'Baik'
    elif x <= p[3]: return 'Sangat Baik'
    else: return 'Istimewa'

# -------------------- Fungsi Plot Kurva -------------------- #
def plot_kurva(scores, nipps, title):
    x_min, x_max = 90, 110
    mean_kpi, std_kpi = np.mean(scores), np.std(scores) if np.std(scores) > 0 else 1e-3
    x = np.linspace(x_min, x_max, 1000)
    y = norm.pdf(x, mean_kpi, std_kpi)

    # Kategori label & batas persentil
    batas_percentile = [0.0, 0.10, 0.25, 0.75, 0.90, 1.0]
    kategori_labels_order = ['Kurang', 'Cukup', 'Baik', 'Sangat Baik', 'Istimewa']
    kategori_colors = ['#ff9999', '#ffe4b2', '#b2e1ff', '#a6e3a1', '#ffe066']

    fig, ax = plt.subplots(figsize=(12, 4.2))
    # Area kategori warna
    for i in range(5):
        x_left = norm.ppf(batas_percentile[i], mean_kpi, std_kpi)
        x_right = norm.ppf(batas_percentile[i+1], mean_kpi, std_kpi)
        x_fill = x[(x >= x_left) & (x <= x_right)]
        y_fill = y[(x >= x_left) & (x <= x_right)]
        ax.fill_between(x_fill, 0, y_fill, color=kategori_colors[i], alpha=0.33)
    
    # Kurva normal
    ax.plot(x, y, color='black', lw=3, label='Kurva Normal')

    # Titik NIPP
    ymin = -0.025 * y.max()
    ax.scatter(scores, np.full_like(scores, ymin), color='grey', alpha=0.75, s=42, label='NIPP Pegawai')
    for sc, nipp in zip(scores, nipps):
        ax.annotate(str(int(nipp)), xy=(sc, ymin-0.01), fontsize=7, ha='center', rotation=90)

    # Label kategori di x-axis (tanpa angka)
    kategori_x_pos = []
    for i in range(5):
        x_left = norm.ppf(batas_percentile[i], mean_kpi, std_kpi)
        x_right = norm.ppf(batas_percentile[i+1], mean_kpi, std_kpi)
        kategori_x_pos.append((x_left + x_right) / 2)
    for xpos, label, color in zip(kategori_x_pos, kategori_labels_order, kategori_colors):
        ax.text(xpos, ymin-0.04, label, fontsize=13, color='black', ha='center', va='top', fontweight='bold',
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.13', alpha=0.33))
    ax.set_xticks([])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ymin-0.09, y.max()*1.08)
    ax.set_xlabel('')
    ax.set_ylabel('Densitas')
    ax.set_title(title, fontsize=15, weight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

# -------------------- TABEL KATEGORI -------------------- #
def tampilkan_tabel_kategori(df_sub, all_scores):
    for label in ['Kurang', 'Cukup', 'Baik', 'Sangat Baik', 'Istimewa']:
        df_label = df_sub[df_sub['Skor_KPI_Final'].apply(lambda x: kategori_from_score(x, all_scores)) == label][['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']]
        st.markdown(f"**{label}**")
        st.write(df_label.reset_index(drop=True))

# ==================== APP ==================== #

st.title("Kurva Distribusi Normal KPI Pegawai Berdasarkan Kategori (x-axis = Kategori)")
all_scores = df['Skor_KPI_Final'].values
all_nipps = df['NIPP_Pekerja'].values

# Kurva & tabel seluruh pegawai
plot_kurva(all_scores, all_nipps, "Kurva Distribusi Normal Seluruh Pegawai Pelindo")
st.subheader("Tabel Pegawai Berdasarkan Kategori (Seluruh Pegawai)")
tampilkan_tabel_kategori(df, all_scores)

# Analisis per Group/Atasan Langsung
st.header("Analisis Per Group/Atasan Langsung (Nama_Posisi)")
list_atasan = sorted(df['Nama_Posisi'].unique())
pilihan_atasan = st.selectbox(
    "Pilih Nama_Posisi untuk melihat distribusi group/dept:", list_atasan
)
df_group = df[df['Nama_Posisi'] == pilihan_atasan]
if len(df_group) > 0:
    group_scores = df_group['Skor_KPI_Final'].values
    group_nipps = df_group['NIPP_Pekerja'].values
    st.subheader(f"Kurva Distribusi Normal Pegawai {pilihan_atasan}")
    plot_kurva(group_scores, group_nipps, f"Kurva Distribusi Normal Pegawai {pilihan_atasan}")
    st.subheader(f"Tabel Pegawai Berdasarkan Kategori ({pilihan_atasan})")
    tampilkan_tabel_kategori(df_group, group_scores)
else:
    st.info("Tidak ada pegawai pada group/posisi ini.")

