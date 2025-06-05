import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Fungsi kategori
def kategori_from_score(x, all_scores):
    p = np.percentile(all_scores, [5, 20, 80, 95])
    if x <= p[0]: return 'Kurang'
    elif x <= p[1]: return 'Cukup'
    elif x <= p[2]: return 'Baik'
    elif x <= p[3]: return 'Sangat Baik'
    else: return 'Istimewa'

# Definisi warna & label kategori
kategori_labels = [
    ('Kurang',    '#ff9999',  'red',        'Kurang'),
    ('Cukup',     '#ffe4b2',  'orange',     'Cukup'),
    ('Baik',      '#b2e1ff',  'deepskyblue','Baik'),
    ('Sangat Baik','#a6e3a1', 'green',      'Sangat Baik'),
    ('Istimewa',  '#ffe066',  'gold',       'Istimewa'),
]

def plot_kurva(scores, nipps, title):
    # Range tetap
    x_min, x_max = 90, 110
    scores = np.clip(scores, x_min, x_max)
    mu, sigma = np.mean(scores), np.std(scores) if np.std(scores) > 0 else 1e-3
    x = np.linspace(x_min, x_max, 400)
    y = norm.pdf(x, mu, sigma)

    # Percentiles untuk area
    percentiles = np.percentile(scores, [5, 20, 80, 95])
    batas = [x_min, percentiles[0], percentiles[1], percentiles[2], percentiles[3], x_max]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    
    # Background area
    for i, (label, color, fontcolor, lbl_txt) in enumerate(kategori_labels):
        ax.axvspan(batas[i], batas[i+1], color=color, alpha=0.3, label=label if i == 0 else None)
    
    # Kurva normal
    ax.plot(x, y, color='black', lw=3, label='Kurva Normal')
    
    # Titik NIPP (scatter di bawah sumbu X)
    ymin = -0.015 * y.max()
    ax.scatter(scores, np.full_like(scores, ymin), color='grey', alpha=0.7, s=55, label='NIPP Pekerja')
    # Label NIPP di bawah titik
    for sc, nipp in zip(scores, nipps):
        ax.annotate(str(int(nipp)), xy=(sc, ymin - 0.01), fontsize=7, ha='center', rotation=90)

    # Label kategori rapi di bawah sumbu X
    for i, (label, color, fontcolor, lbl_txt) in enumerate(kategori_labels):
        xtext = (batas[i]+batas[i+1])/2
        ax.text(xtext, ymin - 0.035, lbl_txt, fontsize=12, color=fontcolor, weight='bold', ha='center', va='top')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ymin - 0.06, y.max() * 1.05)
    ax.set_xlabel('Skor KPI')
    ax.set_ylabel('Densitas')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(title, fontsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("Penilaian_Kinerja.csv")

df = load_data()
df['Skor_KPI_Final'] = df['Skor_KPI_Final'].clip(lower=90, upper=110)

st.title("Kurva Distribusi Normal KPI Seluruh Pegawai (Korporasi/Pelindo)")

# --- Kurva Seluruh Pegawai
all_scores = df['Skor_KPI_Final'].values
all_nipps = df['NIPP_Pekerja'].values
plot_kurva(all_scores, all_nipps, "Kurva Distribusi Normal Skor KPI Pegawai Pelindo")

# --- Tabel kategori seluruh pegawai
st.subheader("Tabel Pegawai Berdasarkan Kategori (Seluruh Pegawai)")
for label, _, _, _ in kategori_labels:
    df_label = df[df['Skor_KPI_Final'].apply(lambda x: kategori_from_score(x, all_scores)) == label][['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']]
    st.markdown(f"**{label}**")
    st.write(df_label.reset_index(drop=True))

# --- Analisis per Group/Atasan Langsung
st.header("Analisis Per Group/Atasan Langsung (Nama_Posisi)")
list_atasan = sorted(df['Nama_Posisi'].unique())
pilihan_atasan = st.selectbox(
    "Pilih Nama_Posisi untuk melihat distribusi group/dept:", list_atasan
)
df_group = df[df['Nama_Posisi'] == pilihan_atasan]
if len(df_group) > 0:
    group_scores = df_group['Skor_KPI_Final'].clip(lower=90, upper=110).values
    group_nipps = df_group['NIPP_Pekerja'].values
    st.subheader(f"Kurva Distribusi Normal Pegawai {pilihan_atasan}")
    plot_kurva(group_scores, group_nipps, title=f"Kurva Distribusi Normal KPI Pegawai {pilihan_atasan}")

    st.subheader(f"Tabel Pegawai Berdasarkan Kategori KPI ({pilihan_atasan})")
    group_scores_all = group_scores
    for label, _, _, _ in kategori_labels:
        df_label = df_group[
            df_group['Skor_KPI_Final'].apply(lambda x: kategori_from_score(x, group_scores_all)) == label
        ][['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']]
        st.markdown(f"**{label}**")
        st.write(df_label.reset_index(drop=True))
else:
    st.info("Tidak ada pegawai pada group/posisi ini.")

