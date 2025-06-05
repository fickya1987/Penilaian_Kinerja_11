import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------- KATEGORI ---------------------- #
kategori_labels = [
    ("Kurang", 0.00, 0.05, "#FFB3B3", "red"),
    ("Cukup", 0.05, 0.10, "#FFE3A1", "orange"),
    ("Baik", 0.10, 0.85, "#B3E0FF", "deepskyblue"),
    ("Sangat Baik", 0.85, 0.95, "#B3FFB3", "green"),
    ("Istimewa", 0.95, 1.00, "#FFFACD", "gold"),
]

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

# ------------------- DATA LOADER ------------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("Penilaian_Kinerja.csv")
    df = df.dropna(subset=['Skor_KPI_Final', 'NIPP_Pekerja', 'Nama_Posisi'])
    return df

df = load_data()
all_scores = df['Skor_KPI_Final'].values
df['Kategori_KPI'] = df['Skor_KPI_Final'].apply(lambda x: kategori_from_score(x, all_scores))

# ------------------- PLOTTING FUNCTION ------------------- #
def plot_kurva(scores, nipp_list=None, title=None):
    mean_kpi = np.mean(scores)
    std_kpi = np.std(scores)
    x = np.linspace(90, 110, 1000)
    y = norm.pdf(x, mean_kpi, std_kpi)

    fig, ax = plt.subplots(figsize=(12, 4))
    # Warnai area kategori
    xlims = []
    for label, pmin, pmax, color, _ in kategori_labels:
        xmin = np.clip(norm.ppf(pmin, mean_kpi, std_kpi), 90, 110)
        xmax = np.clip(norm.ppf(pmax, mean_kpi, std_kpi), 90, 110)
        ax.fill_between(x, 0, y, where=(x >= xmin) & (x <= xmax), color=color, alpha=0.4)
        xlims.append((xmin, xmax))
    ax.plot(x, y, color='black', linewidth=3, label="Kurva Normal")
    if nipp_list is not None:
        ax.scatter(scores, np.zeros_like(scores), color='gray', s=30, alpha=0.7, label="NIPP Pekerja")

    ax.set_xlim(90, 110)
    ax.set_xlabel("Skor KPI")
    ax.set_ylabel("Densitas")
    if title: ax.set_title(title, fontsize=16, weight='bold')
    ax.legend()

    # Secondary axis for kategori
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Tempatkan label kategori di tengah area warna
    ax2.set_xticks([(a+b)/2 for a, b in xlims])
    ax2.set_xticklabels(
        [l for l, *_ in kategori_labels],
        fontsize=13,
        color=[c for _, _, _, _, c in kategori_labels],
        fontweight='bold'
    )
    ax2.tick_params(axis='x', length=0)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel("")
    st.pyplot(fig)

# ------------------- MAIN APP ------------------- #

st.title("Kurva Distribusi Normal KPI Seluruh Pegawai (Korporasi/Pelindo)")

plot_kurva(df['Skor_KPI_Final'].values, nipp_list=df['NIPP_Pekerja'].values, title="Kurva Distribusi Normal Skor KPI Pegawai Pelindo")

st.header("Tabel Pegawai Berdasarkan Kategori KPI (Seluruh Pegawai)")
for label, _, _, _, _ in kategori_labels:
    df_label = df[df['Kategori_KPI'] == label][['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']]
    st.subheader(label)
    st.write(df_label.reset_index(drop=True))

# Pilihan Group/Atasan
st.header("Analisis Per Group/Atasan Langsung")
pilihan_atasan = st.selectbox(
    "Pilih Atasan Langsung (Nama_Posisi):",
    sorted(df['Nama_Posisi'].unique())
)
df_group = df[df['Nama_Posisi'] == pilihan_atasan]

if len(df_group) > 1:
    group_scores = df_group['Skor_KPI_Final'].values
    st.subheader(f"Kurva Distribusi Normal Pegawai {pilihan_atasan}")
    plot_kurva(group_scores, nipp_list=df_group['NIPP_Pekerja'].values,
               title=f"Kurva Distribusi Normal KPI Pegawai {pilihan_atasan}")

    # Kategori tabel per group
    st.subheader(f"Tabel Pegawai Berdasarkan Kategori KPI ({pilihan_atasan})")
    group_scores_all = df_group['Skor_KPI_Final'].values
    for label, _, _, _, _ in kategori_labels:
        df_label = df_group[
            df_group['Skor_KPI_Final'].apply(lambda x: kategori_from_score(x, group_scores_all)) == label
        ][['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']]
        st.markdown(f"**{label}**")
        st.write(df_label.reset_index(drop=True))
else:
    st.info("Data group terlalu sedikit untuk menampilkan kurva distribusi normal.")

