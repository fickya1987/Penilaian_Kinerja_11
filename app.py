import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# ===================== KONFIGURASI KATEGORI ===================== #
kategori_labels = [
    ("Kurang",       0.00, 0.05, '#ff9999',    'red'),
    ("Cukup",        0.05, 0.10, '#ffe5b4',    'orange'),
    ("Baik",         0.10, 0.85, '#b3e0ff',    'deepskyblue'),
    ("Sangat Baik",  0.85, 0.97, '#b4ffb4',    'green'),
    ("Istimewa",     0.97, 1.00, '#fff399',    'gold'),
]

# ===================== DATA LOADER ===================== #
@st.cache_data
def load_data():
    df = pd.read_csv("Penilaian_Kinerja.csv")
    df = df.dropna(subset=['Skor_KPI_Final', 'NIPP_Pekerja', 'Nama_Posisi'])
    return df

df = load_data()
scores = df['Skor_KPI_Final'].clip(lower=90, upper=110).values
nipps = df['NIPP_Pekerja'].values

# ===================== KATEGORI FUNGSIONAL ===================== #
def kategori_from_score(score, scores):
    percentiles = np.percentile(scores, [5, 10, 85, 97])
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

df['Kategori_KPI'] = df['Skor_KPI_Final'].apply(lambda x: kategori_from_score(x, scores))

# ===================== PLOTTING FUNCTION ===================== #
def plot_kurva(scores, nipp_list=None, title=None):
    mean_kpi = np.mean(scores)
    std_kpi = np.std(scores)
    x = np.linspace(90, 110, 1000)
    y = norm.pdf(x, mean_kpi, std_kpi if std_kpi > 0 else 0.1)

    fig, ax = plt.subplots(figsize=(13, 5))
    xlims = []
    # Fill background area kategori
    for label, pmin, pmax, fill_color, _ in kategori_labels:
        xmin = np.clip(norm.ppf(pmin, mean_kpi, std_kpi if std_kpi > 0 else 0.1), 90, 110)
        xmax = np.clip(norm.ppf(pmax, mean_kpi, std_kpi if std_kpi > 0 else 0.1), 90, 110)
        ax.fill_between(x, 0, y, where=(x >= xmin) & (x <= xmax), color=fill_color, alpha=0.4)
        xlims.append((xmin, xmax))

    ax.plot(x, y, color='black', linewidth=3, label="Kurva Normal")
    if nipp_list is not None:
        ax.scatter(scores, np.zeros_like(scores), color='gray', s=32, alpha=0.65, label="NIPP Pekerja")
        for (score, nipp) in zip(scores, nipp_list):
            ax.annotate(str(int(nipp)), (score, 0), textcoords="offset points", xytext=(0, 6),
                        ha='center', fontsize=7, color='gray', rotation=90)

    ax.set_xlim(90, 110)
    ax.set_xlabel("Skor KPI")
    ax.set_ylabel("Densitas")
    if title:
        ax.set_title(title, fontsize=18, weight='bold')
    ax.legend(loc='upper left', fontsize=13)

    # Second axis untuk kategori
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([(a+b)/2 for a, b in xlims])
    ax2.set_xticklabels(
        [l for l, *_ in kategori_labels],
        fontsize=14,
        fontweight='bold'
    )
    for xtick, (_, _, _, _, color) in zip(ax2.get_xticklabels(), kategori_labels):
        xtick.set_color(color)
    ax2.tick_params(axis='x', length=0)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel("")
    st.pyplot(fig)

# ===================== MAIN PAGE ===================== #
st.title("Kurva Distribusi Normal KPI Seluruh Pegawai (Korporasi/Pelindo)")

plot_kurva(scores, nipps, title="Kurva Distribusi Normal Skor KPI Pegawai Pelindo")

st.header("Tabel Pegawai Berdasarkan Kategori KPI (Seluruh Pegawai)")
for label, _, _, _, _ in kategori_labels:
    df_label = df[df['Kategori_KPI'] == label][['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']]
    st.subheader(label)
    st.write(df_label.reset_index(drop=True))

# ===================== ANALISIS GROUP/ATASAN ===================== #
st.header("Analisis Per Group/Atasan Langsung (Nama_Posisi)")
list_atasan = sorted(df['Nama_Posisi'].unique())
pilihan_atasan = st.selectbox(
    "Pilih Nama_Posisi untuk melihat distribusi group/dept:", list_atasan
)
df_group = df[df['Nama_Posisi'] == pilihan_atasan]
if len(df_group) > 1:
    group_scores = df_group['Skor_KPI_Final'].clip(lower=90, upper=110).values
    group_nipps = df_group['NIPP_Pekerja'].values
    st.subheader(f"Kurva Distribusi Normal Pegawai {pilihan_atasan}")
    plot_kurva(group_scores, group_nipps, title=f"Kurva Distribusi Normal KPI Pegawai {pilihan_atasan}")

    st.subheader(f"Tabel Pegawai Berdasarkan Kategori KPI ({pilihan_atasan})")
    group_scores_all = group_scores
    for label, _, _, _, _ in kategori_labels:
        df_label = df_group[
            df_group['Skor_KPI_Final'].apply(lambda x: kategori_from_score(x, group_scores_all)) == label
        ][['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']]
        st.markdown(f"**{label}**")
        st.write(df_label.reset_index(drop=True))
else:
    st.info("Data group terlalu sedikit untuk menampilkan kurva distribusi normal.")
