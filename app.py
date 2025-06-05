import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("Penilaian_Kinerja.csv")

df = load_data()

# --- Kategori ---
kategori_labels = [
    ('Kurang', 0.00, 0.05, '#FF6F69'),
    ('Cukup', 0.05, 0.10, '#FFCC5C'),
    ('Baik', 0.10, 0.90, '#96DCEC'),
    ('Sangat Baik', 0.90, 0.975, '#88D498'),
    ('Istimewa', 0.975, 1.00, '#FFD700'),
]

label_props = {
    'Kurang': dict(color='red', weight='bold'),
    'Cukup': dict(color='orange', weight='bold'),
    'Baik': dict(color='deepskyblue', weight='bold'),
    'Sangat Baik': dict(color='green', weight='bold'),
    'Istimewa': dict(color='goldenrod', weight='bold')
}

def kategori_kpi(val, mean, std):
    # Hitung Z-score
    z = (val - mean) / std if std > 0 else 0
    p = norm.cdf(val, loc=mean, scale=std)
    for label, pmin, pmax, _ in kategori_labels:
        if pmin <= p < pmax:
            return label
    return 'Undefined'

# --- Fungsi Plotting ---

def plot_kurva(mean_kpi, std_kpi, df_poin, title, legend_label):
    used_std = std_kpi if (not np.isnan(std_kpi) and std_kpi > 0) else 0.2
    x = np.linspace(90, 110, 1000)
    y = norm.pdf(x, mean_kpi, used_std)
    fig, ax = plt.subplots(figsize=(12, 4))
    # Fill kategori
    for label, pmin, pmax, color in kategori_labels:
        x_fill = norm.ppf([pmin, pmax], mean_kpi, used_std)
        x_fill = np.clip(x_fill, 90, 110)
        mask = (x >= x_fill[0]) & (x <= x_fill[1])
        ax.fill_between(x[mask], y[mask], alpha=0.23, color=color, label=label if title == 'Korporasi/Pelindo' else None)
        xpos = (x_fill[0] + x_fill[1]) / 2
        # tampilkan label hanya jika area cukup lebar
        if (x_fill[1] - x_fill[0]) >= 1.1:
            props = label_props[label]
            ax.annotate(label, (xpos, -0.01), ha='center', va='top',
                        fontsize=10, color=props['color'], fontweight=props['weight'], annotation_clip=False)
    # Kurva normal
    ax.plot(x, y, color='black', linewidth=2, label=legend_label)
    # Plot NIPP
    ax.scatter(df_poin['Skor_KPI_Final'], np.zeros_like(df_poin['Skor_KPI_Final']), 
               color='gray', alpha=0.4, s=30, label='NIPP Pekerja (seluruh korporasi)')
    for i, row in df_poin.iterrows():
        ax.annotate(str(int(row['NIPP_Pekerja'])), (row['Skor_KPI_Final'], -0.014), fontsize=7, color='gray', ha='center', va='top', rotation=90, alpha=0.9)
    ax.set_xlim(90, 110)
    ax.set_ylim(-0.025, y.max() + 0.03)
    ax.set_xlabel('Skor KPI')
    ax.set_ylabel('Densitas')
    ax.set_title(f'Kurva Distribusi Normal Skor KPI Pegawai {title}')
    ax.legend(fontsize=9, loc='upper left')
    plt.subplots_adjust(bottom=0.21)
    st.pyplot(fig)

# --- Statistik dan Tabel ---
def klasifikasi_karyawan(df_input, mean, std):
    kategori_list = []
    for i, row in df_input.iterrows():
        val = row['Skor_KPI_Final']
        label = kategori_kpi(val, mean, std)
        kategori_list.append(label)
    df_input['Kategori_KPI'] = kategori_list
    return df_input

def summary_kategori(df_input):
    return df_input.groupby('Kategori_KPI').size().reindex(
        [k[0] for k in kategori_labels], fill_value=0
    )

def tabel_kategori(df_input):
    cols = ['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final', 'Kategori_KPI']
    for kat, _, _, _ in kategori_labels:
        st.markdown(f"**Kategori: {kat}**")
        sub = df_input[df_input['Kategori_KPI'] == kat][cols]
        if not sub.empty:
            st.dataframe(sub, hide_index=True)
        else:
            st.write("_Tidak ada pegawai di kategori ini._")

# --- STREAMLIT MAIN ---
st.title("Kurva Distribusi Normal KPI Seluruh Pegawai (Korporasi/Pelindo)")

df_kpi = df[['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']].dropna()
mean_kpi = df_kpi['Skor_KPI_Final'].mean()
std_kpi = df_kpi['Skor_KPI_Final'].std(ddof=0)

df_kpi = klasifikasi_karyawan(df_kpi, mean_kpi, std_kpi)
plot_kurva(mean_kpi, std_kpi, df_kpi, title="Pelindo", legend_label="Kurva Normal")

# --- Statistik Distribusi Seluruh Pegawai ---
st.markdown("### Statistik Distribusi Pegawai per Kategori")
kat_summary = summary_kategori(df_kpi)
st.dataframe(kat_summary.reset_index().rename(columns={0: 'Jumlah Pegawai', 'index': 'Kategori'}), hide_index=True)

st.markdown("### Daftar Pegawai per Kategori")
tabel_kategori(df_kpi)

# --- Distribusi per Atasan Langsung (Group/Dept) ---
st.markdown("---")
st.header("Kurva Distribusi Normal KPI untuk Tiap Atasan Langsung (Group/Dept)")

atasans = df['NIPP_Atasan'].dropna().unique()
for atasan in atasans:
    group_df = df[df['NIPP_Atasan'] == atasan][['NIPP_Pekerja', 'Nama_Posisi', 'Skor_KPI_Final']].dropna()
    if len(group_df) < 2: continue
    mean_group = group_df['Skor_KPI_Final'].mean()
    std_group = group_df['Skor_KPI_Final'].std(ddof=0)
    group_df = klasifikasi_karyawan(group_df, mean_group, std_group)
    st.markdown(f"#### Bawahan dari Atasan: {group_df.iloc[0]['Nama_Posisi']} (NIPP {atasan:.0f})")
    plot_kurva(mean_group, std_group, group_df, title=group_df.iloc[0]['Nama_Posisi'], legend_label="Kurva Normal (bawahan)")
    st.markdown(f"**Daftar Pegawai per Kategori (Atasan: {group_df.iloc[0]['Nama_Posisi']})**")
    tabel_kategori(group_df)
    st.markdown("---")
