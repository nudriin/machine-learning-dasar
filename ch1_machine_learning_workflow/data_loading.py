import pandas as pd

df = pd.read_csv(
    "../data/bps-od_17630_indeks_harga_dibayar_petani__sub_kelompok_pengeluaran_.csv"
)

print(df.head())  # menampilkan 5 baris pertama
df.info()  # cek tipe data dari setiap kolom
