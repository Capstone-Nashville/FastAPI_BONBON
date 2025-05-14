from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Student GPA Prediction API")

# Load model
with open("Kmeans_Model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaller.pkl", "rb") as f:
    scaller = pickle.load(f)

# Load pca
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Definisikan input data
class data(BaseModel):
    NoTransaksi: int = Field(..., example=5555, description="Nomor Transaksi")
    Outlet: str = Field(..., example="cendana samarinda", description="Outlet dimana customer membeli produk")
    TotalQuantity: int = Field(..., example=4, description="Total kuantitas yang dibeli dalam 1 transaksi")
    TotalPembelian: int = Field(..., example=36000, description="Total pembelian dalam 1 transaksi")
    AvgHarga: float = Field(..., example=4500.0, description="Rata-rata harga produk dalam 1 transaksi")
    ProdukUnik: int = Field(..., example=2, description="Jumlah produk unik dalam 1 transaksi")
    KategoriUnik: int = Field(..., example=1, description="Jumlah kategori unik dalam 1 transaksi")
    JenisOrder: str = Field(..., example="lainnya", description="Jenis order yang dilakukan dalam 1 transaksi")

# Fungsi untuk membuat fitur tambahan
def preprocess_input(data: data):
    # Buat DataFrame dari input
    df = pd.DataFrame([{
        "No Transaksi": data.NoTransaksi,
        "Outlet": data.Outlet.lower(),
        "Total_Quantity": data.TotalQuantity,
        "Total_Pembelian": data.TotalPembelian,
        "Avg_Harga": data.AvgHarga,
        "Produk_Unik": data.ProdukUnik,
        "Kategori_Unik": data.KategoriUnik,
        "Jenis_Order": data.JenisOrder.lower(),
    }])

    # Konversi Outlet menjadi angka
    df['Outlet'] = df['Outlet'].map({
        'aws smd': 0,
        'big mall smd': 1,
        'cendana smd': 2,
        'Cut Nyak Dien tgr': 3,
        'Kampung Baru tgr': 4,
        'siradj salman smd': 5
    }).fillna(0).astype(int) 

    # Konversi Jenis_Order menjadi angka
    df['Jenis_Order'] = df['Jenis_Order'].map({
        'bungkus': 0,
        'free table': 1,
        'lainnya': 2,
        'grabfood': 3,
    }).fillna(0).astype(int)

    # Urutkan kolom supaya sesuai model
    df = df[[
        "No Transaksi",
        "Outlet",
        "Total_Quantity",
        "Total_Pembelian",
        "Avg_Harga",
        "Produk_Unik",
        "Kategori_Unik",
        "Jenis_Order"
    ]]


    return df

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "Customer Clusters Prediction API is running"}

# Endpoint untuk prediksi GPA
@app.post("/predict")
def predict_clusters(data: data):
    processed = preprocess_input(data)
    scaled = scaller.transform(processed)
    transformed = pca.transform(scaled)
    prediction = int(model.predict(transformed)[0])
    
    return {
        "cluster prediction": int(prediction)
    }

# Jalankan FastAPI dengan uvicorn
# uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)