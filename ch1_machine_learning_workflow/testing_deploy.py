from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model, encoder, dan scaler
# Saat training, pastikan kamu simpan tuple: (model, scaler, category_encoders)
# GBR, scaler, category_features, numeric_features, label_encoders = joblib.load(
#     "gbr_model_encoded_scaler.joblib"
# )
GBR, scaler, category_features, numeric_features, label_encoders = joblib.load(
    "gbr_model_encoded_scaler.joblib"
)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data JSON
        input_data = request.json[
            "data"
        ]  # dict, misalnya {"Item": "Cake", "Quantity": 4, ...}

        # --- Preprocessing ---
        row = []

        # 1. Encode kategori
        for col in category_features:
            if col in input_data:
                val = input_data[col]
                le = label_encoders[col]
                if val in le.classes_:
                    row.append(le.transform([val])[0])
                else:
                    row.append(-1)  # handle kategori unknown
            else:
                row.append(0)

        # 2. Tambahkan fitur numerik
        for col in numeric_features:
            if col == "Total Spent":
                # skip karena ini target (y), bukan fitur
                continue
            val = input_data.get(col, 0)  # default 0 kalau kosong/null
            row.append(float(val))

        X = np.array(row).reshape(1, -1)

        # 3. Scaling numerik (pakai scaler yang sudah fit di training)
        X[:, -len(numeric_features) :] = scaler.transform(
            X[:, -len(numeric_features) :]
        )

        # --- Prediksi ---
        y_pred_scaled = GBR.predict(X)

        # Inverse transform untuk dapatkan harga asli
        dummy = np.zeros((1, len(numeric_features)))
        dummy[0, -1] = y_pred_scaled  # taruh prediksi di posisi target (kolom terakhir)
        y_pred_original = scaler.inverse_transform(dummy)[0, -1]

        return jsonify(
            {
                "prediction_scaled": float(
                    y_pred_scaled[0]
                ),  # hasil dalam skala standar
                "prediction_original": float(
                    y_pred_original
                ),  # hasil dalam skala asli (harga asli)
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
