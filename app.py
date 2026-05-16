import streamlit as st
import numpy as np
import pickle
import json
import os
import gdown
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Diagnosis X-Ray Paru - Kelompok 07",
    page_icon="U+1FAC1",
    layout="wide"
)

# ── ID Google Drive — ISI SETELAH UPLOAD ─────────────────────────
GDRIVE_IDS = {
    "model_final.keras"    : "1SbS40YpUmxm7YWEYtNcTc5prcnyLOE80",
    "tfidf_vectorizer.pkl" : "1hA2ZRswxH4OSFPUFjWG0jFPIOeg7ydkM",  
    "clinical_scaler.pkl"  : "1aLmFg55NW2TenrpLFVUjCvWu9-gOVBeY", 
    "threshold.json"       : "1qNdO2AQRJGQbdSPhJ6upwp9kFSexdsE4",
    "config.json"          : "1_Yeu0HKAQA5VnLomHg4ROHYVgZlJbXFm",
}

@st.cache_resource(show_spinner=False)
def download_and_load():
    os.makedirs("deployment", exist_ok=True)

    for filename, file_id in GDRIVE_IDS.items():
        path = f"deployment/{filename}"
        if not os.path.exists(path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)

    # Registrasi Focal Loss — wajib sebelum load model
    try:
        register_fn = tf.keras.saving.register_keras_serializable
    except AttributeError:
        register_fn = tf.keras.utils.register_keras_serializable

    @register_fn(package="custom")
    def loss_fn(y_true, y_pred):
        gamma, alpha = 2.0, 0.25
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) \
                  - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)

    model = tf.keras.models.load_model(
        "deployment/model_final.keras"
        # Tidak perlu custom_objects karena sudah di-register di atas
    )
    with open("deployment/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("deployment/clinical_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("deployment/threshold.json") as f:
        thresholds = json.load(f)
    with open("deployment/config.json") as f:
        config = json.load(f)

    return model, vectorizer, scaler, thresholds, config

def prediksi(model, vectorizer, scaler, thresholds, config,
             img_pil, teks, umur, jenis_kelamin, jenis_pemeriksaan):

    label_list = config["label_list"]

    # Preprocessing gambar — sama persis dengan training
    img   = img_pil.resize((224, 224)).convert("RGB")
    arr   = np.array(img, dtype=np.float32)
    arr   = tf.keras.applications.resnet50.preprocess_input(arr)
    X_img = np.expand_dims(arr, axis=0)

    # TF-IDF
    X_tfidf = vectorizer.transform([teks.strip() if teks.strip() else "normal"]) \
                         .toarray().astype(np.float32)

    # Fitur klinis
    jk_enc = 1 if jenis_kelamin == "Laki-laki" else 0
    jp_enc = 1 if "PA" in jenis_pemeriksaan else 0
    clin   = np.array([[float(umur), jk_enc, jp_enc]], dtype=np.float32)
    clin[:, 0:1] = scaler.transform(clin[:, 0:1])

    # Prediksi
    proba = model.predict([X_img, X_tfidf, clin], verbose=0)[0]

    return {
        label: {
            "prob"    : float(p),
            "thresh"  : float(thresholds.get(label, 0.5)),
            "positif" : bool(p >= thresholds.get(label, 0.5))
        }
        for label, p in zip(label_list, proba)
    }

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Tentang Model")
    st.markdown("""
    **Arsitektur:** Late Fusion Multimodal
    **Backbone:** ResNet50 (pretrained ImageNet)
    **Teks:** TF-IDF Vectorizer (bigram)
    **Klinis:** Umur, Jenis Kelamin, Pemeriksaan
    **Loss:** Focal Loss
    **Kelompok 07 - IT Del**
    """)
    st.divider()
    st.warning("Bukan diagnosis medis resmi. Konsultasikan dengan dokter/radiolog.")

# ── Header ────────────────────────────────────────────────────────
st.title("Sistem Diagnosis Sementara X-Ray Paru")
st.markdown("**Kelompok 07 — Institut Teknologi Del** | 13 Kelas Penyakit Paru")
st.divider()

# ── Load model ────────────────────────────────────────────────────
with st.spinner("Memuat model (pertama kali beberapa menit)..."):
    try:
        model, vectorizer, scaler, thresholds, config = download_and_load()
        st.success("Model siap digunakan")
    except Exception as e:
        st.error(f"Gagal memuat: {e}")
        st.stop()

# ── Form input ────────────────────────────────────────────────────
col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.subheader("Upload Gambar X-Ray")
    uploaded = st.file_uploader("Pilih file .png / .jpg",
                                type=["png","jpg","jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="X-Ray yang diupload",
                 use_column_width=True)

with col2:
    st.subheader("Data Pasien")
    teks = st.text_area(
        "Teks Temuan Radiologi",
        placeholder="Contoh: Tampak kardiomegali. Tidak tampak infiltrat.",
        height=130
    )
    umur = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=None, step=1, placeholder="Masukkan umur...")
    c1, c2 = st.columns(2)
    with c1:
        jenis_kelamin = st.radio("Jenis Kelamin",
                                  ["Laki-laki","Perempuan"])
    with c2:
        jenis_pemeriksaan = st.radio("Jenis Pemeriksaan",
                                      ["PA (Posteroanterior)",
                                       "AP (Anteroposterior)"])

st.divider()
_, col_btn, _ = st.columns([2,1,2])
with col_btn:
    run = st.button("Analisis X-Ray", type="primary",
                    use_container_width=True)

# ── Hasil ─────────────────────────────────────────────────────────
if run:
    if not uploaded:
        st.error("Upload gambar X-Ray terlebih dahulu.")
    elif umur is None:
        st.error("Harap masukkan umur pasien terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            hasil = prediksi(model, vectorizer, scaler,
                             thresholds, config, img_pil,
                             teks, umur, jenis_kelamin, jenis_pemeriksaan)

        positif = [k for k,v in hasil.items() if v["positif"]]
        st.subheader("Hasil Analisis")

        if positif:
            st.error(f"Terdeteksi {len(positif)} kelainan:")
            cols = st.columns(min(len(positif), 4))
            for i, label in enumerate(positif):
                with cols[i % 4]:
                    st.metric(label, f"{hasil[label]['prob']:.2%}")
        else:
            st.success("Tidak terdeteksi kelainan — Normal")

        st.divider()

        # Bar chart
        labels_s = sorted(hasil, key=lambda x: hasil[x]["prob"], reverse=True)
        probas   = [hasil[l]["prob"] for l in labels_s]
        colors   = ["#E74C3C" if hasil[l]["positif"]
                    else "#2ECC71" for l in labels_s]

        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(labels_s, probas, color=colors, edgecolor="white")
        for l in labels_s:
            ax.axvline(hasil[l]["thresh"], color="orange",
                       linestyle="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Probabilitas")
        ax.set_title("Merah=Positif | Hijau=Negatif | Oranye=Threshold")
        ax.set_xlim([0,1])
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Tabel
        df = pd.DataFrame([
            {"Label": l,
             "Probabilitas": f"{v['prob']:.4f}",
             "Threshold": f"{v['thresh']:.2f}",
             "Status": "POSITIF" if v["positif"] else "Negatif"}
            for l,v in sorted(hasil.items(),
                               key=lambda x: x[1]["prob"], reverse=True)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Hasil prediksi AI untuk keperluan akademik. Bukan diagnosis medis resmi.")
