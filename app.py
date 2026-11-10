"""PantaMitra: crop disease detection + regional-language voice advice.
Run: streamlit run app.py
"""
import io
import json
import os

import numpy as np
import streamlit as st
from gtts import gTTS
from PIL import Image

LANGS = {"English": "en", "తెలుగు": "te", "हिन्दी": "hi"}

# Keyed by the disease part of PlantVillage class names (Crop___Disease).
ADVICE = {
    "healthy": {
        "en": "The leaf looks healthy. Keep monitoring your field regularly.",
        "te": "ఆకు ఆరోగ్యంగా ఉంది. క్రమం తప్పకుండా పొలాన్ని గమనించండి.",
        "hi": "पत्ता स्वस्थ है। खेत की नियमित निगरानी करते रहें।",
    },
    "early_blight": {
        "en": "Early blight detected. Remove infected leaves and avoid wetting the foliage. If it spreads, use a fungicide on the advice of your agriculture officer.",
        "te": "ఇది ముందస్తు ఆకుమచ్చ తెగులు. సోకిన ఆకులను తీసివేసి, ఆకులపై నీరు పడకుండా చూడండి. అవసరమైతే వ్యవసాయ అధికారి సలహాతో శిలీంద్రనాశిని వాడండి.",
        "hi": "यह अगेती झुलसा रोग है। संक्रमित पत्ते हटाएं और पत्तियों पर पानी न पड़ने दें। जरूरत हो तो कृषि अधिकारी की सलाह से फफूंदनाशक का प्रयोग करें।",
    },
    "late_blight": {
        "en": "Late blight detected. It spreads fast. Remove infected plants immediately and contact your agriculture officer.",
        "te": "ఇది ఆలస్య ఆకుమచ్చ తెగులు, వేగంగా వ్యాపిస్తుంది. సోకిన మొక్కలను వెంటనే తొలగించి, వ్యవసాయ అధికారిని సంప్రదించండి.",
        "hi": "यह पछेती झुलसा रोग है, जो तेजी से फैलता है। संक्रमित पौधे तुरंत हटाएं और कृषि अधिकारी से संपर्क करें।",
    },
    "leaf_mold": {
        "en": "Leaf mold detected. Improve air flow between plants and remove infected leaves.",
        "te": "ఇది ఆకు బూజు తెగులు. గాలి బాగా ఆడేలా మొక్కల మధ్య ఎడం ఉంచండి, సోకిన ఆకులు తీసివేయండి.",
        "hi": "यह पत्ती फफूंद रोग है। पौधों के बीच हवा के लिए जगह रखें और संक्रमित पत्ते हटाएं।",
    },
    "bacterial_spot": {
        "en": "Bacterial spot detected. Remove infected leaves, avoid overhead watering and ask your agriculture officer for treatment.",
        "te": "ఇది బ్యాక్టీరియా మచ్చ తెగులు. సోకిన ఆకులు తీసివేసి, పై నుంచి నీరు చల్లడం మానండి, వ్యవసాయ అధికారి సలహా తీసుకోండి.",
        "hi": "यह जीवाणु धब्बा रोग है। संक्रमित पत्ते हटाएं, ऊपर से पानी देना बंद करें और कृषि अधिकारी से सलाह लें।",
    },
}
GENERIC = {
    "en": "A possible disease was detected. Please show the leaf to your local agriculture officer for confirmation.",
    "te": "ఒక తెగులు ఉండే అవకాశం ఉంది. నిర్ధారణ కోసం స్థానిక వ్యవసాయ అధికారికి ఆకును చూపించండి.",
    "hi": "संभावित रोग मिला है। पुष्टि के लिए स्थानीय कृषि अधिकारी को पत्ता दिखाएं।",
}
LOW_CONF = {
    "en": "I am not sure. Please retake a clear, close photo of a single leaf.",
    "te": "నాకు ఖచ్చితంగా తెలియదు. ఒక ఆకు యొక్క స్పష్టమైన, దగ్గరి ఫోటో మళ్లీ తీయండి.",
    "hi": "मुझे यकीन नहीं है। कृपया एक पत्ते की साफ, नजदीकी फोटो दोबारा लें।",
}


@st.cache_resource
def load_model():
    if not (os.path.exists("crop_model.keras") and os.path.exists("classes.json")):
        return None, None
    import tensorflow as tf
    return tf.keras.models.load_model("crop_model.keras"), json.load(open("classes.json"))


def speak(text, lang):
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)  # needs internet
    return buf.getvalue()


st.set_page_config(page_title="PantaMitra", page_icon="🌿")
st.title("🌿 PantaMitra")
st.caption("AI crop doctor that speaks your language")

lang_name = st.radio("Language / భాష / भाषा", list(LANGS), horizontal=True)
lang = LANGS[lang_name]
model, classes = load_model()

label, conf = None, 1.0
if model is None:
    st.warning("Demo mode: no trained model found. Run train.py to enable photo detection.")
    label = st.selectbox("Pick a disease to preview advice",
                         ["Tomato___Early_blight", "Tomato___Late_blight",
                          "Tomato___Leaf_Mold", "Tomato___Bacterial_spot",
                          "Potato___healthy"])
else:
    src = st.camera_input("Take a leaf photo") or st.file_uploader(
        "...or upload one", type=["jpg", "jpeg", "png"])
    if src:
        img = Image.open(src).convert("RGB")
        st.image(img, use_container_width=True)
        arr = np.array(img.resize((224, 224)), dtype="float32")[None]
        probs = model.predict(arr, verbose=0)[0]
        i = int(np.argmax(probs))
        label, conf = classes[i], float(probs[i])

if label:
    crop, _, disease = label.partition("___")
    st.subheader(f"{crop.replace('_', ' ')}: {disease.replace('_', ' ')}")
    st.progress(conf, text=f"Confidence: {conf:.0%}")
    if conf < 0.6:
        text = LOW_CONF[lang]
    else:
        text = ADVICE.get(disease.lower(), GENERIC)[lang]
    st.info(text)
    try:
        st.audio(speak(text, lang), format="audio/mp3")
    except Exception:
        st.caption("Voice needs an internet connection.")
