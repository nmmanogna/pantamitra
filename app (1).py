"""PantaMitra: crop disease detection + full advice in English, Telugu, Hindi.
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

# Full advice per disease (English). Telugu/Hindi are translated at runtime.
KB = {
    "healthy": dict(sev="none", name="Healthy leaf",
        cause="No disease signs were found on this leaf.",
        symptoms="Even green colour, no spots, curling or webbing.",
        cure="No treatment needed.",
        prevent="Keep checking the field every few days, water at the base of the plant and keep the field clean."),
    "early_blight": dict(sev="medium", name="Early blight",
        cause="A fungus (Alternaria solani) that spreads in warm, humid weather.",
        symptoms="Brown spots with ring patterns like a target, starting on older lower leaves, with yellowing around them. Leaves may dry and fall.",
        cure="Remove and destroy badly affected leaves. If it keeps spreading, spray a fungicide such as mancozeb or chlorothalonil as per the label and your agriculture officer's advice.",
        prevent="Rotate crops every 2 to 3 years, space the plants for air flow, mulch the soil and avoid wetting the leaves."),
    "late_blight": dict(sev="high", name="Late blight",
        cause="A water mould (Phytophthora infestans) that spreads very fast in cool, wet weather.",
        symptoms="Large dark, greasy-looking patches on leaves and stems, white fuzzy growth underneath in humid weather, and rotting fruit or tubers.",
        cure="Act immediately. Remove and destroy infected plants (do not compost them) and contact your agriculture officer at once for the right fungicide and schedule.",
        prevent="Use resistant varieties and healthy seed, avoid overhead watering, give good spacing and do not leave infected plant waste in the field."),
    "bacterial_spot": dict(sev="medium", name="Bacterial spot",
        cause="Bacteria (Xanthomonas) that spread by water splash, tools and infected seed.",
        symptoms="Small dark, water-soaked spots on leaves that turn brown or black, often with a yellow ring. Rough scabby spots on fruit.",
        cure="Remove infected leaves and avoid working in the field when plants are wet. Copper-based sprays can slow it, used as per the label and your agriculture officer's advice.",
        prevent="Use disease-free seed, rotate crops, water at the base instead of overhead and clean tools."),
    "leaf_mold": dict(sev="medium", name="Leaf mold",
        cause="A fungus (Passalora fulva) that thrives in high humidity and poor air flow.",
        symptoms="Pale green or yellow patches on the upper side of leaves and an olive-green to brown velvety mould underneath.",
        cure="Improve ventilation, remove infected leaves and, if it spreads, use a fungicide as per the label and your agriculture officer's advice.",
        prevent="Reduce humidity, keep plants well spaced and pruned and avoid wetting the leaves."),
    "septoria": dict(sev="medium", name="Septoria leaf spot",
        cause="A fungus (Septoria lycopersici) that spreads by rain splash.",
        symptoms="Many small round spots with dark edges and grey-white centres, often with tiny black dots, starting on lower leaves.",
        cure="Remove infected lower leaves. If it spreads, spray a fungicide such as chlorothalonil, mancozeb or copper as per the label and your agriculture officer's advice.",
        prevent="Rotate crops, mulch to stop soil splash, avoid overhead watering and clear plant waste after harvest."),
    "spider": dict(sev="medium", name="Spider mite damage",
        cause="Tiny two-spotted spider mites (a pest, not a fungus) that multiply in hot, dry weather.",
        symptoms="Fine yellow speckles, bronze or dry leaves and thin webbing on the underside of leaves.",
        cure="Spray water on the underside of leaves, then neem oil or insecticidal soap. For heavy attacks, use a miticide only on your agriculture officer's advice.",
        prevent="Avoid dust and water stress, check leaf undersides regularly and protect natural predators of mites."),
    "target_spot": dict(sev="medium", name="Target spot",
        cause="A fungus (Corynespora cassiicola) that spreads in warm, humid conditions.",
        symptoms="Brown spots with ring patterns and a yellow edge on leaves, sometimes on stems and fruit.",
        cure="Remove infected leaves and, if it spreads, use a fungicide as per the label and your agriculture officer's advice.",
        prevent="Improve air flow, rotate crops and avoid wetting the leaves."),
    "yellow_leaf_curl": dict(sev="high", name="Yellow leaf curl virus",
        cause="A virus spread by whiteflies. The virus itself cannot be cured.",
        symptoms="Leaves curl upward and turn yellow at the edges, the plant stays small and gives very few fruits.",
        cure="Remove and destroy infected plants early so the virus does not spread. Control whiteflies with yellow sticky traps, neem oil or an insecticide on your agriculture officer's advice.",
        prevent="Use resistant varieties, cover nursery beds with fine net, remove weeds and control whiteflies from the start."),
    "mosaic": dict(sev="high", name="Mosaic virus",
        cause="A virus that spreads by touch, tools, hands and infected seed. The virus itself cannot be cured.",
        symptoms="Light and dark green mottled patches, curled or narrow leaves and stunted plants.",
        cure="Remove and destroy infected plants. Wash hands and disinfect tools after handling them.",
        prevent="Use clean seed and resistant varieties, disinfect tools and avoid handling plants after tobacco use."),
}
UI = {
    "en": dict(cause="What is it", symptoms="Symptoms", cure="Treatment", prevent="Prevention", conf="Confidence"),
    "te": dict(cause="ఇది ఏమిటి", symptoms="లక్షణాలు", cure="చికిత్స", prevent="నివారణ", conf="నమ్మకం"),
    "hi": dict(cause="यह क्या है", symptoms="लक्षण", cure="इलाज", prevent="रोकथाम", conf="भरोसा"),
}
CAUTION = "Always follow the product label and confirm serious cases with your local agriculture officer."
LOW = "I am not sure. Please retake a clear, close photo of a single leaf."
UNKNOWN = "A possible disease was detected. Please show the leaf to your local agriculture officer for confirmation and treatment."


@st.cache_data(show_spinner=False)
def tr(text, lang):
    """Translate English text to Telugu/Hindi (needs internet). Falls back to English."""
    if lang == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=lang).translate(text)
    except Exception:
        return text


def lookup(disease):
    d = disease.lower()
    return next((k for k in KB if k in d), None)


@st.cache_resource
def load_model():
    if not (os.path.exists("crop_model.keras") and os.path.exists("classes.json")):
        return None, None
    import tensorflow as tf
    return tf.keras.models.load_model("crop_model.keras"), json.load(open("classes.json"))


def say(text, lang):
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)  # needs internet
        st.audio(buf.getvalue(), format="audio/mp3")
    except Exception:
        st.caption("Voice needs an internet connection.")


st.set_page_config(page_title="PantaMitra", page_icon="🌿")
st.title("🌿 PantaMitra")
st.caption("AI crop doctor that speaks your language")

lang_name = st.radio("Language / భాష / भाषा", list(LANGS), horizontal=True)
lang = LANGS[lang_name]
L = UI[lang]
model, classes = load_model()

label, conf = None, 1.0
if model is None:
    st.warning("Demo mode: no trained model found. Run train.py to enable photo detection.")
    label = st.selectbox("Pick a disease to preview advice", [
        "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Bacterial_spot",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"])
else:
    mode = st.radio("Photo source", ["Back camera", "Front camera", "Upload from gallery"],
                    horizontal=True)
    src = None
    if mode == "Back camera":
        try:
            from streamlit_back_camera_input import back_camera_input
            st.caption("Tap the video to capture. Allow camera permission when asked.")
            src = back_camera_input()
        except Exception:
            st.warning("Back camera add-on is not installed. Use another option.")
    elif mode == "Front camera":
        src = st.camera_input("Take a leaf photo")
    else:
        src = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    if src is not None:
        if isinstance(src, Image.Image):
            img = src
        elif isinstance(src, (bytes, bytearray)):
            img = Image.open(io.BytesIO(src))
        else:
            img = Image.open(src)
        img = img.convert("RGB")
        st.image(img, use_container_width=True)
        arr = np.array(img.resize((224, 224)), dtype="float32")[None]
        probs = model.predict(arr, verbose=0)[0]
        i = int(np.argmax(probs))
        label, conf = classes[i], float(probs[i])

if label:
    crop, _, disease = label.partition("___")
    crop_t = tr(crop.replace("_", " "), lang)
    key = lookup(disease)
    st.progress(conf, text=f"{L['conf']}: {conf:.0%}")
    if conf < 0.6:
        msg = tr(LOW, lang)
        st.warning(msg)
        say(msg, lang)
    elif key is None:
        msg = f"{crop_t}: {tr(disease.replace('_', ' '), lang)}. {tr(UNKNOWN, lang)}"
        st.warning(msg)
        say(msg, lang)
    else:
        k = KB[key]
        title = f"{crop_t}: {tr(k['name'], lang)}"
        {"none": st.success, "medium": st.warning, "high": st.error}[k["sev"]](f"### {title}")
        parts = {f: tr(k[f], lang) for f in ("cause", "symptoms", "cure", "prevent")}
        for f in ("cause", "symptoms", "cure", "prevent"):
            st.markdown(f"**{L[f]}**\n\n{parts[f]}")
        st.caption(tr(CAUTION, lang))
        say(f"{title}. {parts['cure']} {parts['prevent']}", lang)
