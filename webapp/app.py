import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

MODEL_DIR = r"D:\Codess\nlp_mental_health\models\distilbert_model"  # absolute path avoids cwd issues

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    # tokenizer first
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # config + resilient model load (bin or safetensors; CPU; low mem)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    try:
        mdl = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            config=cfg,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
    except Exception:
        mdl = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            config=cfg,
            from_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )

    mdl.to("cpu")
    mdl.eval()

    # ensure consistent labels (your dataset uses 0=not_depressed, 1=depressed)
    if not getattr(mdl.config, "id2label", None) or "LABEL_0" in str(mdl.config.id2label):
        mdl.config.id2label = {0: "not_depressed", 1: "depressed"}
        mdl.config.label2id = {"not_depressed": 0, "depressed": 1}

    return mdl, tok

model, tokenizer = load_model_and_tokenizer()

st.set_page_config(page_title="Depression Detector", page_icon="🧠", layout="centered")
st.title("🧠 Depression Detection Web App (DistilBERT Powered)")
st.write("Enter text below and the AI will analyze emotional and mental health indication:")

user_input = st.text_area("Type your text here...", height=150)

if st.button("Analyze"):
    txt = user_input.strip()
    if not txt:
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu()

        pred_id = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs).item() * 100)
        label = model.config.id2label[pred_id]

        if label == "depressed":
            st.error(f"⚠️ Depression Likely — Confidence: {conf:.2f}%")
        else:
            st.success(f"✅ No Depression Detected — Confidence: {conf:.2f}%")
