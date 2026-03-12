"""
ШАГ 5 — Streamlit v2 с Ensemble + TTA инференсом
==================================================
Запустить: streamlit run step5_app_v2.py
"""

import json, time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ── Auto-download weights from Google Drive if missing ────────────────────
@st.cache_resource(show_spinner=False)
def init_weights():
    try:
        from download_weights import ensure_weights
        return ensure_weights()
    except Exception:
        return False

MODELS_DIR    = Path("models")
PRICE_DB_PATH = Path("price_database.json")
IMG_SIZE      = 224
DEVICE        = "cpu"

st.set_page_config(
    page_title="🛒 Smart Price Estimator v2",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
  border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.hero {
  background: linear-gradient(135deg, #0d1117, #161b22, #1f2937);
  border: 1px solid #30363d; border-radius: 16px;
  padding: 28px 32px 22px; margin-bottom: 24px;
}
.hero-title { font-size: 1.9rem; font-weight: 900; color: #f0f6fc; margin: 0 0 4px; }
.hero-title span {
  background: linear-gradient(90deg, #58a6ff, #56d364);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #8b949e; font-size: 0.9rem; margin: 0 0 18px; }
.chips { display: flex; gap: 10px; flex-wrap: wrap; }
.chip {
  background: #21262d; border: 1px solid #30363d;
  border-radius: 20px; padding: 6px 14px;
  display: inline-flex; align-items: center; gap: 6px;
}
.chip .clabel { font-size: 10px; color: #8b949e; font-weight: 700; text-transform: uppercase; }
.chip .cval   { font-size: 13px; color: #f0f6fc; font-weight: 800; }
.chip.hi { border-color: #56d364; }
.chip.hi .cval { color: #56d364; }
.chip.gold { border-color: #e3b341; }
.chip.gold .cval { color: #e3b341; }
.pred-cat   { font-size: 22px; font-weight: 800; color: #f0f6fc; margin-bottom: 6px; }
.conf-green  { color: #56d364; font-weight: 700; font-size: 14px; }
.conf-yellow { color: #e3b341; font-weight: 700; font-size: 14px; }
.conf-red    { color: #f85149; font-weight: 700; font-size: 14px; }
.price-tag  { font-size: 30px; font-weight: 900; color: #58a6ff; margin: 10px 0 6px; }
.infer-time { font-size: 11px; color: #6e7681; }
.receipt {
  background: linear-gradient(135deg, #0d1117, #1c2128);
  border: 2px solid #56d364; border-radius: 16px;
  padding: 28px; text-align: center; margin-top: 24px;
}
.receipt-label { font-size: 11px; color: #56d364; letter-spacing: 3px;
                 font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
.receipt-total { font-size: 52px; font-weight: 900; color: #f0f6fc; }
.receipt-count { font-size: 14px; color: #8b949e; margin-top: 6px; }
.upload-zone {
  border: 2px dashed #30363d; border-radius: 16px;
  padding: 60px 20px; text-align: center;
  background: #0d1117; color: #6e7681;
}
.stitle {
  font-size: 11px; font-weight: 700; color: #8b949e;
  text-transform: uppercase; letter-spacing: 2px;
  margin: 24px 0 12px;
  display: flex; align-items: center; gap: 10px;
}
.stitle::after { content:''; flex:1; height:1px; background:#30363d; }
.model-badge {
  background: linear-gradient(135deg, #1f6feb, #388bfd);
  border-radius: 10px; padding: 10px 16px; text-align: center;
  font-weight: 800; font-size: 15px; color: white; margin-top: 10px;
  box-shadow: 0 0 20px rgba(31,111,235,0.3);
}
.ensemble-badge {
  background: linear-gradient(135deg, #238636, #2ea043);
  border-radius: 10px; padding: 8px 14px; text-align: center;
  font-weight: 700; font-size: 13px; color: white; margin-top: 8px;
}
.mt-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0; border-bottom: 1px solid #21262d;
}
.mt-row:last-child { border-bottom: none; }
.mt-lbl { color: #8b949e; font-size: 12px; }
.mt-val { color: #f0f6fc; font-size: 13px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_price_db():
    with open(PRICE_DB_PATH, encoding="utf-8") as f: return json.load(f)

@st.cache_data
def load_all_metrics():
    p = MODELS_DIR / "all_metrics.json"
    if not p.exists(): return []
    with open(p) as f: return json.load(f)

@st.cache_data
def get_available_models():
    if not MODELS_DIR.exists(): return []
    return [d.name for d in sorted(MODELS_DIR.iterdir())
            if d.is_dir() and (d/"best_model.pth").exists()]

@st.cache_resource
def load_single_model(name):
    path = MODELS_DIR / name / "best_model.pth"
    if not path.exists(): return None, None
    ckpt = torch.load(path, map_location=DEVICE)
    cn   = ckpt["class_names"]; nc = len(cn)
    if name == "alexnet":
        m = models.alexnet(weights=None); m.classifier[6] = nn.Linear(4096, nc)
    elif name == "vgg16":
        m = models.vgg16(weights=None);   m.classifier[6] = nn.Linear(4096, nc)
    elif name == "googlenet":
        m = models.googlenet(weights=None, aux_logits=False); m.fc = nn.Linear(1024, nc)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, nc))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = models.efficientnet_b0().classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, nc))
    else: return None, None
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, cn

@st.cache_resource
def load_all_models():
    names = ["resnet50","vgg16","googlenet","alexnet","efficientnet_b0"]
    return {n: load_single_model(n) for n in names}

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
NORMALIZE = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
BASE_CROP  = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG_SIZE)])

def make_tf(*extra):
    return transforms.Compose([BASE_CROP, *extra, transforms.ToTensor(), NORMALIZE])

TTA_TFS = [
    make_tf(),
    make_tf(transforms.RandomHorizontalFlip(p=1.0)),
    make_tf(transforms.RandomRotation(degrees=(10,10))),
    make_tf(transforms.RandomRotation(degrees=(-10,-10))),
    make_tf(transforms.ColorJitter(brightness=0.3)),
    make_tf(transforms.ColorJitter(contrast=0.3)),
    make_tf(transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0))),
    make_tf(transforms.Pad(20, padding_mode="reflect"),
            transforms.CenterCrop(IMG_SIZE)),
]

MODEL_WEIGHTS = {
    "resnet50": 0.40, "vgg16": 0.25, "googlenet": 0.15,
    "alexnet": 0.10, "efficientnet_b0": 0.10,
}

def infer_probs(model, img_rgb, use_tta: bool) -> np.ndarray:
    tfs = TTA_TFS if use_tta else [TTA_TFS[0]]
    all_p = []
    for tf in tfs:
        try:
            t = tf(img_rgb).unsqueeze(0)
            with torch.no_grad():
                out = model(t)
                if hasattr(out,"logits"): out = out.logits
                all_p.append(torch.softmax(out,1)[0].numpy())
        except Exception:
            pass
    return np.mean(all_p, axis=0)

def predict_best(img: Image.Image, mode: str, single_model_name: str = "resnet50"):
    """
    mode:
      'single'       — один ResNet50, без TTA
      'tta'          — один ResNet50 + TTA
      'ensemble_tta' — все модели + TTA (лучшее качество)
    """
    img_rgb = img.convert("RGB")
    t0 = time.time()

    all_models = load_all_models()

    if mode == "single":
        m, cn = all_models[single_model_name]
        probs = infer_probs(m, img_rgb, use_tta=False)
        cn_ref = cn
    elif mode == "tta":
        m, cn = all_models[single_model_name]
        probs = infer_probs(m, img_rgb, use_tta=True)
        cn_ref = cn
    else:  # ensemble_tta
        weighted = None; total_w = 0.0; cn_ref = None
        for name, (m, cn) in all_models.items():
            if m is None: continue
            w = MODEL_WEIGHTS.get(name, 0.1)
            p = infer_probs(m, img_rgb, use_tta=True)
            weighted = p*w if weighted is None else weighted + p*w
            total_w += w
            if cn_ref is None: cn_ref = cn
        probs = weighted / total_w

    ms   = (time.time()-t0)*1000
    idx5 = np.argsort(probs)[::-1][:5]
    return [cn_ref[i] for i in idx5], [float(probs[i]) for i in idx5], ms

def get_price(key, pdb):
    cat = pdb["categories"].get(key,{})
    return cat.get("name_ru", key), cat.get("avg_price", 0)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:16px 0 8px;'>
          <div style='font-size:36px;'>🛒</div>
          <div style='font-size:16px;font-weight:800;color:#f0f6fc;'>Smart Price v2</div>
          <div style='font-size:11px;color:#6e7681;'>CNN + TTA + Ensemble</div>
        </div>
        <hr style='border-color:#30363d;margin:8px 0;'>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:10px;color:#6e7681;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
                    "Navigation</div>", unsafe_allow_html=True)
        page = st.radio("", [" 1. PREDICTION & PRICE", " 2. EVALUATION METRICS"],
                        label_visibility="collapsed")

        st.markdown("<hr style='border-color:#30363d;margin:12px 0;'>", unsafe_allow_html=True)

        # Inference mode selector
        st.markdown("<div style='font-size:10px;color:#6e7681;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
                    "Inference Mode</div>", unsafe_allow_html=True)

        mode = st.radio("", [
            "🚀 Single Model (быстро)",
            "🎯 Single + TTA (точнее)",
            "🏆 Ensemble + TTA (лучшее)",
        ], index=2, label_visibility="collapsed")

        mode_map = {
            "🚀 Single Model (быстро)":    "single",
            "🎯 Single + TTA (точнее)":    "tta",
            "🏆 Ensemble + TTA (лучшее)":  "ensemble_tta",
        }
        inference_mode = mode_map[mode]

        # Speed / quality hints
        hints = {
            "single":       ("~30ms",  "Baseline"),
            "tta":          ("~250ms", "+TTA ×8"),
            "ensemble_tta": ("~1.2s",  "5 models ×8 TTA"),
        }
        spd, qual = hints[inference_mode]
        st.markdown(f"""
        <div style='background:#21262d;border:1px solid #30363d;border-radius:8px;
                    padding:10px 14px;margin-top:6px;font-size:12px;'>
          <span style='color:#8b949e;'>⚡ Speed: </span>
          <span style='color:#e3b341;font-weight:700;'>{spd}</span><br>
          <span style='color:#8b949e;'>✨ Mode: </span>
          <span style='color:#56d364;font-weight:700;'>{qual}</span>
        </div>""", unsafe_allow_html=True)

        # Model selector — shown for single / tta modes
        st.markdown("<hr style='border-color:#30363d;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:10px;color:#6e7681;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
                    "Select Model</div>", unsafe_allow_html=True)

        available = get_available_models()
        if not available:
            st.error("No models found."); return None, None, page

        best_txt = MODELS_DIR / "best_model_name.txt"
        default  = best_txt.read_text().strip() if best_txt.exists() else available[0]
        if default not in available: default = available[0]

        # In ensemble mode show info instead of selector
        if inference_mode == "ensemble_tta":
            st.markdown("""
            <div style='background:#21262d;border:1px solid #238636;border-radius:8px;
                        padding:10px 14px;font-size:12px;color:#8b949e;'>
              🏆 <span style='color:#56d364;font-weight:700;'>All 5 models active</span><br>
              <span style='font-size:11px;'>ResNet50 · VGG16 · GoogLeNet<br>
              AlexNet · EfficientNet-B0</span>
            </div>""", unsafe_allow_html=True)
            selected_model = default  # not used but kept for hero display
        else:
            labels = {"alexnet":"AlexNet","vgg16":"VGG-16","googlenet":"GoogLeNet",
                      "resnet50":"ResNet-50","efficientnet_b0":"EfficientNet-B0"}
            selected_model = st.selectbox("", available,
                                          index=available.index(default),
                                          label_visibility="collapsed")
            st.markdown(f"<div class='model-badge'>✦ {labels.get(selected_model, selected_model)}</div>",
                        unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#30363d;margin:12px 0;'>", unsafe_allow_html=True)

        # Model metrics summary
        all_m  = load_all_metrics()
        best_m = next((m for m in all_m if m["model_name"]=="resnet50"), None)
        if best_m:
            st.markdown("<div style='font-size:10px;color:#6e7681;font-weight:700;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
                        "Best Model Stats (ResNet50)</div>", unsafe_allow_html=True)
            for lbl, val in [
                ("Top-1",     f"{best_m['top1_accuracy']*100:.1f}%"),
                ("Top-5",     f"{best_m['top5_accuracy']*100:.1f}%"),
                ("Price MAE", f"{best_m['price_mae']:.0f} тг"),
                ("Size",      f"{best_m['model_size_mb']} MB"),
            ]:
                st.markdown(f"<div class='mt-row'>"
                            f"<span class='mt-lbl'>{lbl}</span>"
                            f"<span class='mt-val'>{val}</span></div>",
                            unsafe_allow_html=True)

        return inference_mode, selected_model, page

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
def render_hero(mode, selected_model):
    labels = {"alexnet":"AlexNet","vgg16":"VGG-16","googlenet":"GoogLeNet",
              "resnet50":"ResNet-50","efficientnet_b0":"EfficientNet-B0"}
    mode_labels = {
        "single":       (f"Single {labels.get(selected_model, selected_model)}", "#8b949e"),
        "tta":          (f"{labels.get(selected_model, selected_model)} + TTA ×8", "#e3b341"),
        "ensemble_tta": ("Ensemble 5 CNNs + TTA ×8", "#56d364"),
    }
    mlabel, mcolor = mode_labels[mode]
    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">🛒 <span>CNN Product Recognition</span> & Price Estimation</div>
      <div class="hero-sub">Upload product photos → AI identifies each item → Calculates total cost</div>
      <div class="chips">
        <div class="chip hi"><span class="clabel">TOP-1</span><span class="cval">98.3%</span></div>
        <div class="chip hi"><span class="clabel">TOP-5</span><span class="cval">99.7%</span></div>
        <div class="chip"><span class="clabel">MODELS</span><span class="cval">5 CNNs</span></div>
        <div class="chip"><span class="clabel">CLASSES</span><span class="cval">16</span></div>
        <div class="chip gold">
          <span class="clabel">MODE</span>
          <span class="cval" style="color:{mcolor};">{mlabel}</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def page_predict(mode, pdb, selected_model="resnet50"):
    import plotly.graph_objects as go
    import base64, io

    labels = {"alexnet":"AlexNet","vgg16":"VGG-16","googlenet":"GoogLeNet",
              "resnet50":"ResNet-50","efficientnet_b0":"EfficientNet-B0"}
    sel_label = labels.get(selected_model, selected_model)

    uploaded = st.file_uploader(
        "Upload product image(s)",
        type=["jpg","jpeg","png","webp","heic"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if not uploaded:
        st.markdown("""
        <div class="upload-zone">
          <div style='font-size:40px;margin-bottom:12px;'>📦</div>
          <div style='font-size:18px;font-weight:700;color:#c9d1d9;margin-bottom:6px;'>
            Drop product photos here
          </div>
          <div style='font-size:13px;'>JPG · PNG · WEBP · HEIC &nbsp;|&nbsp; up to 200MB per file</div>
        </div>""", unsafe_allow_html=True)
        return

    spinner_msg = {
        "single":       f"🤖 Running {sel_label}...",
        "tta":          f"🎯 Running {sel_label} + TTA ×8...",
        "ensemble_tta": "🏆 Running Ensemble of 5 CNNs × 8 TTA each...",
    }

    results = []
    with st.spinner(spinner_msg[mode]):
        for i, ufile in enumerate(uploaded):
            img = Image.open(ufile).convert("RGB")
            top5_lbl, top5_prob, ms = predict_best(img, mode, selected_model)
            cat_name, price = get_price(top5_lbl[0], pdb)
            buf = io.BytesIO()
            thumb = img.copy(); thumb.thumbnail((400,400))
            thumb.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            results.append({
                "idx": i, "file": ufile.name, "img": img, "b64": b64,
                "cat_name": cat_name, "conf": top5_prob[0],
                "price": price, "top5_lbl": top5_lbl,
                "top5_prob": top5_prob, "ms": ms,
            })

    total = sum(r["price"] for r in results)
    st.markdown(f"<div class='stitle'>🔍 Results — {len(results)} product(s) detected</div>",
                unsafe_allow_html=True)

    for r in results:
        conf_pct = r["conf"] * 100
        if conf_pct >= 70:
            conf_class, icon, btype = "conf-green",  "✅", "success"
        elif conf_pct >= 40:
            conf_class, icon, btype = "conf-yellow", "⚠️", "warning"
        else:
            conf_class, icon, btype = "conf-red",    "❓", "error"

        if btype == "success":
            st.success(f"✅ **{r['cat_name']}** — High confidence ({conf_pct:.1f}%)")
        elif btype == "warning":
            st.warning(f"⚠️ **{r['cat_name']}** — Medium confidence ({conf_pct:.1f}%)")
        else:
            st.error(f"❓ **{r['cat_name']}** — Low confidence ({conf_pct:.1f}%)")

        col_img, col_pred, col_chart = st.columns([1, 1.1, 2])

        with col_img:
            st.markdown(f"""
            <div style='background:#0d1117;border:1px solid #30363d;border-radius:12px;
                        padding:12px;height:220px;display:flex;align-items:center;
                        justify-content:center;overflow:hidden;'>
              <img src="data:image/jpeg;base64,{r['b64']}"
                   style='max-width:100%;max-height:196px;object-fit:contain;border-radius:6px;'>
            </div>
            <div style='font-size:11px;color:#6e7681;text-align:center;margin-top:6px;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>
              {r["file"]}
            </div>""", unsafe_allow_html=True)

        with col_pred:
            st.markdown(f"""
            <div style='background:#161b22;border:1px solid #30363d;border-radius:12px;
                        padding:20px;height:220px;display:flex;flex-direction:column;
                        justify-content:center;'>
              <div class='pred-cat'>{r["cat_name"]}</div>
              <div class='{conf_class}'>{icon} {conf_pct:.1f}% confidence</div>
              <div class='price-tag'>💰 {r["price"]:,} тг</div>
              <div class='infer-time'>⏱ {r["ms"]:.0f} ms &nbsp;·&nbsp; item #{r["idx"]+1}</div>
            </div>""", unsafe_allow_html=True)
            st.progress(min(conf_pct/100, 1.0))

        with col_chart:
            top5_names = [pdb["categories"].get(l,{}).get("name_ru",l) for l in r["top5_lbl"]]
            pairs  = sorted(zip(top5_names, r["top5_prob"]), key=lambda x: x[1])
            s_names = [p[0] for p in pairs]
            s_probs = [p[1] for p in pairs]
            colors  = ["#56d364" if j==len(pairs)-1 else "#30363d" for j in range(len(pairs))]

            fig = go.Figure(go.Bar(
                y=s_names, x=[p*100 for p in s_probs],
                orientation="h",
                marker=dict(color=colors, line=dict(color="#21262d",width=1)),
                text=[f"<b>{p*100:.1f}%</b>" for p in s_probs],
                textposition="outside",
                textfont=dict(size=12, color="#c9d1d9"),
            ))
            fig.update_layout(
                height=220, margin=dict(l=0,r=60,t=10,b=10),
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                xaxis=dict(title="Confidence (%)", range=[0,120],
                           tickfont=dict(color="#8b949e",size=10),
                           gridcolor="#21262d", zeroline=False),
                yaxis=dict(tickfont=dict(color="#e6edf3",size=12),
                           gridcolor="#21262d"),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{r['idx']}")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="receipt">
      <div class="receipt-label">🧾 Total Receipt</div>
      <div class="receipt-total">{total:,} тг</div>
      <div class="receipt-count">{len(results)} product(s) · mode: {mode}</div>
    </div>""", unsafe_allow_html=True)

    if len(results) >= 1:
        import pandas as pd
        st.markdown("<div class='stitle' style='margin-top:24px;'>📋 Itemized Bill</div>",
                    unsafe_allow_html=True)
        df = pd.DataFrame([{
            "#": r["idx"]+1, "File": r["file"],
            "Category": r["cat_name"],
            "Confidence": f"{r['conf']*100:.1f}%",
            "Price (тг)": r["price"],
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: METRICS
# ─────────────────────────────────────────────────────────────────────────────
def page_metrics():
    import pandas as pd
    import plotly.graph_objects as go

    st.markdown("<div class='stitle'>📊 Model Comparison Dashboard</div>",
                unsafe_allow_html=True)
    all_m = load_all_metrics()
    if not all_m:
        st.warning("No metrics found. Run step2_train.py"); return

    all_m_s = sorted(all_m, key=lambda x: x["top1_accuracy"], reverse=True)
    df = pd.DataFrame([{
        "Model":          m["model_name"],
        "Top-1 (%)":      round(m["top1_accuracy"]*100,1),
        "Top-5 (%)":      round(m["top5_accuracy"]*100,1),
        "MAE (тг)":       m["price_mae"],
        "RMSE (тг)":      m["price_rmse"],
        "Train Time (s)": m["train_time_sec"],
        "Inference (ms)": m["inference_ms"],
        "Size (MB)":      m["model_size_mb"],
    } for m in all_m_s])

    st.subheader("📋 Summary Table — sorted by Top-1 ↓")
    st.dataframe(df.set_index("Model"), use_container_width=True)
    st.divider()

    DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3", family="Inter"))
    def ax(): return dict(tickfont=dict(color="#c9d1d9",size=12),
                          title_font=dict(color="#c9d1d9",size=13),
                          gridcolor="#21262d", zeroline=False)

    df_acc = df.sort_values("Top-1 (%)", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Top-1", x=df_acc["Model"], y=df_acc["Top-1 (%)"],
                             marker_color="#58a6ff",
                             text=[f"<b>{v}%</b>" for v in df_acc["Top-1 (%)"]],
                             textposition="outside", textfont=dict(color="#58a6ff",size=12)))
        fig.add_trace(go.Bar(name="Top-5", x=df_acc["Model"], y=df_acc["Top-5 (%)"],
                             marker_color="#56d364",
                             text=[f"<b>{v}%</b>" for v in df_acc["Top-5 (%)"]],
                             textposition="outside", textfont=dict(color="#56d364",size=12)))
        fig.update_layout(**DARK,
            title=dict(text="📈 Classification Accuracy ↓",
                       font=dict(color="#f0f6fc",size=15)),
            barmode="group", yaxis=dict(**ax(), range=[88,102]),
            xaxis=dict(**ax()), legend=dict(font=dict(color="#c9d1d9")),
            margin=dict(t=50,b=10))
        st.plotly_chart(fig, use_container_width=True, key="acc")

    df_p = df.sort_values("MAE (тг)", ascending=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="MAE", x=df_p["Model"], y=df_p["MAE (тг)"],
                             marker_color="#56d364",
                             text=[f"<b>{v}</b>" for v in df_p["MAE (тг)"]],
                             textposition="outside", textfont=dict(color="#56d364",size=12)))
        fig.add_trace(go.Bar(name="RMSE", x=df_p["Model"], y=df_p["RMSE (тг)"],
                             marker_color="#f85149",
                             text=[f"<b>{v}</b>" for v in df_p["RMSE (тг)"]],
                             textposition="outside", textfont=dict(color="#f85149",size=12)))
        fig.update_layout(**DARK,
            title=dict(text="💸 Price Error — lower is better ↑",
                       font=dict(color="#f0f6fc",size=15)),
            barmode="group", yaxis=dict(**ax()),
            xaxis=dict(**ax()), legend=dict(font=dict(color="#c9d1d9")),
            margin=dict(t=50,b=10))
        st.plotly_chart(fig, use_container_width=True, key="price")

    df_t = df.sort_values("Train Time (s)", ascending=False)
    fig = go.Figure(go.Bar(
        x=df_t["Model"], y=df_t["Train Time (s)"],
        marker_color=["#58a6ff","#56d364","#e3b341","#f85149","#a371f7"][:len(df_t)],
        text=[f"<b>{v}s</b>" for v in df_t["Train Time (s)"]],
        textposition="outside", textfont=dict(color="#c9d1d9",size=12),
    ))
    fig.update_layout(**DARK,
        title=dict(text="⏱ Training Time in Seconds ↓",
                   font=dict(color="#f0f6fc",size=15)),
        yaxis=dict(**ax()), xaxis=dict(**ax()),
        showlegend=False, margin=dict(t=50,b=10))
    st.plotly_chart(fig, use_container_width=True, key="time")

    st.divider()
    st.subheader("🔀 Confusion Matrices")
    items = [(m, MODELS_DIR/m["model_name"]/"confusion_matrix.png")
             for m in all_m if (MODELS_DIR/m["model_name"]/"confusion_matrix.png").exists()]
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, (m, p) in enumerate(items[i:i+2]):
            with cols[j]:
                st.markdown(f"<div style='text-align:center;font-weight:700;color:#f0f6fc;"
                            f"margin-bottom:6px;'>{m['model_name'].upper()} — "
                            f"Top-1: {m['top1_accuracy']*100:.1f}%</div>",
                            unsafe_allow_html=True)
                st.image(str(p), use_container_width=True)

    st.divider()
    st.subheader("📈 Training Curves")
    tc = [(m, MODELS_DIR/m["model_name"]/"training_curves.png")
          for m in all_m if (MODELS_DIR/m["model_name"]/"training_curves.png").exists()]
    for i in range(0, len(tc), 2):
        cols = st.columns(2)
        for j, (m, p) in enumerate(tc[i:i+2]):
            with cols[j]:
                st.markdown(f"<div style='text-align:center;font-weight:700;color:#f0f6fc;"
                            f"margin-bottom:6px;'>{m['model_name'].upper()}</div>",
                            unsafe_allow_html=True)
                st.image(str(p), use_container_width=True)

    st.divider()
    best_txt = MODELS_DIR/"best_model_name.txt"
    if best_txt.exists():
        bname = best_txt.read_text().strip()
        bm = next((m for m in all_m if m["model_name"]==bname), None)
        if bm:
            st.subheader(f"🏆 F1-Score per Class — {bname.upper()} (Best Model)")
            rep  = bm.get("classification_report",{})
            clss = [k for k in rep if k not in ("accuracy","macro avg","weighted avg")]
            f1s  = [rep[c]["f1-score"] for c in clss]
            pairs = sorted(zip(clss,f1s), key=lambda x: x[1], reverse=True)
            sc = [p[0] for p in pairs]; sf = [p[1] for p in pairs]
            colors = ["#56d364" if v>=0.95 else "#e3b341" if v>=0.85 else "#f85149"
                      for v in sf]
            fig = go.Figure(go.Bar(
                x=sc, y=sf, marker_color=colors,
                text=[f"<b>{v:.2f}</b>" for v in sf],
                textposition="outside", textfont=dict(color="#c9d1d9",size=11),
            ))
            fig.update_layout(**DARK,
                title=dict(text="F1 per Class ↓  (🟢 ≥0.95 · 🟡 ≥0.85 · 🔴 <0.85)",
                           font=dict(color="#f0f6fc",size=14)),
                yaxis=dict(**ax(), range=[0,1.12]),
                xaxis=dict(**ax(), tickangle=-35),
                showlegend=False, margin=dict(t=55,b=10))
            st.plotly_chart(fig, use_container_width=True, key="f1")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("⬇️ Checking model weights..."):
        init_weights()
    inference_mode, selected_model, page = render_sidebar()
    render_hero(inference_mode, selected_model)
    if page == " 1. PREDICTION & PRICE":
        page_predict(inference_mode, load_price_db(), selected_model)
    else:
        page_metrics()

main()