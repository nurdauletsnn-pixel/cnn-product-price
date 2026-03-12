"""
ШАГ 3 — Streamlit приложение (финальный дизайн)
Запустить: streamlit run step3_app.py
"""

import json, time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODELS_DIR    = Path("models")
PRICE_DB_PATH = Path("price_database.json")
IMG_SIZE      = 224
DEVICE        = "cpu"

st.set_page_config(
    page_title="🛒 Smart Price Estimator",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
  border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Hero */
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

/* ── Uniform result card ── */
.rcard {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 14px;
  margin-bottom: 16px;
  overflow: hidden;
}
.rcard-inner {
  display: grid;
  grid-template-columns: 200px 1fr 1.8fr;
  gap: 0;
  align-items: stretch;
  min-height: 220px;
}
.rcard-img {
  background: #0d1117;
  border-right: 1px solid #30363d;
  display: flex; align-items: center; justify-content: center;
  padding: 12px;
}
.rcard-img img { width: 100%; height: 180px; object-fit: contain; border-radius: 8px; }
.rcard-pred {
  padding: 20px 18px;
  border-right: 1px solid #30363d;
  display: flex; flex-direction: column; justify-content: center;
}
.rcard-chart { padding: 12px 16px; display: flex; align-items: center; }

/* Prediction text */
.pred-cat   { font-size: 22px; font-weight: 800; color: #f0f6fc; margin-bottom: 6px; }
.conf-green { color: #56d364; font-weight: 700; font-size: 14px; }
.conf-yellow{ color: #e3b341; font-weight: 700; font-size: 14px; }
.conf-red   { color: #f85149; font-weight: 700; font-size: 14px; }
.price-tag  { font-size: 30px; font-weight: 900; color: #58a6ff; margin: 10px 0 6px; }
.infer-time { font-size: 11px; color: #6e7681; }
.conf-label { font-size: 11px; color: #6e7681; margin-top: 10px; margin-bottom: 3px; }

/* Receipt */
.receipt {
  background: linear-gradient(135deg, #0d1117, #1c2128);
  border: 2px solid #56d364; border-radius: 16px;
  padding: 28px; text-align: center; margin-top: 24px;
}
.receipt-label { font-size: 11px; color: #56d364; letter-spacing: 3px;
                 font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
.receipt-total { font-size: 52px; font-weight: 900; color: #f0f6fc; }
.receipt-count { font-size: 14px; color: #8b949e; margin-top: 6px; }

/* Upload */
.upload-zone {
  border: 2px dashed #30363d; border-radius: 16px;
  padding: 60px 20px; text-align: center;
  background: #0d1117; color: #6e7681;
}

/* Section title */
.stitle {
  font-size: 11px; font-weight: 700; color: #8b949e;
  text-transform: uppercase; letter-spacing: 2px;
  margin: 24px 0 12px;
  display: flex; align-items: center; gap: 10px;
}
.stitle::after { content:''; flex:1; height:1px; background:#30363d; }

/* Model badge */
.model-badge {
  background: linear-gradient(135deg, #1f6feb, #388bfd);
  border-radius: 10px; padding: 10px 16px; text-align: center;
  font-weight: 800; font-size: 15px; color: white; margin-top: 10px;
  box-shadow: 0 0 20px rgba(31,111,235,0.3);
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

# ── DATA ──────────────────────────────────────────────────────────────────────
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
def load_model(name):
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

TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict(model, class_names, img):
    t = TF(img.convert("RGB")).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        out = model(t)
        if hasattr(out,"logits"): out = out.logits
        probs = torch.softmax(out,1)[0].numpy()
    ms   = (time.time()-t0)*1000
    idx5 = np.argsort(probs)[::-1][:5]
    return [class_names[i] for i in idx5], [float(probs[i]) for i in idx5], ms

def get_price(key, pdb):
    cat = pdb["categories"].get(key,{})
    return cat.get("name_ru", key), cat.get("avg_price", 0)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:16px 0 8px;'>
          <div style='font-size:36px;'>🛒</div>
          <div style='font-size:16px;font-weight:800;color:#f0f6fc;'>Smart Price</div>
          <div style='font-size:11px;color:#6e7681;'>CNN Product Estimator</div>
        </div>
        <hr style='border-color:#30363d;margin:8px 0;'>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:10px;color:#6e7681;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
                    "Navigation</div>", unsafe_allow_html=True)
        page = st.radio("", [" 1. PREDICTION & PRICE", " 2. EVALUATION METRICS"],
                        label_visibility="collapsed")

        st.markdown("<hr style='border-color:#30363d;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:10px;color:#6e7681;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
                    "Select Model</div>", unsafe_allow_html=True)

        available = get_available_models()
        if not available:
            st.error("No models found. Run step2_train.py"); return None, page

        best_txt = MODELS_DIR/"best_model_name.txt"
        default  = best_txt.read_text().strip() if best_txt.exists() else available[0]
        if default not in available: default = available[0]

        selected = st.selectbox("", available,
                                index=available.index(default),
                                label_visibility="collapsed")
        labels = {"alexnet":"AlexNet","vgg16":"VGG-16","googlenet":"GoogLeNet",
                  "resnet50":"ResNet-50","efficientnet_b0":"EfficientNet-B0"}
        st.markdown(f"<div class='model-badge'>✦ {labels.get(selected,selected)}</div>",
                    unsafe_allow_html=True)

        all_m  = load_all_metrics()
        m_info = next((m for m in all_m if m["model_name"]==selected), None)
        if m_info:
            st.markdown("<hr style='border-color:#30363d;margin:14px 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:10px;color:#6e7681;font-weight:700;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
                        "Model Stats</div>", unsafe_allow_html=True)
            for lbl, val in [
                ("Top-1 Accuracy", f"{m_info['top1_accuracy']*100:.1f}%"),
                ("Top-5 Accuracy", f"{m_info['top5_accuracy']*100:.1f}%"),
                ("Price MAE",      f"{m_info['price_mae']:.0f} тг"),
                ("Price RMSE",     f"{m_info['price_rmse']:.0f} тг"),
                ("Model Size",     f"{m_info['model_size_mb']} MB"),
                ("Inference",      f"{m_info['inference_ms']} ms"),
            ]:
                st.markdown(f"<div class='mt-row'>"
                            f"<span class='mt-lbl'>{lbl}</span>"
                            f"<span class='mt-val'>{val}</span></div>",
                            unsafe_allow_html=True)
        return selected, page

# ── HERO ──────────────────────────────────────────────────────────────────────
def render_hero(sel):
    all_m  = load_all_metrics()
    mi     = next((m for m in all_m if m["model_name"]==sel), None)
    top1   = f"{mi['top1_accuracy']*100:.1f}%"  if mi else "—"
    top5   = f"{mi['top5_accuracy']*100:.1f}%"  if mi else "—"
    mae    = f"{mi['price_mae']:.0f} тг"         if mi else "—"
    rmse   = f"{mi['price_rmse']:.0f} тг"        if mi else "—"
    size   = f"{mi['model_size_mb']} MB"          if mi else "—"
    infer  = f"{mi['inference_ms']} ms"           if mi else "—"
    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">🛒 <span>CNN-Based Product Recognition</span> & Price Estimation</div>
      <div class="hero-sub">Upload a product photo → AI identifies the category → Shows the price instantly</div>
      <div class="chips">
        <div class="chip hi"><span class="clabel">TOP-1</span><span class="cval">{top1}</span></div>
        <div class="chip hi"><span class="clabel">TOP-5</span><span class="cval">{top5}</span></div>
        <div class="chip"><span class="clabel">MAE</span><span class="cval">{mae}</span></div>
        <div class="chip"><span class="clabel">RMSE</span><span class="cval">{rmse}</span></div>
        <div class="chip"><span class="clabel">SIZE</span><span class="cval">{size}</span></div>
        <div class="chip"><span class="clabel">SPEED</span><span class="cval">{infer}</span></div>
        <div class="chip"><span class="clabel">MODEL</span><span class="cval">{sel.upper()}</span></div>
        <div class="chip"><span class="clabel">CLASSES</span><span class="cval">16</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── PAGE 1: PREDICT ───────────────────────────────────────────────────────────
def page_predict(sel, pdb):
    import plotly.graph_objects as go
    import base64, io

    model, class_names = load_model(sel)
    if model is None:
        st.error(f"Cannot load model '{sel}'"); return

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

    results = []
    with st.spinner("🤖 Analysing images..."):
        for i, ufile in enumerate(uploaded):
            img = Image.open(ufile).convert("RGB")
            top5_lbl, top5_prob, ms = predict(model, class_names, img)
            cat_name, price = get_price(top5_lbl[0], pdb)
            # encode image to base64 for HTML embedding
            buf = io.BytesIO()
            img.thumbnail((400, 400))
            img.save(buf, format="JPEG", quality=85)
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
            conf_class, icon, banner_col = "conf-green",  "✅", "success"
        elif conf_pct >= 40:
            conf_class, icon, banner_col = "conf-yellow", "⚠️", "warning"
        else:
            conf_class, icon, banner_col = "conf-red",    "❓", "error"

        # Feedback banner
        if banner_col == "success":
            st.success(f"✅ **{r['cat_name']}** — High confidence ({conf_pct:.1f}%)")
        elif banner_col == "warning":
            st.warning(f"⚠️ **{r['cat_name']}** — Medium confidence ({conf_pct:.1f}%)")
        else:
            st.error(f"❓ **{r['cat_name']}** — Low confidence ({conf_pct:.1f}%)")

        # Uniform 3-column card
        col_img, col_pred, col_chart = st.columns([1, 1.1, 2])

        with col_img:
            # Fixed height container for image
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
              <div class='infer-time'>⏱ {r["ms"]:.1f} ms &nbsp;·&nbsp; item #{r["idx"]+1}</div>
            </div>""", unsafe_allow_html=True)

            st.progress(min(conf_pct / 100, 1.0))

        with col_chart:
            top5_names = [pdb["categories"].get(l,{}).get("name_ru",l) for l in r["top5_lbl"]]
            # Sort descending by confidence
            pairs = sorted(zip(top5_names, r["top5_prob"]), key=lambda x: x[1])
            s_names = [p[0] for p in pairs]
            s_probs = [p[1] for p in pairs]
            colors  = ["#56d364" if j == len(pairs)-1 else "#30363d" for j in range(len(pairs))]

            fig = go.Figure(go.Bar(
                y=s_names, x=[p*100 for p in s_probs],
                orientation="h",
                marker=dict(color=colors, line=dict(color="#21262d", width=1)),
                text=[f"<b>{p*100:.1f}%</b>" for p in s_probs],
                textposition="outside",
                textfont=dict(size=12, color="#c9d1d9"),
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=0, r=60, t=10, b=10),
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                xaxis=dict(
                    title="Confidence (%)", range=[0,120],
                    tickfont=dict(color="#8b949e", size=10),
                    gridcolor="#21262d", zeroline=False,
                ),
                yaxis=dict(
                    tickfont=dict(color="#e6edf3", size=12),
                    gridcolor="#21262d",
                ),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{r['idx']}")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Receipt
    st.markdown(f"""
    <div class="receipt">
      <div class="receipt-label">🧾 Total Receipt</div>
      <div class="receipt-total">{total:,} тг</div>
      <div class="receipt-count">{len(results)} product(s) scanned</div>
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

# ── PAGE 2: METRICS ───────────────────────────────────────────────────────────
def page_metrics():
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("<div class='stitle'>📊 Model Comparison Dashboard</div>",
                unsafe_allow_html=True)
    all_m = load_all_metrics()
    if not all_m:
        st.warning("No metrics found. Run step2_train.py"); return

    # Sort by Top-1 descending
    all_m_sorted = sorted(all_m, key=lambda x: x["top1_accuracy"], reverse=True)

    df = pd.DataFrame([{
        "Model":           m["model_name"],
        "Top-1 (%)":       round(m["top1_accuracy"]*100, 1),
        "Top-5 (%)":       round(m["top5_accuracy"]*100, 1),
        "MAE (тг)":        m["price_mae"],
        "RMSE (тг)":       m["price_rmse"],
        "Train Time (s)":  m["train_time_sec"],
        "Inference (ms)":  m["inference_ms"],
        "Size (MB)":       m["model_size_mb"],
    } for m in all_m_sorted])

    # Summary table sorted descending by Top-1
    st.subheader("📋 Summary Table — sorted by Top-1 ↓")
    st.dataframe(df.set_index("Model"), use_container_width=True)
    st.divider()

    DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3", family="Inter"))

    def axis_style():
        return dict(
            tickfont=dict(color="#c9d1d9", size=12),
            title_font=dict(color="#c9d1d9", size=13),
            gridcolor="#21262d", zeroline=False,
        )

    # Accuracy chart — sorted descending
    df_acc = df.sort_values("Top-1 (%)", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Top-1", x=df_acc["Model"], y=df_acc["Top-1 (%)"],
            marker_color="#58a6ff",
            text=[f"<b>{v}%</b>" for v in df_acc["Top-1 (%)"]],
            textposition="outside", textfont=dict(color="#58a6ff", size=12),
        ))
        fig.add_trace(go.Bar(
            name="Top-5", x=df_acc["Model"], y=df_acc["Top-5 (%)"],
            marker_color="#56d364",
            text=[f"<b>{v}%</b>" for v in df_acc["Top-5 (%)"]],
            textposition="outside", textfont=dict(color="#56d364", size=12),
        ))
        fig.update_layout(
            **DARK, title=dict(text="📈 Classification Accuracy (sorted ↓)",
                               font=dict(color="#f0f6fc", size=15)),
            barmode="group", yaxis=dict(**axis_style(), range=[88, 102]),
            xaxis=dict(**axis_style()), legend=dict(font=dict(color="#c9d1d9")),
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, key="acc_chart")

    # Price error chart — sorted ascending (lower = better)
    df_price = df.sort_values("MAE (тг)", ascending=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="MAE", x=df_price["Model"], y=df_price["MAE (тг)"],
            marker_color="#56d364",
            text=[f"<b>{v}</b>" for v in df_price["MAE (тг)"]],
            textposition="outside", textfont=dict(color="#56d364", size=12),
        ))
        fig.add_trace(go.Bar(
            name="RMSE", x=df_price["Model"], y=df_price["RMSE (тг)"],
            marker_color="#f85149",
            text=[f"<b>{v}</b>" for v in df_price["RMSE (тг)"]],
            textposition="outside", textfont=dict(color="#f85149", size=12),
        ))
        fig.update_layout(
            **DARK, title=dict(text="💸 Price Prediction Error — lower is better ↑",
                               font=dict(color="#f0f6fc", size=15)),
            barmode="group", yaxis=dict(**axis_style()),
            xaxis=dict(**axis_style()), legend=dict(font=dict(color="#c9d1d9")),
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, key="price_chart")

    # Training time — sorted descending
    df_time = df.sort_values("Train Time (s)", ascending=False)
    fig = go.Figure(go.Bar(
        x=df_time["Model"], y=df_time["Train Time (s)"],
        marker_color=["#58a6ff","#56d364","#e3b341","#f85149","#a371f7"][:len(df_time)],
        text=[f"<b>{v}s</b>" for v in df_time["Train Time (s)"]],
        textposition="outside", textfont=dict(color="#c9d1d9", size=12),
    ))
    fig.update_layout(
        **DARK, title=dict(text="⏱ Training Time in Seconds (sorted ↓)",
                           font=dict(color="#f0f6fc", size=15)),
        yaxis=dict(**axis_style()), xaxis=dict(**axis_style()),
        showlegend=False, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="time_chart")

    st.divider()

    # Confusion matrices — 2 per row for readability
    st.subheader("🔀 Confusion Matrices")
    items = [(m, MODELS_DIR/m["model_name"]/"confusion_matrix.png")
             for m in all_m if (MODELS_DIR/m["model_name"]/"confusion_matrix.png").exists()]
    for row_start in range(0, len(items), 2):
        cols = st.columns(2)
        for j, (m, p) in enumerate(items[row_start:row_start+2]):
            with cols[j]:
                st.markdown(f"<div style='text-align:center;font-weight:700;"
                            f"color:#f0f6fc;margin-bottom:6px;'>{m['model_name'].upper()}"
                            f" &nbsp; Top-1: {m['top1_accuracy']*100:.1f}%</div>",
                            unsafe_allow_html=True)
                st.image(str(p), use_container_width=True)

    st.divider()

    # Training curves — 2 per row
    st.subheader("📈 Training Curves")
    tc_items = [(m, MODELS_DIR/m["model_name"]/"training_curves.png")
                for m in all_m if (MODELS_DIR/m["model_name"]/"training_curves.png").exists()]
    for row_start in range(0, len(tc_items), 2):
        cols = st.columns(2)
        for j, (m, p) in enumerate(tc_items[row_start:row_start+2]):
            with cols[j]:
                st.markdown(f"<div style='text-align:center;font-weight:700;"
                            f"color:#f0f6fc;margin-bottom:6px;'>{m['model_name'].upper()}</div>",
                            unsafe_allow_html=True)
                st.image(str(p), use_container_width=True)

    st.divider()

    # F1 per class — sorted descending
    best_txt = MODELS_DIR/"best_model_name.txt"
    if best_txt.exists():
        bname = best_txt.read_text().strip()
        bm    = next((m for m in all_m if m["model_name"]==bname), None)
        if bm:
            st.subheader(f"🏆 F1-Score per Class — {bname.upper()} (Best Model)")
            rep  = bm.get("classification_report",{})
            clss = [k for k in rep if k not in ("accuracy","macro avg","weighted avg")]
            f1s  = [rep[c]["f1-score"] for c in clss]
            # Sort descending
            pairs = sorted(zip(clss, f1s), key=lambda x: x[1], reverse=True)
            s_cls = [p[0] for p in pairs]
            s_f1s = [p[1] for p in pairs]
            colors = ["#56d364" if v >= 0.95 else "#e3b341" if v >= 0.85 else "#f85149"
                      for v in s_f1s]
            fig = go.Figure(go.Bar(
                x=s_cls, y=s_f1s,
                marker_color=colors,
                text=[f"<b>{v:.2f}</b>" for v in s_f1s],
                textposition="outside",
                textfont=dict(color="#c9d1d9", size=11),
            ))
            fig.update_layout(
                **DARK,
                title=dict(text="F1-Score per Class — sorted ↓ (green ≥ 0.95, yellow ≥ 0.85, red < 0.85)",
                           font=dict(color="#f0f6fc", size=14)),
                yaxis=dict(**axis_style(), range=[0, 1.12]),
                xaxis=dict(**axis_style(), tickangle=-35),
                showlegend=False, margin=dict(t=55, b=10),
            )
            st.plotly_chart(fig, use_container_width=True, key="f1_chart")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    result = render_sidebar()
    if result is None or result[0] is None:
        st.error("No models found. Run step2_train.py first."); return
    selected_model, page = result
    render_hero(selected_model)
    if page == " 1. PREDICTION & PRICE":
        page_predict(selected_model, load_price_db())
    else:
        page_metrics()

if __name__ == "__main__":
    main()