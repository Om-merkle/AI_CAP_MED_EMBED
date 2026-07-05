"""Streamlit UI for the medical embedding fine-tuning pipeline.

Talks to the FastAPI backend over HTTP (set API_URL if it is not on localhost:8000).
Walk top-to-bottom: configure -> prepare data -> collect triplets -> benchmark models
-> baseline -> train -> evaluate -> compare (with a before/after chart).
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Medical Embedding Fine-Tuning", page_icon="🩺", layout="wide")
st.title("🩺 Medical Embedding Fine-Tuning — end to end")
st.caption(f"Backend: {API_URL}")


def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=30).json()


def api_post(path: str, json_body: dict | None = None):
    return requests.post(f"{API_URL}{path}", json=json_body, timeout=30).json()


def run_job(path: str, label: str) -> dict | None:
    """Launch a background job and poll until it finishes, showing a spinner."""
    ref = api_post(path)
    job_id = ref.get("job_id")
    if not job_id:
        st.error(f"Could not start {label}: {ref}")
        return None
    with st.spinner(f"{label} running (job {job_id}) …"):
        while True:
            job = api_get(f"/jobs/{job_id}")
            if job["status"] == "done":
                st.success(f"{label} done ✅")
                return job["result"]
            if job["status"] == "error":
                st.error(f"{label} failed:\n\n```\n{job['error']}\n```")
                return None
            time.sleep(2)


# ---- Connectivity check --------------------------------------------------------------
try:
    health = api_get("/health")
except Exception:
    st.error(f"Cannot reach the API at {API_URL}. Start it with `uvicorn api.main:app --reload`.")
    st.stop()

# ---- Sidebar: configuration ----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    cfg = api_get("/config")
    domains = cfg.get("domains", ["nfcorpus", "flashcards", "medembed"])
    base_model = st.text_input("Base model", cfg["base_model"])
    domain = st.selectbox("Medical domain", domains, index=domains.index(cfg["domain"]))
    st.caption(f"Official MTEB task: **{cfg['mteb_task']}**")
    sample_size = st.number_input("Training pairs (0 = all)", 0, 500000, int(cfg["sample_size"] or 0))
    eval_queries = st.number_input("Eval queries", 10, 2000, int(cfg["eval_queries"]))
    num_negatives = st.number_input("Hard negatives / pair", 1, 10, int(cfg["num_negatives"]))
    epochs = st.number_input("Epochs", 1, 10, int(cfg["epochs"]))
    batch_size = st.number_input("Batch size", 4, 256, int(cfg["batch_size"]))
    run_mteb = st.checkbox("Run official MTEB task (slower)", value=bool(cfg["run_mteb"]))

    if st.button("Save config"):
        new_cfg = api_post("/config", {
            "base_model": base_model,
            "domain": domain,
            "sample_size": int(sample_size) or None,
            "eval_queries": int(eval_queries),
            "num_negatives": int(num_negatives),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "run_mteb": bool(run_mteb),
        })
        st.success("Config saved")
        st.json(new_cfg)

    st.info(f"Compute device: **{health['device']}**")

# ---- Pipeline steps ------------------------------------------------------------------
st.subheader("Pipeline")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**1. Data & triplets**")
    if st.button("Prepare medical data"):
        st.json(run_job("/prepare-data", "Prepare data"))
    if st.button("Collect triplets (mine / native)"):
        st.json(run_job("/mine-triplets", "Collect triplets"))
    if st.button("LLM clinical triplets (optional)"):
        result = run_job("/generate-triplets-llm", "LLM triplets")
        if result:
            st.json(result)
            usage = result.get("usage")
            if usage:
                m1, m2, m3 = st.columns(3)
                m1.metric("Input tokens", f"{usage['input_tokens']:,}")
                m2.metric("Output tokens", f"{usage['output_tokens']:,}")
                m3.metric("Est. cost (USD)", f"${usage['estimated_cost_usd']}")
                st.caption(f"Model: {usage['model']} · {usage['num_api_calls']} API calls")

with c2:
    st.markdown("**2. Shortlist, baseline & train**")
    if st.button("Benchmark candidate models"):
        result = run_job("/benchmark-models", "Model benchmark")
        if result:
            st.dataframe(pd.DataFrame(result["models"]), use_container_width=True)
    if st.button("Run MTEB/IR baseline"):
        st.json(run_job("/baseline", "Baseline"))
    if st.button("Fine-tune 🚀"):
        st.json(run_job("/train", "Fine-tune"))

with c3:
    st.markdown("**3. Evaluate**")
    if st.button("Evaluate fine-tuned"):
        st.json(run_job("/evaluate", "Evaluate"))

# ---- Results -------------------------------------------------------------------------
st.subheader("📊 Before vs After")
if st.button("Show comparison", type="primary"):
    resp = requests.get(f"{API_URL}/compare", timeout=30)
    if resp.status_code != 200:
        st.warning(f"Not ready yet: {resp.json().get('detail')}")
    else:
        result = resp.json()
        delta = result.get("headline_ir_ndcg@10_delta")
        st.metric("IR nDCG@10 improvement", delta,
                  delta="improved" if result.get("improved") else "no gain")

        df = pd.DataFrame(result["rows"]).set_index("metric")
        st.dataframe(df, use_container_width=True)

        chart_df = df[["baseline", "finetuned"]].dropna()
        if not chart_df.empty:
            st.bar_chart(chart_df)

        if result.get("triplet_accuracy_finetuned") is not None:
            st.caption(f"Held-out triplet accuracy (fine-tuned): "
                       f"{result['triplet_accuracy_finetuned']}")

st.subheader("💰 LLM token usage & cost")
if st.button("Show LLM usage"):
    resp = requests.get(f"{API_URL}/llm-usage", timeout=30)
    if resp.status_code != 200:
        st.info("No LLM triplet generation has run yet — token usage appears here after one.")
    else:
        usage = resp.json()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Input tokens", f"{usage['input_tokens']:,}")
        m2.metric("Output tokens", f"{usage['output_tokens']:,}")
        m3.metric("Total tokens", f"{usage['total_tokens']:,}")
        m4.metric("Est. cost (USD)", f"${usage['estimated_cost_usd']}")
        st.caption(
            f"Model: {usage['model']} · {usage['num_api_calls']} API calls · "
            f"prices: ${usage['input_price_per_1m_usd']}/1M in, "
            f"${usage['output_price_per_1m_usd']}/1M out"
        )

st.subheader("🏆 Run leaderboard")
if st.button("Show leaderboard"):
    rows = api_get("/leaderboard").get("rows", [])
    if not rows:
        st.info("Leaderboard is empty — finish a pipeline run first.")
    else:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
