"""
Streamlit Demo — Intelligent Candidate Discovery & Ranking
Redrob Data & AI Challenge
"""
import streamlit as st, json, os, csv
from rank_candidates import run_ranking, parse_jd, build_candidate_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, date

st.set_page_config(page_title="Intelligent Candidate Discovery",
                   page_icon="🎯", layout="wide")

st.title("🎯 Intelligent Candidate Discovery & Ranking")
st.caption("Redrob Data & AI Challenge · TF-IDF Semantic Matching + Career & Behavioural Signals")

# ── Sidebar controls ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_n       = st.slider("Show Top N", 10, 100, 20)
    filter_open = st.checkbox("Only Open-to-Work", False)
    min_yoe     = st.slider("Min YoE", 0, 15, 3)
    max_yoe     = st.slider("Max YoE", 1, 25, 15)
    st.divider()
    st.subheader("📊 Score Weights")
    w_sem = st.slider("Semantic Match",   0.0, 1.0, 0.40, 0.05)
    w_car = st.slider("Career Fit",       0.0, 1.0, 0.30, 0.05)
    w_tec = st.slider("Technical Depth",  0.0, 1.0, 0.15, 0.05)
    w_beh = st.slider("Behavioural",      0.0, 1.0, 0.15, 0.05)
    total = w_sem+w_car+w_tec+w_beh
    if abs(total-1.0) > 0.06:
        st.warning(f"Weights sum = {total:.2f} (aim for 1.0)")

# ── Check files ─────────────────────────────────────────────────
CAND_PATH = "candidates.jsonl"
JD_PATH   = "job_description.txt"
missing = [p for p in [CAND_PATH, JD_PATH] if not os.path.exists(p)]
if missing:
    st.error(f"Missing files: {missing}\nPlace them in the same folder as app.py")
    st.stop()

# ── Load & score ────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading candidates & computing TF-IDF…")
def load_and_score(cand_path, jd_path):
    from rank_candidates import (parse_jd, build_candidate_text,
                                  score_career, score_technical, score_behavioral,
                                  expand_text)
    with open(jd_path,"r",encoding="utf-8") as f: jd_text=f.read()
    jd = parse_jd(jd_text)
    candidates, texts = [], []
    with open(cand_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                c=json.loads(line)
                candidates.append(c)
                texts.append(build_candidate_text(c))
            except: pass
    vectorizer = TfidfVectorizer(max_features=15000,sublinear_tf=True,min_df=3)
    all_texts  = [jd["expanded_text"]] + texts
    tfidf      = vectorizer.fit_transform(all_texts)
    sims = cosine_similarity(tfidf[0], tfidf[1:]).flatten()
    sim_max = sims.max() or 1.0
    sims_n  = sims / sim_max
    rows = []
    for i,c in enumerate(candidates):
        sem  = float(sims_n[i])
        car  = score_career(c, jd)
        tec  = score_technical(c)
        beh  = score_behavioral(c)
        rows.append(dict(
            candidate_id=c["candidate_id"],
            semantic=round(sem*100,2),
            career=round(car*100,2),
            technical=round(tec*100,2),
            behavioral=round(beh*100,2),
            title=c["profile"].get("current_title",""),
            yoe=c["profile"].get("years_of_experience",0),
            rr=c["redrob_signals"].get("recruiter_response_rate",0),
            otw=c["redrob_signals"].get("open_to_work_flag",False),
            last_active=c["redrob_signals"].get("last_active_date",""),
        ))
    return rows

all_rows = load_and_score(CAND_PATH, JD_PATH)

# ── Apply weights + filters ──────────────────────────────────────
def composite(r):
    return (r["semantic"]*w_sem + r["career"]*w_car +
            r["technical"]*w_tec + r["behavioral"]*w_beh)

filtered = [r for r in all_rows
            if min_yoe <= r["yoe"] <= max_yoe
            and (not filter_open or r["otw"])]
filtered.sort(key=lambda r: (-composite(r), r["candidate_id"]))
display  = filtered[:top_n]

# ── Metrics ──────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total candidates", f"{len(all_rows):,}")
c2.metric("After filters",    f"{len(filtered):,}")
c3.metric("Top score",        f"{composite(filtered[0]):.1f}" if filtered else "—")
c4.metric("Avg top-10",
          f"{sum(composite(r) for r in filtered[:10])/10:.1f}" if len(filtered)>=10 else "—")

st.divider()

# ── JD summary ───────────────────────────────────────────────────
with st.expander("📄 Job Description", expanded=False):
    with open(JD_PATH) as f: st.text(f.read()[:3000])

# ── Table ─────────────────────────────────────────────────────────
def sc(v): return "🟢" if v>=70 else ("🟡" if v>=50 else "🔴")
def fmt_date(d):
    if not d: return "—"
    try:
        days=(date.today()-datetime.strptime(d,"%Y-%m-%d").date()).days
        return f"{'✅' if days<=7 else '🟡' if days<=30 else '🟠' if days<=90 else '🔴'} {days}d ago"
    except: return d

st.subheader(f"🏆 Top {len(display)} Candidates")
hcols = st.columns([.4,1.8,2.5,.8,1,1,1,1,1.2])
for col,h in zip(hcols,["#","ID","Title","YoE","Semantic","Career","Tech","Signals","Last Active"]):
    col.markdown(f"**{h}**")
st.markdown("---")
for i,r in enumerate(display):
    cols = st.columns([.4,1.8,2.5,.8,1,1,1,1,1.2])
    cols[0].write(i+1)
    cols[1].code(r["candidate_id"])
    cols[2].write(r["title"])
    cols[3].write(f"{r['yoe']:.1f}yr")
    cols[4].write(f"{sc(r['semantic'])} {r['semantic']:.0f}")
    cols[5].write(f"{sc(r['career'])} {r['career']:.0f}")
    cols[6].write(f"{sc(r['technical'])} {r['technical']:.0f}")
    cols[7].write(f"{sc(r['behavioral'])} {r['behavioral']:.0f}")
    cols[8].write(fmt_date(r["last_active"]))

# ── Download ─────────────────────────────────────────────────────
st.divider()
lines = ["candidate_id,rank,score,reasoning"]
for rank,r in enumerate(filtered[:100],1):
    score = composite(r)
    lines.append(
        f"{r['candidate_id']},{rank},{score/100:.4f},"
        f"{r['title']} | {r['yoe']:.1f}yr | "
        f"Semantic:{r['semantic']:.0f} Career:{r['career']:.0f} "
        f"Tech:{r['technical']:.0f} Signals:{r['behavioral']:.0f} | "
        f"resp:{r['rr']:.2f} otw:{r['otw']}"
    )
st.download_button("⬇️ Download Top 100 CSV", "\n".join(lines),
                   "submission.csv", "text/csv")

# ── How it works ──────────────────────────────────────────────────
with st.expander("🧠 How the Ranking Works"):
    st.markdown("""
## Architecture

| Dimension | Default Weight | What it captures |
|-----------|---------------|-----------------|
| **Semantic Match** (TF-IDF) | 40% | Cosine similarity between JD text and candidate corpus. A candidate who built a *"RAG pipeline with LangChain"* scores high for *"LLM + embeddings + retrieval"* thanks to synonym expansion before vectorisation. |
| **Career Fit** | 30% | Experience years (5–9yr sweet spot), title alignment, AI/ML role history, career description JD-keyword overlap, location, notice period. Consulting-only backgrounds receive a 0.3× penalty. |
| **Technical Depth** | 15% | Education institution tier, AI/ML certifications, GitHub activity score, platform assessment scores. |
| **Behavioural Signals** | 15% | Recency of last login, open-to-work flag, recruiter response rate, interview completion rate, saved-by-recruiters — treats availability as a multiplier, not an afterthought. |

### Why TF-IDF over keyword matching?

TF-IDF gives each term a weight based on how **rare** it is across all 100K candidates. Terms like *"FAISS"* or *"NDCG"* that appear in only a few profiles get high weight, while generic words like *"experience"* get down-weighted automatically. Cosine similarity then measures direction in this high-dimensional space — so two documents that *talk about the same concepts* score high even if they use different words.

### JD-agnostic design

The system reads the JD from a file and parses it dynamically (YOE range, location preferences, disqualifier keywords). Swap `job_description.txt` for any other role and rerun — no code changes needed.
""")
