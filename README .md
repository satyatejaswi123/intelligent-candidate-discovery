# Intelligent Candidate Discovery & Ranking
### Redrob Data & AI Challenge — Senior AI Engineer JD

---

## Quick start

```bash
pip install scikit-learn numpy streamlit

# Generate submission CSV (≈90s on CPU, no GPU, no internet needed)
python rank_candidates.py \
  --jd job_description.txt \
  --candidates candidates.jsonl \
  --out submission.csv

# Streamlit demo
streamlit run app.py
```

---

## Files

| File | Purpose |
|------|---------|
| `rank_candidates.py` | Core ranking engine — run this |
| `job_description.txt` | JD as plain text (parsed dynamically) |
| `submission.csv` | Pre-generated top 100 output |
| `app.py` | Streamlit demo |
| `submission_metadata.yaml` | Fill in team info then submit |
| `requirements.txt` | Python dependencies |

---

## How it works

```
job_description.txt
        ↓  parse_jd()  — extract YOE range, locations, disqualifiers
        ↓  expand_text() — add synonyms (RAG↔retrieval, LLM↔genai, ...)
        ↓
candidates.jsonl (100K profiles)
        ↓  build_candidate_text() — concat title+skills+descriptions+certs
        ↓  expand_text() — same synonym enrichment

TF-IDF vectoriser  (15K features, sublinear-TF, min_df=3)
  fits on [JD_text] + [all 100K candidate texts]

cosine_similarity(JD_vector, candidate_matrix)
  → semantic_score[i]  for each candidate i

Final score = semantic(40%) + career_fit(30%) + tech_depth(15%) + behavioural(15%)
                   ↓
         Sort DESC, tie-break by candidate_id ASC
                   ↓
         Top 100 → submission.csv
```

### Semantic scoring (40%) — beyond keyword matching

TF-IDF + cosine similarity handles vocabulary mismatch automatically:
- A candidate who wrote *"built RAG chatbot with LangChain"* scores high for *"LLM + embeddings + retrieval"* because the synonym expansion map adds canonical terms before vectorisation.
- Terms like `FAISS`, `NDCG`, `bi-encoder` are rare in the corpus → high IDF → high weight → small overlap is very significant.
- Generic words like `experience`, `team` have near-zero IDF → don't pollute the score.

### Career fit (30%)

- Experience in JD's stated range (5–9yr) → full score; adjacent bands → partial
- Title match: "Senior ML Engineer / AI Engineer / NLP Engineer" → 1.0; "Marketing Manager" → 0.05
- Consulting-only career (TCS/Infosys/Wipro/etc.) → 0.3× multiplier (explicit JD disqualifier)
- JD keyword overlap in career descriptions, AI/ML role fraction, location, notice period

### Technical depth (15%)

Education institution tier, AI/ML certifications (AWS ML, GCP, TF, PyTorch, etc.), GitHub activity, Redrob platform assessment scores

### Behavioural signals (15%)

Last-active recency, open-to-work flag, recruiter response rate, response time, interview completion rate, saved-by-recruiters, offer acceptance rate, verification status. A perfect-on-paper candidate who is inactive/unresponsive is ranked lower than a slightly weaker but *reachable* one.

### JD-agnostic by design

`parse_jd()` reads experience range, location preferences, and disqualifier keywords from any plain-text JD. Swap `job_description.txt` for a different role and rerun — zero code changes.

---

## Compute constraints compliance

| Constraint | Limit | This system |
|------------|-------|-------------|
| Runtime | ≤5 min | ~90s on 4-core CPU |
| RAM | ≤16 GB | ~600MB |
| GPU | Not allowed | CPU only ✅ |
| Network | Not allowed | Zero external calls ✅ |
