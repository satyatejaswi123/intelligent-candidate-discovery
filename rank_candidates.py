"""
Intelligent Candidate Discovery & Ranking System
Redrob Data & AI Challenge

Architecture:
  1. Semantic Skill Match  (40%) — TF-IDF cosine similarity (memory-efficient)
  2. Career Fit            (30%) — experience, title, company history
  3. Technical Depth       (15%) — education, certifications, GitHub, assessments
  4. Behavioral Signals    (15%) — activity, responsiveness, availability

Usage:
  python rank_candidates.py --jd job_description.txt --candidates candidates.jsonl --out submission.csv
"""

import json, csv, re, argparse
from datetime import datetime, date
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────
# SYNONYM MAP — semantic enrichment layer
# ──────────────────────────────────────────
SYNONYM_MAP = {
    "rag": ["retrieval augmented generation", "rag pipeline", "rag chatbot"],
    "llm": ["large language model", "gpt", "llama", "mistral", "claude", "genai"],
    "embeddings": ["embedding", "vector representation", "dense retrieval",
                   "sentence transformer", "bi encoder", "text embedding"],
    "ranking": ["ranker", "ranking system", "reranker", "learning to rank", "ltr"],
    "nlp": ["natural language processing", "text classification", "named entity",
            "tokenization", "spacy", "nltk"],
    "vector database": ["pinecone", "weaviate", "qdrant", "milvus", "faiss",
                        "chroma", "pgvector", "annoy", "hnsw", "vector store"],
    "production ml": ["mlops", "model deployment", "model serving", "kubeflow",
                      "mlflow", "inference pipeline"],
    "recommendation": ["recommender", "collaborative filtering", "item embedding"],
    "hybrid search": ["bm25", "sparse dense", "hybrid retrieval"],
    "fine tuning": ["lora", "qlora", "peft", "instruction tuning", "sft"],
    "information retrieval": ["search engine", "inverted index", "passage retrieval"],
}

def expand_text(text: str) -> str:
    t = text.lower()
    extras = []
    for canonical, synonyms in SYNONYM_MAP.items():
        if any(h in t for h in [canonical] + synonyms):
            extras.extend([canonical] + synonyms[:2])
    return text + " " + " ".join(extras)

# ──────────────────────────────────────────
# JD PARSER
# ──────────────────────────────────────────
def parse_jd(jd_text: str) -> dict:
    t = jd_text.lower()
    yoe_min, yoe_max = 0, 20
    for pat in [r'(\d+)\s*[–\-–]\s*(\d+)\s*years?', r'(\d+)\+\s*years?',
                r'minimum\s+(\d+)\s*years?', r'at least\s+(\d+)\s*years?']:
        m = re.search(pat, t)
        if m:
            nums = [int(x) for x in m.groups() if x]
            yoe_min, yoe_max = min(nums), max(nums) if len(nums) > 1 else min(nums)+5
            break
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant",
                        "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware"]
    indian_cities = ["pune","noida","hyderabad","mumbai","delhi",
                     "bangalore","bengaluru","gurgaon","chennai","kolkata"]
    pref_locs = [c for c in indian_cities if c in t] or (["india"] if "india" in t else [])
    notice_pref = 30
    m = re.search(r'(\d+)[- ]day\s*notice', t)
    if m: notice_pref = int(m.group(1))
    return {
        "raw_text": jd_text,
        "expanded_text": expand_text(jd_text),
        "yoe_min": yoe_min, "yoe_max": yoe_max,
        "consulting_firms": consulting_firms,
        "disqualify_consulting": any(f in t for f in consulting_firms),
        "preferred_locations": pref_locs,
        "notice_pref_days": notice_pref,
    }

# ──────────────────────────────────────────
# CANDIDATE TEXT BUILDER
# ──────────────────────────────────────────
PROF_WEIGHT = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}

def build_candidate_text(c: dict) -> str:
    parts = []
    p = c.get("profile", {})
    parts.append(p.get("current_title", "") + " " + p.get("current_title", ""))  # repeat title
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    for s in c.get("skills", []):
        w = PROF_WEIGHT.get(s.get("proficiency", "beginner"), 1)
        parts.append((s["name"] + " ") * w)
    for role in c.get("career_history", []):
        parts.append(role.get("title", "") + " " + role.get("description", ""))
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", "") + " " + cert.get("issuer", ""))
    for edu in c.get("education", []):
        parts.append(edu.get("field_of_study", "") + " " + edu.get("degree", ""))
    return expand_text(" ".join(p for p in parts if p.strip()))

# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────
def days_since(ds):
    if not ds: return 9999
    try: return (date.today() - datetime.strptime(ds, "%Y-%m-%d").date()).days
    except: return 9999

def clamp(v, lo=0.0, hi=1.0): return max(lo, min(hi, v))
def norm(v, lo, hi): return clamp((v - lo) / (hi - lo) if hi != lo else 0.5)

# ──────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────
POS_TITLES = ["machine learning","ml engineer","ai engineer","nlp engineer",
              "data scientist","applied scientist","search engineer",
              "deep learning","recommendation","ranking","staff engineer",
              "principal engineer","founding engineer","research engineer"]
NEG_TITLES = ["marketing","hr manager","content writer","graphic designer",
              "project manager","sales","accountant","ui designer",
              "customer success","finance manager"]

def score_career(c: dict, jd: dict) -> float:
    profile = c["profile"]
    career = c.get("career_history", [])
    yoe = profile.get("years_of_experience", 0)
    lo, hi = jd["yoe_min"], jd["yoe_max"]
    if lo <= yoe <= hi:           yoe_s = 1.0
    elif lo-1 <= yoe < lo:        yoe_s = 0.80
    elif hi < yoe <= hi+3:        yoe_s = 0.75
    elif yoe < lo-1:              yoe_s = max(0.2, yoe/lo*0.7) if lo>0 else 0.4
    else:                         yoe_s = max(0.4, 1-(yoe-hi)/20)

    title = profile.get("current_title","").lower()
    if any(p in title for p in POS_TITLES): title_s = 1.0
    elif any(p in title for p in ["engineer","developer","architect","scientist",
                                   "analyst","researcher"]): title_s = 0.55
    elif any(p in title for p in NEG_TITLES): title_s = 0.05
    else: title_s = 0.25

    consulting_count = sum(1 for r in career
        if any(f in r.get("company","").lower() for f in jd["consulting_firms"]))
    all_consulting = jd["disqualify_consulting"] and career and consulting_count==len(career)
    cf = 0.3 if all_consulting else 1.0

    ai_count = sum(1 for r in career
        if any(t in r.get("title","").lower() for t in
               ["ai","ml","machine learning","nlp","search","ranking",
                "recommendation","data scientist"]))
    ai_frac = ai_count / max(1, len(career))

    jd_kw = set(w for w in re.findall(r'\b\w{5,}\b', jd["raw_text"].lower())
                if len(w) >= 5)
    prod_s = 0.0
    for role in career:
        desc = (role.get("description","")+" "+role.get("title","")).lower()
        desc_kw = set(re.findall(r'\b\w{5,}\b', desc))
        prod_s += min(0.3, len(jd_kw & desc_kw) / max(1,len(jd_kw)) * 6)
    prod_s = min(1.0, prod_s)

    loc = (profile.get("location","")+" "+profile.get("country","")).lower()
    loc_s = 1.0 if (not jd["preferred_locations"] or
                    any(l in loc for l in jd["preferred_locations"]) or
                    "india" in loc) else 0.45

    notice = c["redrob_signals"].get("notice_period_days", 90)
    pn = jd["notice_pref_days"]
    if notice<=pn: n_s=1.0
    elif notice<=pn*2: n_s=0.7
    elif notice<=90: n_s=0.5
    else: n_s=0.3

    return (yoe_s*0.30 + title_s*0.30 + prod_s*0.15 +
            ai_frac*0.10 + loc_s*0.08 + n_s*0.07) * cf

def score_technical(c: dict) -> float:
    edu_s = 0.4
    for e in c.get("education",[]):
        tv = {"tier_1":1.0,"tier_2":0.8,"tier_3":0.6,"tier_4":0.45}.get(e.get("tier",""),0.4)
        edu_s = max(edu_s, tv)
        if any(f in e.get("field_of_study","").lower()
               for f in ["computer","ai","machine learning","data science","statistics","math"]):
            edu_s = min(1.0, edu_s+0.08)

    ai_cert_kw = ["machine learning","deep learning","artificial intelligence","nlp",
                  "aws","gcp","azure","tensorflow","pytorch","data science","mlops","databricks"]
    ai_certs = sum(1 for cert in c.get("certifications",[])
                   if any(kw in (cert.get("name","")+cert.get("issuer","")).lower()
                          for kw in ai_cert_kw))
    cert_s = min(1.0, ai_certs*0.3)

    gh = c["redrob_signals"].get("github_activity_score",-1)
    gh_s = norm(gh,0,100) if gh>=0 else 0.3

    asm = c["redrob_signals"].get("skill_assessment_scores",{})
    ai_asm = [v for k,v in asm.items()
              if any(kw in k.lower() for kw in
                     ["python","ml","nlp","ai","data","deep","machine","llm","sql"])]
    asm_s = (sum(ai_asm)/len(ai_asm)/100) if ai_asm else 0.45

    comp_s = norm(c["redrob_signals"].get("profile_completeness_score",50),0,100)
    return edu_s*0.35 + cert_s*0.25 + gh_s*0.22 + asm_s*0.10 + comp_s*0.08

def score_behavioral(c: dict) -> float:
    s = c["redrob_signals"]
    di = days_since(s.get("last_active_date",""))
    if di<=7: act=1.0
    elif di<=30: act=0.85
    elif di<=60: act=0.65
    elif di<=90: act=0.45
    elif di<=180: act=0.25
    else: act=0.08

    otw = 1.0 if s.get("open_to_work_flag",False) else 0.40
    rr  = s.get("recruiter_response_rate",0.5)
    rth = s.get("avg_response_time_hours",48)
    rt  = 1.0 if rth<=4 else (0.8 if rth<=24 else (0.6 if rth<=48 else max(0.15,1-rth/200)))
    icr = s.get("interview_completion_rate",0.5)
    saved = norm(s.get("saved_by_recruiters_30d",0),0,10)
    oar = s.get("offer_acceptance_rate",-1); oar = oar if oar>=0 else 0.5
    verif = ((1.0 if s.get("verified_email",False) else 0)+
             (0.5 if s.get("verified_phone",False) else 0)+
             (0.5 if s.get("linkedin_connected",False) else 0))/2
    return act*0.25+otw*0.15+rr*0.20+rt*0.10+icr*0.10+saved*0.07+oar*0.07+verif*0.06

# ──────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────
def run_ranking(jd_path, candidates_path, output_csv, top_n=100):
    # 1. Parse JD
    print(f"[1/4] Loading JD: {jd_path}")
    with open(jd_path,"r",encoding="utf-8") as f: jd_text=f.read()
    jd = parse_jd(jd_text)
    print(f"      YOE: {jd['yoe_min']}–{jd['yoe_max']}yr | "
          f"Locations: {jd['preferred_locations']} | "
          f"Consulting flag: {jd['disqualify_consulting']}")

    # 2. Stream candidates
    print(f"[2/4] Streaming: {candidates_path}")
    candidates, texts = [], []
    with open(candidates_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                c=json.loads(line)
                candidates.append(c)
                texts.append(build_candidate_text(c))
            except: pass
            if len(candidates)%20000==0:
                print(f"      {len(candidates):,} loaded...")
    print(f"      {len(candidates):,} total candidates")

    # 3. TF-IDF semantic similarity (memory-optimised)
    print("[3/4] TF-IDF cosine similarity (semantic matching)...")
    # Unigrams only, 15K features — keeps sparse matrix ~400MB
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,1),
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )
    all_texts = [jd["expanded_text"]] + texts
    tfidf = vectorizer.fit_transform(all_texts)
    jd_vec   = tfidf[0]
    cand_mat = tfidf[1:]
    # cosine_similarity returns (1, N); flatten to (N,)
    sim_scores = cosine_similarity(jd_vec, cand_mat).flatten()
    # Normalise to 0-1 (max observed sim ~ 0.25-0.40 for text)
    sim_max = sim_scores.max() if sim_scores.max()>0 else 1.0
    sim_norm = sim_scores / sim_max
    del tfidf, cand_mat  # free memory

    # 4. Full scoring
    print("[4/4] Scoring all candidates...")
    results = []
    for i, c in enumerate(candidates):
        sem  = float(sim_norm[i])
        car  = score_career(c, jd)
        tech = score_technical(c)
        beh  = score_behavioral(c)
        comp = sem*0.40 + car*0.30 + tech*0.15 + beh*0.15
        results.append({
            "candidate_id": c["candidate_id"],
            "score": round(comp*100, 4),
            "semantic": round(sem*100, 2),
            "career":   round(car*100, 2),
            "tech":     round(tech*100, 2),
            "behavioral": round(beh*100, 2),
            "title": c["profile"].get("current_title",""),
            "yoe":   c["profile"].get("years_of_experience",0),
            "rr":    c["redrob_signals"].get("recruiter_response_rate",0),
            "otw":   c["redrob_signals"].get("open_to_work_flag",False),
        })

    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top = results[:top_n]

    with open(output_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["candidate_id","rank","score","reasoning"])
        for rank, r in enumerate(top,1):
            w.writerow([
                r["candidate_id"], rank,
                round(r["score"]/100,4),
                f"{r['title']} | {r['yoe']:.1f}yr | "
                f"Semantic:{r['semantic']:.0f} Career:{r['career']:.0f} "
                f"Tech:{r['tech']:.0f} Signals:{r['behavioral']:.0f} | "
                f"resp_rate:{r['rr']:.2f} open_to_work:{r['otw']}"
            ])

    print(f"\nDone. Top {top_n} → {output_csv}")
    print("\nTop 10:")
    for r in top[:10]:
        print(f"  #{top.index(r)+1:2d} {r['candidate_id']} "
              f"Score:{r['score']:.1f} | {r['title']} | {r['yoe']:.1f}yr | "
              f"Sem:{r['semantic']:.0f} Resp:{r['rr']:.0%}")
    return top

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd",         default="job_description.txt")
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out",        default="submission.csv")
    parser.add_argument("--top",        type=int, default=100)
    args = parser.parse_args()
    run_ranking(args.jd, args.candidates, args.out, args.top)
