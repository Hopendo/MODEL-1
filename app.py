import json
import pandas as pd
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# Page configuration
st.set_page_config(
    page_title="Workforce Skills-Gap & Semantic Matcher",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def load_vector_collection(csv_path: str):
    """Loads cleaned employee data and caches the ChromaDB collection in memory."""
    df = pd.read_csv(csv_path)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="workforce_dashboard",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        collection.add(
            ids=df["emp_id"].tolist(),
            documents=df["searchable_profile"].tolist(),
            metadatas=df[["full_name", "department", "role_title", "experience_years"]].to_dict(orient="records")
        )
    return df, collection

def load_default_role(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        roles = json.load(f)
    return roles[0]

# --- HEADER ---
st.title("🎯 Workforce Skills-Gap & Semantic Matcher")
st.markdown(
    "Semantic talent discovery powered by **Sentence-Transformers (`all-MiniLM-L6-v2`)**, "
    "**ChromaDB**, and **Pandas**."
)

# Load data & cached vector index
df, collection = load_vector_collection("data/cleaned_employees.csv")
default_role = load_default_role("data/role_requirements.json")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Matching Controls")

semantic_weight = st.sidebar.slider(
    "Semantic AI Weight (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=5,
    help="Balance between deep contextual meaning (ChromaDB) and exact keyword overlap."
)
keyword_weight = 100 - semantic_weight
st.sidebar.caption(f"**Scoring Formula:** {semantic_weight}% Semantic + {keyword_weight}% Keyword")

min_experience = st.sidebar.slider(
    "Minimum Experience (Years)",
    min_value=0,
    max_value=int(df["experience_years"].max()),
    value=0
)

departments = ["All"] + sorted(df["department"].unique().tolist())
selected_dept = st.sidebar.selectbox("Filter by Current Department", departments)

# --- MAIN INPUT AREA ---
col1, col2 = st.columns([2, 1])

with col1:
    role_title = st.text_input("Target Role Title", value=default_role["title"])
    job_description = st.text_area(
        "Job Description (Edit or paste any custom role description below)",
        value=default_role["description"],
        height=120
    )

with col2:
    keywords_raw = st.text_input(
        "Required Skills / Keywords (comma-separated)",
        value=", ".join(default_role["required_keywords"])
    )
    st.info(
        "💡 **Tip:** Try editing the description to *'Cloud infrastructure, Linux troubleshooting, and Cisco networking'* to watch rankings shift dynamically!"
    )

required_skills = sorted({k.strip().lower() for k in keywords_raw.split(",") if k.strip()})

# --- QUERY & RANKING ---
if job_description.strip():
    results = collection.query(
        query_texts=[job_description],
        n_results=len(df)
    )

    ranked_candidates = []
    for emp_id, doc, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if meta["experience_years"] < min_experience:
            continue
        if selected_dept != "All" and meta["department"] != selected_dept:
            continue

        semantic_score = max(0.0, round((1 - distance) * 100, 1))

        doc_lower = doc.lower()
        matched = [s for s in required_skills if s in doc_lower]
        missing = [s for s in required_skills if s not in doc_lower]
        keyword_score = round((len(matched) / len(required_skills)) * 100, 1) if required_skills else 100.0

        hybrid_score = round(
            (semantic_score * (semantic_weight / 100)) + (keyword_score * (keyword_weight / 100)),
            1
        )

        ranked_candidates.append({
            "emp_id": emp_id,
            "full_name": meta["full_name"],
            "role_title": meta["role_title"],
            "department": meta["department"],
            "experience_years": meta["experience_years"],
            "hybrid_score": hybrid_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "matched": matched,
            "missing": missing,
            "profile": doc
        })

    ranked_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

    st.divider()
    st.subheader(f"📊 Ranked Candidates for: {role_title} ({len(ranked_candidates)} matches)")

    if not ranked_candidates:
        st.warning("No candidates match the current sidebar filters. Try lowering the minimum experience.")
    else:
        chart_df = pd.DataFrame([
            {
                "Candidate": f"{c['full_name']} ({c['emp_id']})",
                "Hybrid Readiness (%)": c["hybrid_score"],
                "Semantic Fit (%)": c["semantic_score"],
                "Keyword Match (%)": c["keyword_score"]
            }
            for c in ranked_candidates
        ]).set_index("Candidate")

        st.bar_chart(chart_df[["Hybrid Readiness (%)", "Semantic Fit (%)"]])

        for idx, cand in enumerate(ranked_candidates, start=1):
            with st.container(border=True):
                top_left, top_right = st.columns([3, 1])
                with top_left:
                    st.markdown(
                        f"### #{idx} {cand['full_name']} (`{cand['emp_id']}`) — "
                        f"*{cand['role_title']}* ({cand['department']}, {cand['experience_years']} yrs)"
                    )
                with top_right:
                    st.metric(
                        label="Overall Readiness",
                        value=f"{cand['hybrid_score']}%",
                        delta=f"Semantic: {cand['semantic_score']}%"
                    )

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.success(f"**✅ Matched Skills:** {', '.join(cand['matched']) if cand['matched'] else 'None'}")
                with m_col2:
                    if cand["missing"]:
                        st.error(f"**⚠️ Skill Gaps (Upskilling Needed):** {', '.join(cand['missing'])}")
                    else:
                        st.info("**🎉 Full Skill Match — Zero Gaps!**")

                with st.expander("View Indexed Searchable Profile"):
                    st.write(cand["profile"])