import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

def run_semantic_matcher(cleaned_csv_path: str, roles_json_path: str):
    # 1. Load cleaned workforce data and target role requirements
    df = pd.read_csv(cleaned_csv_path)
    with open(roles_json_path, "r", encoding="utf-8") as f:
        roles = json.load(f)

    # 2. Initialize local Sentence-Transformer embedding model (all-MiniLM-L6-v2)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # 3. Initialize in-memory ChromaDB client with Cosine Similarity
    client = chromadb.Client()
    collection = client.create_collection(
        name="workforce_profiles",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Ingest employee searchable profiles and metadata into ChromaDB
    collection.add(
        ids=df["emp_id"].tolist(),
        documents=df["searchable_profile"].tolist(),
        metadatas=df[["full_name", "department", "role_title", "experience_years"]].to_dict(orient="records")
    )

    # 5. Query ChromaDB for each role and analyze skill gaps
    for role in roles:
        print("=" * 65)
        print(f"TARGET ROLE: {role['title']} ({role['role_id']}) - {role['department']}")
        print(f"Description: {role['description']}")
        print("=" * 65)

        results = collection.query(
            query_texts=[role["description"]],
            n_results=len(df)
        )

        required_skills = set(k.lower() for k in role["required_keywords"])

        for rank, (emp_id, doc, meta, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ),
            start=1
        ):
            # Convert cosine distance (0 = identical, 2 = opposite) into a 0-100% similarity score
            similarity_pct = round((1 - distance) * 100, 1)

            # Identify matched vs. missing required keywords
            doc_lower = doc.lower()
            matched_skills = sorted([skill for skill in required_skills if skill in doc_lower])
            missing_skills = sorted([skill for skill in required_skills if skill not in doc_lower])

            print(f"\n#{rank} | {meta['full_name']} ({emp_id}) - {meta['role_title']} ({meta['experience_years']} yrs)")
            print(f"    Semantic Fit Score : {similarity_pct}%")
            print(f"    Matched Skills     : {', '.join(matched_skills) if matched_skills else 'None'}")
            print(f"    Skill Gaps         : {', '.join(missing_skills) if missing_skills else 'None (Full Match)'}")

if __name__ == "__main__":
    run_semantic_matcher("data/cleaned_employees.csv", "data/role_requirements.json")