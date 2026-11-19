# 🎯 Workforce Skills-Gap & Semantic Matcher

A production-grade, AI-powered talent intelligence dashboard designed for modern HR and technical recruitment. Built with **Python**, **Pandas**, **NumPy**, **ChromaDB**, **Sentence-Transformers**, and **Streamlit**.

---

## 🚀 Executive Summary
Traditional Applicant Tracking Systems (ATS) rely strictly on exact keyword matching, frequently rejecting qualified candidates whose resumes use different terminology (e.g., searching for `"Python"` but missing a candidate who wrote `"Pandas and FastAPI"`). 

This project solves that problem by implementing a **Hybrid Semantic & Keyword Matching Engine** that converts employee profiles and job descriptions into dense vector embeddings, allowing recruiters to discover relevant talent conceptually while identifying exact skill gaps for targeted upskilling.

---

## 🛠️ Tech Stack & Architecture
* **Data Processing:** Pandas (ETL, missing value imputation, text normalization) & NumPy (numerical vector operations).
* **Embedding Model:** HuggingFace `all-MiniLM-L6-v2` (Sentence-Transformers) generating 384-dimensional dense semantic vectors.
* **Vector Database:** ChromaDB utilizing HNSW indexing and Cosine Similarity (`hnsw:space: cosine`).
* **Web Interface:** Streamlit interactive recruiter dashboard featuring real-time hybrid scoring sliders and dynamic analytics charts.

```text
employees.csv ──▶ clean_data.py (Pandas ETL) ──▶ cleaned_employees.csv
                                                          │
role_requirements.json ──▶ ChromaDB + MiniLM Embeddings ◄┘
                                   │
                       app.py (Streamlit Dashboard)s