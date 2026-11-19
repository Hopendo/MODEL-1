import pandas as pd
import numpy as np

def clean_workforce_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # 1. Handle missing values
    df["full_name"] = df["full_name"].fillna("Unknown Employee")

    # 2. Clean text columns: strip whitespace and standardize title casing
    df["role_title"] = df["role_title"].str.strip().str.title()
    df["department"] = df["department"].str.strip()
    df["bio_and_skills"] = df["bio_and_skills"].str.strip()

    # 3. Ensure numeric types for experience
    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce").fillna(0).astype(int)

    # 4. Create a unified rich text field for vector embedding
    df["searchable_profile"] = (
        "Role: " + df["role_title"] + " | "
        "Department: " + df["department"] + " | "
        "Experience: " + df["experience_years"].astype(str) + " years | "
        "Skills: " + df["bio_and_skills"]
    )

    df.to_csv(output_path, index=False)
    print(f"Cleaned {len(df)} records and saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_workforce_data("data/employees.csv", "data/cleaned_employees.csv")