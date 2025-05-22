#Install Required Libraries
#!pip install -q -U google-generativeai pandas==2.2.2 scikit-learn

#Import Libraries
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import re
import textwrap

#Gemini API Key
genai_api_key = input("Enter your Gemini API key: ")
genai.configure(api_key=genai_api_key)

#Load Dataset
file_path = "netflix-rotten-tomatoes-metacritic-imdb.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}. Please upload it using the Files tab in Colab.")

df = pd.read_csv(file_path)
df_cleaned = df.copy()

#Inspect Columns and Fix Naming
print("📄 Columns in dataset:", df_cleaned.columns.tolist())

# Auto-detect the summary column (case-insensitive search)
summary_col = None
for col in df_cleaned.columns:
    if "summary" in col.lower():
        summary_col = col
        break

if not summary_col:
    raise ValueError("Could not detect the 'Movie Summary' column. Please ensure your dataset contains it.")

# Normalize the summary column name
df_cleaned.rename(columns={summary_col: "Movie Summary"}, inplace=True)

#Convert Box Office column to numeric
def convert_box_office(value):
    if isinstance(value, str) and value.startswith("$"):
        return float(value.replace("$", "").replace(",", ""))
    return np.nan

df_cleaned["Boxoffice"] = df_cleaned["Boxoffice"].apply(convert_box_office)

#Progressive Missing Value Filling
def progressive_missing_value_handling(df, target_column):
    numeric_df = df.select_dtypes(include=[np.number])
    missing = numeric_df.isnull().sum()
    sorted_missing = missing[missing > 0].sort_values()

    if target_column not in sorted_missing.index:
        return df

    least_missing_attr = sorted_missing.index[0]
    df[least_missing_attr] = SimpleImputer(strategy="median").fit_transform(df[[least_missing_attr]])
    filled_columns = [least_missing_attr]

    for attr in sorted_missing.index[1:]:
        available = df.dropna(subset=filled_columns)
        if not available.empty:
            X = available[filled_columns]
            y = available[attr]
            if y.isnull().sum() > 0:
                y = SimpleImputer(strategy="median").fit_transform(y.values.reshape(-1, 1)).ravel()
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            mask = df[attr].isna() & df[filled_columns].notna().all(axis=1)
            if mask.sum() > 0:
                df.loc[mask, attr] = model.predict(df.loc[mask, filled_columns])
        filled_columns.append(attr)

    return df

#Clean Key Columns
for col in ["Rotten Tomatoes Score", "Metacritic Score", "Boxoffice", "IMDb Votes"]:
    df_cleaned = progressive_missing_value_handling(df_cleaned, col)

#Gemini Text Generation Wrapper
def generate_text(prompt, max_tokens=400):
    model = genai.GenerativeModel("models/gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

#Main Movie Generation Function
def generate_movie_story(user_prompt):
    extract_prompt = (
        f"From the user request below, extract the main movie genre and success criteria "
        f"(like IMDb, box office, awards). Format: Genre | Criteria\n\nRequest: {user_prompt}"
    )
    extracted_info = generate_text(extract_prompt, max_tokens=60)

    if "|" not in extracted_info:
        return "Could not extract genre and success factors from prompt. Try rephrasing it."

    genre, success_factors = map(str.strip, extracted_info.split("|"))
    if not genre:
        return "Genre not detected. Try using a more descriptive prompt."

    # Filter dataset
    genre_escaped = re.escape(genre)
    filtered = df_cleaned[df_cleaned["Genre"].str.contains(genre_escaped, case=False, na=False, regex=True)]

    if "IMDb" in success_factors:
        filtered = filtered[filtered["IMDb Score"] > df_cleaned["IMDb Score"].median()]
    if "Box" in success_factors:
        filtered = filtered[filtered["Boxoffice"] > df_cleaned["Boxoffice"].median()]
    if "Award" in success_factors:
        filtered = filtered[filtered["Awards Received"] > df_cleaned["Awards Received"].median()]

    if filtered.empty:
        return f"No successful movies found in genre '{genre}' with the given criteria."

    # Analyze successful summaries
    summaries = " ".join(filtered["Movie Summary"].dropna().tolist())
    theme_prompt = (
        f"Analyze the following successful {genre} movie summaries and extract the common themes and patterns:\n\n{summaries}"
    )
    themes = generate_text(theme_prompt, max_tokens=150)

    story_prompt = f"Using the common themes ({themes}), generate a brand new original {genre} movie plot that satisfies: {success_factors}."
    return generate_text(story_prompt, max_tokens=300)


# Helper function to force subheadings to new lines
def fix_subheadings(text):
    subheadings = [
        "Themes:", "Patterns:", "Logline:", "Synopsis:", "Themes & Patterns Hit:",
        "Box Office Appeal:", "Box Office Potential:", "Box Office Success:"
    ]
    for sh in subheadings:
        text = text.replace(sh, f"\n{sh}")  # Insert newline before subheading
    return text

# Run it
user_input = input("\n🎯 Enter your movie request (e.g., 'generate a thriller movie that wins awards'):\n")
full_output = generate_movie_story(user_input)

# Step 1: Fix subheadings
full_output_fixed = fix_subheadings(full_output)

# Step 2: Split into major sections
sections = re.split(r"(🧩.*?:|🎬.*?:)", full_output_fixed)

# Step 3: Print nicely
print("\n📜 Final Output:\n")
for part in sections:
    if part.strip() == "":
        continue
    if part.startswith("🧩") or part.startswith("🎬"):
        print(f"\n{part.strip()}\n")
    else:
        # wrap each paragraph separately
        paragraphs = part.strip().split("\n")
        for para in paragraphs:
            if para.strip() == "":
                continue
            if para.endswith(":"):  # subheading, don't wrap
                print(f"\n{para.strip()}")
            else:
                wrapped = "\n".join(textwrap.wrap(para.strip(), width=100))
                print(wrapped)

