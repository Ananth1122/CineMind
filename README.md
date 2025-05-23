# CineMind
AI-Powered Movie Plot Generator using Gemini & ML
This project is an AI-driven movie story generator that combines Google Gemini's generative text capabilities with machine learning-based data cleaning and analysis to create original movie plotlines based on successful trends from real-world datasets.

#Overview
Using a dataset of movies (including IMDb scores, Rotten Tomatoes ratings, Box Office earnings, and summaries), this tool:

Cleans and imputes missing values using advanced techniques like Random Forest-based prediction.

-Uses Google Gemini (Gemini 1.5 Pro) to:

-Extract genre and success criteria from user input

-Analyze real movie summaries for recurring themes and patterns

-Generate an original movie plot that mimics real successful examples

-Supports flexible natural language input like:

-“Generate a romantic drama that performs well at the box office and IMDb.”

#Dependencies
Install required libraries before running the script:

bash
Copy
Edit
pip install -q -U google-generativeai pandas==2.2.2 scikit-learn
Python Libraries Used:
pandas: Data manipulation

numpy: Numerical operations

scikit-learn: ML models and data imputation

google-generativeai: Access to Gemini text generation

re, textwrap, os: Utilities

#Dataset
File: netflix-rotten-tomatoes-metacritic-imdb.csv

#Expected Columns: Should include Genre, Boxoffice, IMDb Score, Rotten Tomatoes Score, Metacritic Score, Awards Received, and a Movie Summary column (case-insensitive match).

If the file is missing, the script will raise an error. Upload it via Colab or ensure it exists locally.

#How It Works
#Data Cleaning

Converts dollar values to numeric

Fills missing values using median imputation and RandomForest regressors

#Prompt Handling

Extracts the target genre and success metrics (IMDb score, Box Office, Awards) using Gemini

#Data Filtering

Selects only the movies that match genre and success criteria

#Theme Analysis

Summarizes patterns from successful movie plots

#Story Generation

Constructs a brand-new plot based on the learned themes

#How to Use
Enter your Gemini API Key when prompted.

Provide a movie prompt, e.g.:

csharp
Copy
Edit
generate a sci-fi movie that is critically acclaimed and does well on IMDb
Read the generated plot, which includes themes and a story optimized to match real-world success indicators.

Example Input
css
Copy
Edit
generate a mystery thriller movie that wins awards
Example Output
text
Copy
Edit
Genre: Thriller
Criteria: IMDb, Awards

#Themes:
- Unreliable narrators
- Psychological tension
- Twist endings

#Logline:
A disoriented detective discovers his past may hold the key to a string of chilling murders.

#Synopsis:
Notes
Works best with complete and clean datasets.

Make sure the column "Movie Summary" exists (case-insensitive).

The script will fail gracefully if genre/criteria can't be extracted or no matches are found.

Powered By
Google Gemini 1.5 Pro

Scikit-Learn

Pandas

