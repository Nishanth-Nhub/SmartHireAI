import pandas as pd
import numpy as np
import os

np.random.seed(42)
num_samples = 1000

years_experience = np.random.exponential(scale=3, size=num_samples).clip(0, 15).astype(int)
data = {
    "keyword_count": np.random.poisson(lam=2, size=num_samples).clip(0, 5),
    "years_experience": years_experience,
    "word_count": np.random.normal(loc=150 + years_experience * 10, scale=30, size=num_samples).clip(50, 300).astype(int),
    "role_count": np.random.binomial(n=2, p=0.5, size=num_samples),
    "education_level": np.random.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.2, 0.1], size=num_samples),
    "cert_count": np.random.binomial(n=2, p=0.3, size=num_samples),
    "sentiment_score": np.random.normal(loc=0.2, scale=0.1, size=num_samples).clip(-0.5, 0.5),
    "public_repos": np.random.exponential(scale=5, size=num_samples).clip(0, 20).astype(int),
    "followers": np.random.exponential(scale=50, size=num_samples).clip(0, 1000).astype(int),
    "github_years": np.random.exponential(scale=3, size=num_samples).clip(0, 15).astype(int),
    "avg_stars_per_repo": np.random.exponential(scale=2, size=num_samples).clip(0, 10),
    "connections": np.random.exponential(scale=100, size=num_samples).clip(0, 1000).astype(int),
    "num_skills": np.random.poisson(lam=5 + years_experience * 0.2, size=num_samples).clip(0, 10),
    "has_linkedin": np.random.choice([0, 1], p=[0.1, 0.9], size=num_samples),
    "relevant_skills": np.random.binomial(n=3, p=0.5, size=num_samples)
}
data["relevant_skills"] = np.where(data["has_linkedin"] == 1, data["relevant_skills"], 0)

df = pd.DataFrame(data)
df["resume_score"] = (df["keyword_count"] * 3 + df["years_experience"] * 2 + df["role_count"] * 5 +
                      df["education_level"] * 5 + df["cert_count"] * 5 + df["sentiment_score"] * 10).clip(upper=40)
df["github_score"] = (df["public_repos"].clip(upper=10) + (df["followers"] // 100).clip(upper=10) +
                     df["github_years"].clip(upper=5) + df["avg_stars_per_repo"].clip(upper=5)).clip(upper=30)
df["linkedin_score"] = ((df["connections"] // 50).clip(upper=10) + (df["num_skills"] * 2).clip(upper=10) +
                       df["has_linkedin"] * 5 + df["relevant_skills"] * 2).clip(upper=30)
df["total_score"] = (df["resume_score"] + df["github_score"] + df["linkedin_score"] +
                     np.random.normal(0, 5, num_samples)).clip(0, 100)

# Ensure the 'data' directory exists
if not os.path.exists("data"):
    os.makedirs("data")

df.to_csv("data/training_data.csv", index=False)
print("Training data saved to data/training_data.csv")