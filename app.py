from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from dotenv import load_dotenv
import os
import re
import requests
import joblib
import numpy as np
import torch
import torch.nn as nn
from textblob import TextBlob
import shap
import pandas as pd
import fitz
from google.generativeai import GenerativeModel, configure
import json
import time
import logging
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "User-Agent": "SmartHireAI"}
configure(api_key=os.getenv("GEMINI_API_KEY"))
chatbot_model = GenerativeModel("gemini-1.5-flash")
conversation_history = []

class CandidateScorer(nn.Module):
    def __init__(self, input_size):
        super(CandidateScorer, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

scaler = joblib.load("data/scaler.pkl")
model = CandidateScorer(input_size=15)
model.load_state_dict(torch.load("data/candidate_scorer_model.pth"))
model.eval()

features = ["keyword_count", "years_experience", "word_count", "role_count", "education_level",
            "cert_count", "sentiment_score", "public_repos", "followers", "github_years",
            "avg_stars_per_repo", "connections", "num_skills", "has_linkedin", "relevant_skills"]

df = pd.read_csv("data/training_data.csv")
background_data = scaler.transform(np.array(df[features].iloc[:100]))
background_data = torch.FloatTensor(background_data)
explainer = shap.KernelExplainer(lambda x: model(torch.FloatTensor(x)).detach().numpy(), background_data.numpy())

SKILL_TO_ROLES = {
    "python": ["Data Scientist", "Machine Learning Engineer", "Backend Developer"],
    "javascript": ["Frontend Developer", "Full Stack Developer"],
    "java": ["Backend Developer", "Software Engineer"],
    "c++": ["Software Engineer", "Systems Engineer"],
    "sql": ["Data Analyst", "Database Administrator"],
    "react": ["Frontend Developer", "Full Stack Developer"],
    "node": ["Backend Developer", "Full Stack Developer"],
    "django": ["Backend Developer", "Full Stack Developer"],
    "flask": ["Backend Developer", "Full Stack Developer"],
    "ai": ["Machine Learning Engineer", "AI Engineer"],
    "power bi": ["Data Analyst", "Business Intelligence Analyst"],
    "tableau": ["Data Analyst", "Business Intelligence Analyst"],
    "pandas": ["Data Scientist", "Data Analyst"],
    "numpy": ["Data Scientist", "Data Analyst"],
    "matplotlib": ["Data Scientist", "Data Analyst"],
    "aws": ["Cloud Engineer", "DevOps Engineer"],
    "docker": ["DevOps Engineer", "Backend Developer"],
    "tensorflow": ["Machine Learning Engineer", "AI Engineer"],
    "c": ["Software Engineer", "Systems Engineer"]
}

def recommend_job_roles(skills):
    recommended_roles = set()
    for skill in skills:
        if skill.lower() in SKILL_TO_ROLES:
            recommended_roles.update(SKILL_TO_ROLES[skill.lower()])
    return sorted(list(recommended_roles))[:3] or ["General Technical Role"]

def fetch_github_data(username):
    github_url = f"https://api.github.com/users/{username}"
    retry_count = 0
    max_retries = 3
    github_data = {}

    while retry_count < max_retries:
        try:
            logger.debug(f"Fetching GitHub profile for {username}")
            response = requests.get(github_url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                github_data = response.json()
                break
            elif response.status_code == 403:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - time.time(), 0) + 1
                logger.warning(f"Rate limit exceeded for {username}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"GitHub API error for {username}: {response.status_code}")
                retry_count += 1
                time.sleep(5)
        except requests.RequestException as e:
            logger.error(f"Request error for {username}: {e}")
            retry_count += 1
            time.sleep(5)

    if not github_data:
        logger.error(f"Failed to fetch GitHub profile for {username}")
        return {"public_repos": 0, "followers": 0, "created_at": "2025", "public_gists": 0}

    try:
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"
        logger.debug(f"Fetching repos for {username}")
        repos_response = requests.get(repos_url, headers=HEADERS, timeout=10)
        if repos_response.status_code == 200:
            repos = repos_response.json()
            total_stars = sum(repo.get("stargazers_count", 0) for repo in repos)
            total_commits = 0
            for repo in repos[:5]:
                commits_url = f"https://api.github.com/repos/{username}/{repo['name']}/commits?per_page=100"
                try:
                    commits_response = requests.get(commits_url, headers=HEADERS, timeout=10)
                    if commits_response.status_code == 200:
                        total_commits += len(commits_response.json())
                    else:
                        logger.warning(f"Failed to fetch commits for {repo['name']}")
                except requests.RequestException as e:
                    logger.error(f"Commit fetch error for {repo['name']}: {e}")
            github_data["avg_stars_per_repo"] = total_stars / len(repos) if repos else 0
            github_data["total_commits"] = min(total_commits, 1000)
            github_data["languages"] = list(set(repo.get("language", None) for repo in repos if repo.get("language")))
            github_data["top_repos"] = sorted(
                [{"name": r["name"], "stars": r["stargazers_count"]} for r in repos],
                key=lambda x: x["stars"],
                reverse=True
            )[:3]
        else:
            github_data["avg_stars_per_repo"] = 0
            github_data["total_commits"] = 0
            github_data["languages"] = []
            github_data["top_repos"] = []
    except Exception as e:
        logger.error(f"Repo fetch error for {username}: {e}")
        github_data["avg_stars_per_repo"] = 0
        github_data["total_commits"] = 0
        github_data["languages"] = []
        github_data["top_repos"] = []

    github_data["github_years"] = max(2025 - int(github_data.get("created_at", "2025")[:4]), 0)
    logger.info(f"GitHub data for {username}: repos={github_data.get('public_repos', 0)}, commits={github_data.get('total_commits', 0)}")
    return github_data

def normalize_skills(skills):
    skill_map = {
        "cc": "C", "c": "C", "ai": "AI", "python": "Python", "sql": "SQL",
        "tableau": "Tableau", "power bi": "Power BI", "pandas": "Pandas",
        "numpy": "NumPy", "matplotlib": "Matplotlib"
    }
    normalized = set(skill_map.get(skill.lower(), skill.capitalize()) for skill in skills)
    return sorted(list(normalized))

def categorize_resume_score(score):
    if score < 20:
        return "Low"
    elif score <= 30:
        return "Average"
    else:
        return "High"

def evaluate_resume_format(pdf_document):
    score = 0
    fonts = set()
    font_sizes = set()
    line_spacings = []
    page_margins = []

    try:
        for page in pdf_document:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            fonts.add(span["font"])
                            font_sizes.add(span["size"])
                        if len(block["lines"]) > 1:
                            y0 = line["bbox"][1]
                            next_line = block["lines"][block["lines"].index(line) + 1]["bbox"][1] if block["lines"].index(line) + 1 < len(block["lines"]) else y0
                            line_spacings.append(abs(next_line - y0))

            page_rect = page.rect
            text_rect = page.get_text("text").strip().splitlines()
            if text_rect:
                margin_left = page.search_for(text_rect[0])[0].x0 if page.search_for(text_rect[0]) else 0
                margin_right = page_rect.width - (page.search_for(text_rect[-1])[0].x1 if page.search_for(text_rect[-1]) else page_rect.width)
                page_margins.append((margin_left, margin_right))

        if len(fonts) <= 2:
            score += 3
        elif len(fonts) <= 3:
            score += 2
        if len(font_sizes) <= 4:
            score += 3
        elif len(font_sizes) <= 6:
            score += 2
        if page_margins:
            left_margins = [m[0] for m in page_margins]
            right_margins = [m[1] for m in page_margins]
            margin_variance = max(max(left_margins) - min(left_margins), max(right_margins) - min(right_margins))
            if margin_variance < 20:
                score += 3
            elif margin_variance < 40:
                score += 2
        if line_spacings:
            spacing_variance = max(line_spacings) - min(line_spacings)
            if spacing_variance < 5:
                score += 3
            elif spacing_variance < 10:
                score += 2
        section_headers = ["education", "experience", "skills", "projects", "certifications"]
        text_lower = pdf_document.get_page_text(0).lower()
        headers_found = sum(1 for header in section_headers if header in text_lower)
        if headers_found >= 4:
            score += 3
        elif headers_found >= 2:
            score += 2
    except Exception as e:
        logger.error(f"Resume format evaluation failed: {str(e)}")
        score = 5
    return min(score, 15)

def extract_features(resume_text, github_data, linkedin_data, pdf_document):
    features = {}
    resume_text_lower = resume_text.lower()
    keywords = list(SKILL_TO_ROLES.keys())
    features["keyword_count"] = sum(1 for keyword in keywords if keyword in resume_text_lower)
    experience_matches = re.findall(r'(\d+\+?\s*(?:years|yrs|months))', resume_text_lower, re.IGNORECASE)
    total_years = sum(int(m.split()[0].rstrip('+')) / (12 if "months" in m.lower() else 1) for m in experience_matches)
    features["years_experience"] = min(total_years, 20)
    features["word_count"] = len(re.split(r'\s+', resume_text.strip()))
    role_keywords = ["engineer", "developer", "analyst", "scientist", "manager"]
    features["role_count"] = sum(1 for role in role_keywords if role in resume_text_lower)
    education_keywords = ["certificate", "diploma", "bachelor", "master", "phd"]
    features["education_level"] = max([0] + [1 + i for i, edu in enumerate(education_keywords) if edu in resume_text_lower])
    cert_keywords = ["google", "sololearn", "aws", "pmp", "certified"]
    features["cert_count"] = sum(1 for cert in cert_keywords if cert in resume_text_lower)
    features["sentiment_score"] = max(TextBlob(resume_text).sentiment.polarity, 0)

    if github_data:
        features["public_repos"] = min(github_data.get("public_repos", 0), 100)
        features["followers"] = min(github_data.get("followers", 0), 1000)
        features["github_years"] = github_data.get("github_years", 0)
        features["avg_stars_per_repo"] = min(github_data.get("avg_stars_per_repo", 0), 50)
    else:
        features["public_repos"] = 0
        features["followers"] = 0
        features["github_years"] = 0
        features["avg_stars_per_repo"] = 0

    if linkedin_data:
        features["connections"] = min(linkedin_data.get("mock_connections", 0), 1000)
        features["num_skills"] = len(set(linkedin_data.get("mock_skills", [])))
        features["has_linkedin"] = 1
        features["relevant_skills"] = sum(1 for skill in linkedin_data.get("mock_skills", []) if skill.lower() in SKILL_TO_ROLES)
    else:
        features["connections"] = 0
        features["num_skills"] = 0
        features["has_linkedin"] = 0
        features["relevant_skills"] = 0

    feature_array = np.array([[features[feat] for feat in features]])
    feature_array_scaled = scaler.transform(feature_array)
    return torch.FloatTensor(feature_array_scaled), features, pdf_document

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    pdf_stream = BytesIO(file.read())
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
    text = "".join(page.get_text() for page in pdf)

    github_pattern = r'(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_-]+)(?:/|\s|$)'
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/([\w-]+)'
    skills_pattern = r'(?:' + '|'.join(SKILL_TO_ROLES.keys()) + r')'
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    phone_pattern = r'(?:\+\d{1,3}\s?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}'
    name_pattern = r'^\s*([A-Z][a-zA-Z\'-]+(?: |$)(?:[A-Z][a-zA-Z\'-]+)?)\s*(?:$|\n)'
    cert_pattern = r'(?:(?:Google|AWS|Microsoft|Coursera|edX|Udemy|SoloLearn)\s+[\w\s]+(?:Certification|Certificate|Professional))|(?:PMP|Certified\s+[\w\s]+(?:Analyst|Engineer|Developer|Professional))'

    github_match = re.search(github_pattern, text, re.IGNORECASE)
    linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
    skills = normalize_skills(re.findall(skills_pattern, text, re.IGNORECASE))
    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    name_matches = re.findall(name_pattern, text, re.MULTILINE)
    cert_matches = re.findall(cert_pattern, text, re.IGNORECASE)

    github_data = {}
    if github_match:
        github_username = github_match.group(1)
        logger.debug(f"Extracted GitHub username: {github_username}")
        github_data = fetch_github_data(github_username)
        if not github_data:
            logger.warning(f"No GitHub data for {github_username}")

    linkedin_data = {}
    if linkedin_match:
        linkedin_data = {
            "username": linkedin_match.group(1),
            "mock_connections": min(len(skills) * 50 + 100, 700),
            "mock_skills": skills,
            "mock_endorsements": len(skills) * 10,
            "mock_headline": f"{skills[0]} Professional" if skills else "Professional"
        }

    name = None
    name_candidates = []
    header_text = text[:300]
    for candidate in re.findall(name_pattern, header_text, re.MULTILINE):
        candidate = candidate.strip()
        if (len(candidate) > 2 and len(candidate) < 30 and
            not any(keyword in candidate.lower() for keyword in ["mobile", "email", "phone", "address", "resume", "summary", "objective", "linkedin", "github"])):
            name_candidates.append((candidate, 0))

    for i, (candidate, score) in enumerate(name_candidates):
        if text.lower().find(candidate.lower()) < 100:
            score += 5
        if len(candidate.split()) >= 2:
            score += 3
        elif len(candidate.split()) == 1 and len(candidate) >= 5:
            score += 2
        if email and candidate.lower().replace(" ", "").startswith(
            re.sub(r'\d+', '', email.group(0).split("@")[0]).lower()[:len(candidate.replace(" ", ""))]
        ):
            score += 4
        for page in pdf[:1]:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if candidate.lower() in span["text"].lower() and span["size"] > 12:
                                score += 3
        name_candidates[i] = (candidate, score)

    if name_candidates:
        name = max(name_candidates, key=lambda x: x[1])[0]
    if not name and email:
        email_prefix = re.sub(r'\d+', '', email.group(0).split("@")[0])
        name = email_prefix.replace(".", " ").title().strip()
    if not name and linkedin_data.get("username"):
        name = linkedin_data["username"].replace("-", " ").title()
    if not name:
        name = "Candidate"

    certs = []
    for cert in cert_matches:
        cert = cert.strip()
        if (len(cert.split()) >= 3 or any(term in cert.lower() for term in ["certificate", "certification", "professional"])) and len(cert) > 10:
            if not any(invalid in cert.lower() for invalid in ["in progress", "pursuing", "enrolled"]):
                certs.append(cert)
    certs = list(dict.fromkeys(certs))[:5] or ["None listed"]

    features_scaled, feature_dict, pdf_document = extract_features(text, github_data, linkedin_data, pdf)
    with torch.no_grad():
        score = model(features_scaled).item() * 100

    resume_score = 0
    resume_score += evaluate_resume_format(pdf_document)
    resume_score += min(feature_dict["keyword_count"], 5) * 1.5
    resume_score += min(feature_dict["cert_count"], 3) * 1.5
    word_count = feature_dict["word_count"]
    if 300 <= word_count <= 600:
        resume_score += 8
    elif 200 <= word_count < 800:
        resume_score += 6
    else:
        resume_score += 4
    resume_score += min(feature_dict["sentiment_score"] * 5, 5)
    resume_score = min(resume_score, 40)

    github_score = min(
        (feature_dict["public_repos"] * 0.5 + 
         feature_dict["followers"] * 0.05 + 
         feature_dict["avg_stars_per_repo"] * 0.4 + 
         github_data.get("total_commits", 0) * 0.01 + 
         feature_dict["github_years"] * 2),
        40  # Updated maximum GitHub score to 40
    ) if github_data else 0
    linkedin_score = min(
        (feature_dict["connections"] * 0.01 + 
         feature_dict["num_skills"] * 1.0 + 
         feature_dict["relevant_skills"] * 1.2),
        20  # Updated maximum LinkedIn score to 20
    ) if linkedin_data else 0
    total_score = int(min(resume_score + github_score + linkedin_score, 100))
    fit_score = min((feature_dict["relevant_skills"] * 7), 70) if skills else 0
    resume_score_category = categorize_resume_score(resume_score)

    strengths = sorted([(k, v) for k, v in feature_dict.items()], key=lambda x: x[1], reverse=True)[:3]
    weaknesses = [
        "Limited GitHub activity" if github_score < 10 else None,
        "Few certifications" if feature_dict["cert_count"] < 2 else None,
        "Limited experience" if feature_dict["years_experience"] < 2 else None
    ]
    weaknesses = [w for w in weaknesses if w]
    roles = recommend_job_roles(skills)
    questions = [
        f"Describe your experience with {skills[0]} projects." if skills else "Tell me about your technical skills.",
        "What’s your most starred GitHub project?" if github_data.get("top_repos") else "Any coding projects to share?",
        f"How do you approach data analysis?" if "analyst" in roles else "How do you solve complex problems?"
    ]

    skill_confidence = {skill: min(100, 50 + 10 * text.lower().count(skill.lower())) for skill in skills[:4]} if skills else {}

    analysis_data = {
        "total_score": total_score,
        "resume_score": resume_score,
        "resume_score_category": resume_score_category,
        "github_score": github_score,
        "linkedin_score": linkedin_score,
        "fit_score": fit_score,
        "total_progress": total_score,
        "resume_progress": (resume_score / 40 * 100),
        "github_progress": (github_score / 40 * 100),  # Updated to reflect new max of 40
        "linkedin_progress": (linkedin_score / 20 * 100),  # Updated to reflect new max of 20
        "strengths": [f"{k.replace('_', ' ').title()}: {v:.1f}" for k, v in strengths],
        "weaknesses": weaknesses,
        "recommended_roles": ", ".join(roles),
        "resume_text": text[:500] + "..." if len(text) > 500 else text,
        "github": {
            "username": github_data.get("login", "N/A"),
            "repos": github_data.get("public_repos", 0),
            "followers": github_data.get("followers", 0),
            "following": github_data.get("following", 0),
            "created": github_data.get("created_at", "N/A"),
            "bio": github_data.get("bio", "No bio provided"),
            "gists": github_data.get("public_gists", 0),
            "stars": github_data.get("avg_stars_per_repo", 0),
            "commits": github_data.get("total_commits", 0),
            "languages": github_data.get("languages", []),
            "top_repos": github_data.get("top_repos", [])
        },
        "linkedin": linkedin_data,
        "interview_questions": questions,
        "trajectory": f"{'Mid' if feature_dict['years_experience'] > 3 else 'Entry'}-level {roles[0] if roles else 'Professional'}",
        "name": name,
        "email": email.group(0) if email else "N/A",
        "phone": phone.group(0) if phone else "N/A",
        "years_experience": feature_dict["years_experience"],
        "education_level": feature_dict["education_level"],
        "key_achievement": "Developed SmartHireAI chatbot for resume analysis" if "SmartHireAI" in text else "Contributed to data-driven projects",
        "skill_confidence": skill_confidence,
        "certifications": certs
    }
    session['analysis_data'] = analysis_data
    pdf.close()
    return jsonify({"message": "Upload successful", "redirect": url_for('result')})

@app.route('/result')
def result():
    if 'analysis_data' not in session:
        logger.error("No analysis_data in session for /result")
        return redirect(url_for('upload'))
    analysis_data = session['analysis_data']
    return render_template('result.html', analysis_data=analysis_data)

@app.route('/download_report')
def download_report():
    if 'analysis_data' not in session:
        logger.error("No analysis_data in session for /download_report")
        return redirect(url_for('upload'))
    try:
        analysis_data = session['analysis_data']
        logger.info(f"Generating PDF with keys: {list(analysis_data.keys())}")

        # Create PDF with reportlab
        pdf_output = BytesIO()
        c = canvas.Canvas(pdf_output, pagesize=letter)
        width, height = letter
        y = height - 50
        line_height = 14

        def draw_text(text, font_size, bold=False):
            nonlocal y
            text = text.encode('ascii', 'ignore').decode('ascii')
            c.setFont("Helvetica-Bold" if bold else "Helvetica", font_size)
            text_object = c.beginText(50, y)
            text_object.setFont("Helvetica-Bold" if bold else "Helvetica", font_size)
            text_object.setLeading(line_height)
            for line in text.split('\n'):
                text_object.textLine(line)
                y -= line_height
            c.drawText(text_object)
            logger.debug(f"Drew text: {text}")

        # Title
        name = analysis_data.get('name', 'Candidate')
        draw_text(f"Talent Spotlight: {name}", 16, bold=True)
        y -= 10

        # Candidate Snapshot
        draw_text("Candidate Snapshot", 14, bold=True)
        skills = analysis_data.get('linkedin', {}).get('mock_skills', [])[:3] or ['technical expertise']
        certs = analysis_data.get('certifications', ['relevant certifications'])[:1]
        role = analysis_data.get('recommended_roles', 'data-driven roles').split(', ')[0]
        snapshot = f"{name} excels in {', '.join(skills)}, supported by {', '.join(certs)}.\nStrong fit for {role}."
        draw_text(snapshot, 12)
        y -= 10

        # Score Breakdown
        draw_text("Score Breakdown", 14, bold=True)
        draw_text(f"Total Score: {analysis_data.get('total_score', 0)}/100", 12)
        draw_text(f"Resume Score: {analysis_data.get('resume_score', 0):.1f}/40 ({analysis_data.get('resume_score_category', 'N/A')})", 12)
        draw_text(f"GitHub Score: {analysis_data.get('github_score', 0):.1f}/40", 12)  # Updated to reflect new max of 40
        draw_text(f"LinkedIn Score: {analysis_data.get('linkedin_score', 0):.1f}/20", 12)  # Updated to reflect new max of 20
        y -= 10

        # Top Skills
        draw_text("Top Skills", 14, bold=True)
        skills = skills[:4] or ["N/A"]
        for skill in skills:
            draw_text(f"- {skill}", 12)
        y -= 10

        # Resume Details
        draw_text("Resume Details", 14, bold=True)
        draw_text(f"Full Name: {name}", 12)
        draw_text(f"Email: {analysis_data.get('email', 'N/A')}", 12)
        draw_text(f"Phone: {analysis_data.get('phone', 'N/A')}", 12)
        draw_text(f"Experience: {analysis_data.get('years_experience', 0):.1f} years", 12)
        draw_text(f"Education: {'Bachelor' if analysis_data.get('education_level', 0) == 3 else 'Other'}", 12)
        y -= 10

        # GitHub Insights
        draw_text("GitHub Insights", 14, bold=True)
        github = analysis_data.get('github', {})
        draw_text(f"Username: {github.get('username', 'N/A')}", 12)
        draw_text(f"Repositories: {github.get('repos', 0)}", 12)
        draw_text(f"Followers: {github.get('followers', 0)}", 12)
        draw_text(f"Commits: {github.get('commits', 0)}", 12)
        y -= 10

        # LinkedIn Profile
        draw_text("LinkedIn Profile", 14, bold=True)
        linkedin = analysis_data.get('linkedin', {})
        draw_text(f"Username: {linkedin.get('username', 'N/A')}", 12)
        draw_text(f"Connections: {linkedin.get('mock_connections', 0)}", 12)
        draw_text(f"Headline: {linkedin.get('mock_headline', 'N/A')}", 12)

        c.showPage()
        c.save()
        pdf_output.seek(0)
        logger.info("PDF generated successfully")

        return send_file(
            pdf_output,
            download_name='report.pdf',
            as_attachment=True,
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        return jsonify({"error": "Failed to generate PDF. Please try again."}), 500

@app.route('/schedule_interview', methods=['GET', 'POST'])
def schedule_interview():
    if request.method == 'POST':
        interview_time = request.form.get('interview_time')
        session['interview_time'] = interview_time
        return redirect(url_for('schedule_confirmation'))
    return render_template('schedule.html')

@app.route('/schedule_confirmation')
def schedule_confirmation():
    interview_time = session.get('interview_time', 'Not set')
    return render_template('schedule_confirmation.html', interview_time=interview_time)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        analysis_data = session.get('analysis_data', {})

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        conversation_history.append(f"User: {user_message}")
        if len(conversation_history) > 50:
            conversation_history.pop(0)

        prompt = f"""You’re a SmartHire AI chatbot for hiring and career insights. 
        User said: "{user_message}". 
        Respond naturally:
        - Short input? Quick, max 10 words, unique twist.
        - Detailed input? Markdown, 3-5 bullets or steps, clear vibe.
        - Off-topic? Nudge back to hiring/career, keep it fun.
        Use resume data if relevant: {json.dumps(analysis_data, indent=2)}.
        Suggest assignments (3-5 numbered) if asked, based on skills.
        History: {chr(10).join(conversation_history)}."""

        result = chatbot_model.generate_content(prompt)
        bot_message = result.text
        conversation_history.append(f"Bot: {bot_message}")
        return jsonify({'message': bot_message})
    except Exception as e:
        return jsonify({'error': f"Chat error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)