# ResumeRanker API

**ResumeRanker** is a FastAPI-based Resume / Job Match Scoring API that parses resumes, evaluates keyword coverage, calculates semantic similarity, and provides ATS-friendly scoring.

---

## 🚀 Features

- Parse resumes from **PDF, DOCX, or plain text**.
- Extract structured sections: **Contact, Summary, Experience, Education, Skills**.
- Score resumes against job descriptions using:
  - **Semantic similarity**
  - **Keyword coverage**
  - **Section-wise scoring**
- Generate **ATS-friendly hints** and explainability text.
- **Batch scoring** of multiple resumes.
- API key-based authentication (RapidAPI-ready).
- **Rate-limited** endpoints.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ResumeRanker.git
cd ResumeRanker

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
⚙ Configuration
Create a .env file in the root directory:

```bash
PORT=8000
DEBUG=True

# Auth
API_KEY_SECRET=your-strong-random-key-here

# Embeddings
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxx

# Optional: database or Redis
DATABASE_URL=sqlite:///./resumeranker.db
REDIS_URL=redis://localhost:6379
```
#⚡ Running the API
```bash
python -m uvicorn app.main:app --reload --port 8000
```
Access Swagger UI: http://127.0.0.1:8000/docs
Access Redoc: http://127.0.0.1:8000/redoc

#🔑 Authentication
```bash
Authorization: Bearer YOUR_API_KEY_SECRET
```
#Example using cURL
```bash
curl -X POST "http://127.0.0.1:8000/v1/score-match" \
-H "accept: application/json" \
-H "Authorization: Bearer YOUR_API_KEY_SECRET" \
-H "Content-Type: application/json" \
-d '{
  "resume_text": "Experienced backend engineer with Python and Django...",
  "job_description": "Looking for a backend engineer skilled in Python, Django, and REST APIs"
}'
```
