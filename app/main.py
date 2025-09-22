import os
import io
import re
import math
import time
import string
import asyncio
from collections import Counter
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

load_dotenv()

# Optional imports for PDF/DOCX parsing
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx2txt
except ImportError:
    docx2txt = None

# ----------------------------
# Load API Key and Environment
# ----------------------------
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
if not API_KEY_SECRET:
    raise RuntimeError("API_KEY_SECRET not set in .env")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ----------------------------
# Security Dependency
# ----------------------------
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="Resume Match API",
    version="1.0.0",
    description="Resume / Job Match Scorer API with RapidAPI-compatible API Key authentication"
)

# ----------------------------
# Pydantic Models
# ----------------------------
class ResumeSection(BaseModel):
    contact: Optional[str] = ""
    summary: Optional[str] = ""
    experience: List[str] = []
    education: List[str] = []
    skills: List[str] = []

class ParseResumeResponse(BaseModel):
    status: str
    parsed_text: str
    sections: ResumeSection

class KeywordCoverage(BaseModel):
    required: List[str]
    missing: List[str]

class MatchedSections(BaseModel):
    experience: float
    skills: float
    education: float

class ScoreMatchRequest(BaseModel):
    resume_text: Optional[str] = None
    job_description: str
    top_k_keywords: int = Field(default=20, ge=1, le=50)
    return_rewrites: bool = False

class ScoreMatchResponse(BaseModel):
    score: float
    semantic_similarity: float
    keyword_coverage: KeywordCoverage
    matched_sections: MatchedSections
    ats_hints: List[str]
    explainability: str

# ----------------------------
# Helpers: Document Parsing
# ----------------------------
class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        if PdfReader is None:
            raise HTTPException(status_code=500, detail="PDF parsing not available")
        pdf_reader = PdfReader(io.BytesIO(file_content))
        return "\n".join(page.extract_text() or "" for page in pdf_reader.pages).strip()

    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        if docx2txt is None:
            raise HTTPException(status_code=500, detail="DOCX parsing not available")
        with io.BytesIO(file_content) as temp_file:
            return docx2txt.process(temp_file).strip()

# ----------------------------
# Common Skills & ATS Hints
# ----------------------------
COMMON_SKILLS = [
    "python", "javascript", "java", "react", "node.js", "sql", "mongodb",
    "docker", "kubernetes", "aws", "azure", "git", "machine learning",
    "data science", "html", "css", "typescript", "angular", "vue",
    "postgresql", "mysql", "redis", "elasticsearch", "microservices",
    "api", "rest", "graphql", "agile", "scrum", "leadership", "management"
]

ATS_HINTS = [
    "Use plain text skills list instead of graphics or tables",
    "Avoid headers and footers with important information",
    "Keep bullet points concise, ideally under 2 lines each",
    "Use standard section headings (Experience, Education, Skills)",
    "Include relevant keywords from the job description",
    "Quantify achievements with specific numbers and metrics",
    "Use consistent formatting throughout the document",
    "Save as a .docx or .pdf for better ATS compatibility"
]

# ----------------------------
# Resume Parsing Logic
# ----------------------------
class ResumeParser:
    @staticmethod
    def parse_resume_text(text: str) -> ResumeSection:
        text_lower = text.lower()
        # Extract contact info
        contact_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'linkedin\.com/in/[\w-]+',
            r'github\.com/[\w-]+'
        ]
        contact = " ".join(sum([re.findall(p, text) for p in contact_patterns], []))

        # Extract skills
        skills = []
        for skill in COMMON_SKILLS:
            if skill.lower() in text_lower:
                skills.append(skill)
        skills = list(set(skills))[:20]

        # Simple experience / education extraction
        experience = re.findall(r'(?i)(?:work\s+)?experience:?\s*(.+?)(?=\neducation|\nskills|$)', text, re.DOTALL)
        education = re.findall(r'(?i)education:?\s*(.+?)(?=\nexperience|\nskills|$)', text, re.DOTALL)

        return ResumeSection(
            contact=contact.strip(),
            summary=text[:500],
            experience=[e.strip() for e in experience[:5]],
            education=[e.strip() for e in education[:3]],
            skills=skills
        )

# ----------------------------
# Keyword Extraction
# ----------------------------
class KeywordExtractor:
    @staticmethod
    def extract_keywords(text: str, top_k: int = 20) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        freq = Counter(words)
        return [word for word, _ in freq.most_common(top_k)]

# ----------------------------
# Scoring Engine
# ----------------------------
class MatchingEngine:
    @staticmethod
    async def calculate_semantic_similarity(resume_text: str, job_description: str) -> float:
        # Fallback token-based similarity
        resume_set = set(resume_text.lower().split())
        job_set = set(job_description.lower().split())
        inter = len(resume_set & job_set)
        union = len(resume_set | job_set)
        return inter / union if union > 0 else 0.0

    @staticmethod
    def calculate_keyword_coverage(resume_text: str, job_keywords: List[str]):
        resume_lower = resume_text.lower()
        matched = [k for k in job_keywords if k.lower() in resume_lower]
        missing = [k for k in job_keywords if k not in matched]
        coverage_ratio = len(matched) / len(job_keywords) if job_keywords else 0
        return matched, missing, coverage_ratio

    @staticmethod
    async def calculate_section_scores(resume_sections: ResumeSection, job_description: str) -> MatchedSections:
        score = await MatchingEngine.calculate_semantic_similarity(
            " ".join(resume_sections.experience), job_description
        )
        skill_score = await MatchingEngine.calculate_semantic_similarity(
            " ".join(resume_sections.skills), job_description
        )
        edu_score = await MatchingEngine.calculate_semantic_similarity(
            " ".join(resume_sections.education), job_description
        )
        return MatchedSections(
            experience=round(score, 2),
            skills=round(min(skill_score*1.2,1.0),2),
            education=round(edu_score,2)
        )

# ----------------------------
# Utility Functions
# ----------------------------
def calculate_final_score(semantic_similarity: float, keyword_coverage_ratio: float, section_scores: MatchedSections) -> float:
    section_score = section_scores.experience*0.5 + section_scores.skills*0.3 + section_scores.education*0.1
    final_score = 0.6*semantic_similarity + 0.3*keyword_coverage_ratio + 0.1*section_score
    return round(min(final_score*100,100),1)

def generate_explainability(resume_text: str, job_description: str, matched_keywords: List[str]) -> str:
    parts = []
    if matched_keywords:
        parts.append(f"Matched keywords: {', '.join(matched_keywords[:3])}")
    return "; ".join(parts) if parts else "Limited overlap found"

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
async def root():
    return {"message":"Resume Match API running"}

@app.post("/v1/parse-resume")
async def parse_resume(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
) -> ParseResumeResponse:
    if file:
        content = await file.read()
        if file.content_type == "application/pdf":
            text = DocumentParser.extract_text_from_pdf(content)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = DocumentParser.extract_text_from_docx(content)
        else:
            text = content.decode('utf-8', errors='ignore')
    elif not text:
        raise HTTPException(status_code=400, detail="File or text required")
    sections = ResumeParser.parse_resume_text(text)
    return ParseResumeResponse(status="success", parsed_text=text[:1000], sections=sections)

@app.post("/v1/score-match")
async def score_match(
    request: ScoreMatchRequest,
    api_key: str = Depends(verify_api_key)
) -> ScoreMatchResponse:
    resume_text = request.resume_text or ""
    job_desc = request.job_description

    # Keywords
    job_keywords = KeywordExtractor.extract_keywords(job_desc, top_k=request.top_k_keywords)
    matched_keywords, missing_keywords, coverage_ratio = MatchingEngine.calculate_keyword_coverage(resume_text, job_keywords)

    # Sections
    resume_sections = ResumeParser.parse_resume_text(resume_text)
    section_scores = await MatchingEngine.calculate_section_scores(resume_sections, job_desc)

    # Semantic similarity
    semantic_sim = await MatchingEngine.calculate_semantic_similarity(resume_text, job_desc)

    # Final score
    final_score = calculate_final_score(semantic_sim, coverage_ratio, section_scores)

    explainability = generate_explainability(resume_text, job_desc, matched_keywords)

    return ScoreMatchResponse(
        score=final_score,
        semantic_similarity=round(semantic_sim,2),
        keyword_coverage=KeywordCoverage(required=matched_keywords, missing=missing_keywords),
        matched_sections=section_scores,
        ats_hints=ATS_HINTS,
        explainability=explainability
    )
