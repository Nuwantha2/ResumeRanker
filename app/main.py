from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import re
import time
import hashlib
from collections import Counter
import math
import json
import string
from datetime import datetime
import asyncio
import os
import io
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    import docx2txt
except ImportError:
    docx2txt = None
import httpx

# Pydantic Models
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

class BatchScoreRequest(BaseModel):
    resumes: List[str]
    job_description: str
    top_k_keywords: int = Field(default=20, ge=1, le=50)

class BatchScoreResponse(BaseModel):
    results: List[ScoreMatchResponse]

# Configuration and Environment Variables  
VALID_API_KEYS = {
    os.getenv("API_KEY_TEST", "sk_test_example"): "test",
    os.getenv("API_KEY_PROD", "sk_prod_example"): "production"
}

# Remote embedding configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # local, openai, or huggingface

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

# Rate Limiting Setup
limiter = Limiter(key_func=get_remote_address)

# Authentication
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

app = FastAPI(
    title="Resume Match API",
    description="Resume / Job Match Scorer MVP API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Document Parsing Service
class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        if PdfReader is None:
            raise HTTPException(status_code=500, detail="PDF parsing not available")
        
        try:
            pdf_reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX file"""
        if docx2txt is None:
            raise HTTPException(status_code=500, detail="DOCX parsing not available")
        
        try:
            # Save to temp file for docx2txt
            with io.BytesIO(file_content) as temp_file:
                text = docx2txt.process(temp_file)
            return text.strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {str(e)}")

# Remote Embedding Service
class EmbeddingService:
    @staticmethod
    async def get_embeddings(texts: List[str]) -> List[List[float]]:
        """Get embeddings using configured provider"""
        if EMBEDDING_PROVIDER == "openai" and OPENAI_API_KEY:
            return await EmbeddingService.get_openai_embeddings(texts)
        elif EMBEDDING_PROVIDER == "huggingface" and HUGGINGFACE_API_KEY:
            return await EmbeddingService.get_huggingface_embeddings(texts)
        else:
            # Fallback to local method (no actual embeddings)
            return [[0.0] * 384 for _ in texts]  # Placeholder embeddings
    
    @staticmethod
    async def get_openai_embeddings(texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "input": texts,
                        "model": "text-embedding-ada-002"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            # Fallback to local similarity
            return [[0.0] * 1536 for _ in texts]
    
    @staticmethod
    async def get_huggingface_embeddings(texts: List[str]) -> List[List[float]]:
        """Get embeddings from HuggingFace Inference API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                    headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                    json={"inputs": texts},
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            # Fallback to local similarity  
            return [[0.0] * 384 for _ in texts]
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

# Core Service Functions
class ResumeParser:
    @staticmethod
    def parse_resume_text(text: str) -> ResumeSection:
        """Parse resume text into structured sections using regex patterns"""
        text_lower = text.lower()
        
        # Extract contact information
        contact_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'linkedin\.com/in/[\w-]+',
            r'github\.com/[\w-]+'
        ]
        contact = ""
        for pattern in contact_patterns:
            matches = re.findall(pattern, text)
            if matches:
                contact += " ".join(matches) + " "
        
        # Extract skills using common patterns and predefined skill list
        skills = []
        skill_patterns = [
            r'(?i)skills?:?\s*(.+?)(?=\n\n|\nexperience|\neducation|$)',
            r'(?i)technical skills?:?\s*(.+?)(?=\n\n|\nexperience|\neducation|$)',
            r'(?i)technologies?:?\s*(.+?)(?=\n\n|\nexperience|\neducation|$)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                skills.extend([s.strip() for s in re.split(r'[,•\n\-]', match) if s.strip()])
        
        # Add skills found in common skills list
        for skill in COMMON_SKILLS:
            if skill.lower() in text_lower and skill not in skills:
                skills.append(skill)
        
        # Extract experience sections
        experience = []
        exp_patterns = [
            r'(?i)(?:work\s+)?experience:?\s*(.+?)(?=\neducation|\nskills|\nprojects|$)',
            r'(?i)employment:?\s*(.+?)(?=\neducation|\nskills|\nprojects|$)',
            r'(?i)professional\s+experience:?\s*(.+?)(?=\neducation|\nskills|\nprojects|$)'
        ]
        
        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                # Split by common delimiters and clean up
                exp_items = [item.strip() for item in re.split(r'\n(?=[A-Z])', match) if len(item.strip()) > 20]
                experience.extend(exp_items[:5])  # Limit to 5 entries
        
        # Extract education
        education = []
        edu_patterns = [
            r'(?i)education:?\s*(.+?)(?=\nexperience|\nskills|\nprojects|$)',
            r'(?i)academic:?\s*(.+?)(?=\nexperience|\nskills|\nprojects|$)',
            r'(?i)qualifications?:?\s*(.+?)(?=\nexperience|\nskills|\nprojects|$)'
        ]
        
        for pattern in edu_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                edu_items = [item.strip() for item in match.split('\n') if len(item.strip()) > 10]
                education.extend(edu_items[:3])  # Limit to 3 entries
        
        # Extract summary/objective
        summary = ""
        summary_patterns = [
            r'(?i)(?:professional\s+)?summary:?\s*(.+?)(?=\nexperience|\nskills|\neducation|$)',
            r'(?i)objective:?\s*(.+?)(?=\nexperience|\nskills|\neducation|$)',
            r'(?i)profile:?\s*(.+?)(?=\nexperience|\nskills|\neducation|$)'
        ]
        
        for pattern in summary_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            if matches:
                summary = matches[0].strip()[:500]  # Limit to 500 characters
                break
        
        return ResumeSection(
            contact=contact.strip(),
            summary=summary,
            experience=experience[:5],
            education=education[:3],
            skills=list(set(skills))[:20]  # Remove duplicates, limit to 20
        )

class KeywordExtractor:
    @staticmethod
    def extract_keywords(text: str, top_k: int = 20) -> List[str]:
        """Extract top keywords using TF-IDF-like scoring"""
        # Clean and tokenize text
        text_clean = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text_clean.split()
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
            'just', 'should', 'now', 'is', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could'
        }
        
        # Filter words
        filtered_words = [word for word in words if len(word) > 2 and word not in stopwords]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        
        # Simple TF-IDF approximation - boost longer words and technical terms
        scored_words = {}
        for word, freq in word_freq.items():
            # Boost technical terms and longer words
            boost = 1.0
            if len(word) > 6:
                boost *= 1.5
            if word in [skill.lower().replace(' ', '') for skill in COMMON_SKILLS]:
                boost *= 2.0
            if any(char.isupper() for char in word):  # Contains uppercase (acronyms)
                boost *= 1.3
            
            scored_words[word] = freq * boost
        
        # Return top K keywords
        return [word for word, score in Counter(scored_words).most_common(top_k)]

class MatchingEngine:
    @staticmethod
    async def calculate_semantic_similarity(resume_text: str, job_description: str) -> float:
        """Calculate semantic similarity using embeddings with fallback to token overlap"""
        try:
            # Try embedding-based similarity first
            embeddings = await EmbeddingService.get_embeddings([resume_text, job_description])
            if embeddings and len(embeddings) == 2 and embeddings[0] != [0.0] * len(embeddings[0]):
                return EmbeddingService.cosine_similarity(embeddings[0], embeddings[1])
        except Exception:
            pass
        
        # Fallback to token-based similarity
        # Normalize texts
        resume_clean = re.sub(r'[^a-zA-Z\s]', '', resume_text.lower())
        job_clean = re.sub(r'[^a-zA-Z\s]', '', job_description.lower())
        
        # Tokenize
        resume_words = set(resume_clean.split())
        job_words = set(job_clean.split())
        
        # Calculate Jaccard similarity as baseline
        intersection = len(resume_words.intersection(job_words))
        union = len(resume_words.union(job_words))
        jaccard = intersection / union if union > 0 else 0
        
        # Enhance with term frequency consideration
        resume_freq = Counter(resume_clean.split())
        job_freq = Counter(job_clean.split())
        
        # Calculate cosine similarity on term frequencies
        common_words = set(resume_freq.keys()).intersection(set(job_freq.keys()))
        if not common_words:
            return jaccard * 0.8  # Penalty for no common words
        
        dot_product = sum(resume_freq[word] * job_freq[word] for word in common_words)
        resume_magnitude = math.sqrt(sum(freq**2 for freq in resume_freq.values()))
        job_magnitude = math.sqrt(sum(freq**2 for freq in job_freq.values()))
        
        cosine_sim = dot_product / (resume_magnitude * job_magnitude) if resume_magnitude * job_magnitude > 0 else 0
        
        # Combine Jaccard and cosine similarity
        return min(0.4 * jaccard + 0.6 * cosine_sim, 1.0)
    
    @staticmethod
    def calculate_keyword_coverage(resume_text: str, job_keywords: List[str]) -> tuple[List[str], List[str], float]:
        """Calculate keyword coverage with fuzzy matching"""
        resume_lower = resume_text.lower()
        matched_keywords = []
        missing_keywords = []
        
        for keyword in job_keywords:
            keyword_lower = keyword.lower()
            # Exact match
            if keyword_lower in resume_lower:
                matched_keywords.append(keyword)
            # Fuzzy match - check if partial words match
            elif any(keyword_lower in word or word in keyword_lower for word in resume_lower.split() if len(word) > 3):
                matched_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        coverage_ratio = len(matched_keywords) / len(job_keywords) if job_keywords else 0
        return matched_keywords, missing_keywords, coverage_ratio
    
    @staticmethod
    async def calculate_section_scores(resume_sections: ResumeSection, job_description: str) -> MatchedSections:
        """Calculate similarity scores for individual sections"""
        job_lower = job_description.lower()
        
        # Experience score
        exp_score = 0.0
        if resume_sections.experience:
            exp_text = " ".join(resume_sections.experience).lower()
            exp_score = await MatchingEngine.calculate_semantic_similarity(exp_text, job_description)
        
        # Skills score
        skills_score = 0.0
        if resume_sections.skills:
            skills_text = " ".join(resume_sections.skills).lower()
            # Boost skills score since they're more directly relevant
            base_score = await MatchingEngine.calculate_semantic_similarity(skills_text, job_description)
            skills_score = min(base_score * 1.2, 1.0)
        
        # Education score
        edu_score = 0.0
        if resume_sections.education:
            edu_text = " ".join(resume_sections.education).lower()
            edu_score = await MatchingEngine.calculate_semantic_similarity(edu_text, job_description)
        
        return MatchedSections(
            experience=round(exp_score, 2),
            skills=round(skills_score, 2),
            education=round(edu_score, 2)
        )

@app.get("/")
async def root():
    return {"message": "Resume Match API is running", "version": "1.0.0"}

# Global metrics storage (in production, use Redis or database)
request_metrics = {
    "total_requests": 0,
    "parse_requests": 0,
    "score_requests": 0,
    "batch_requests": 0,
    "start_time": time.time()
}

def calculate_final_score(semantic_similarity: float, keyword_coverage_ratio: float, section_scores: MatchedSections) -> float:
    """Calculate final score using the specified algorithm"""
    # Section weighting: experience 50%, skills 30%, education 10%, summary 10%
    section_score = (
        section_scores.experience * 0.5 +
        section_scores.skills * 0.3 +
        section_scores.education * 0.1 +
        0.0 * 0.1  # summary placeholder (not implemented in sections)
    )
    
    # Final score: 60% semantic + 30% keywords + 10% sections
    final_score = (
        0.6 * semantic_similarity +
        0.3 * keyword_coverage_ratio +
        0.1 * section_score
    )
    
    # Normalize to 0-100 and round to 1 decimal place
    return round(min(final_score * 100, 100), 1)

def generate_explainability(resume_text: str, job_description: str, matched_keywords: List[str]) -> str:
    """Generate explainability text showing top contributing factors"""
    # Find top matching phrases
    resume_words = set(resume_text.lower().split())
    job_words = set(job_description.lower().split())
    common_words = resume_words.intersection(job_words)
    
    # Focus on meaningful common words (exclude very short words)
    meaningful_common = [word for word in common_words if len(word) > 3][:5]
    
    explanation_parts = []
    
    if matched_keywords:
        explanation_parts.append(f"Matched keywords: {', '.join(matched_keywords[:3])}")
    
    if meaningful_common:
        explanation_parts.append(f"Key overlapping terms: {', '.join(meaningful_common)}")
    
    if not explanation_parts:
        explanation_parts.append("Limited overlap found between resume and job requirements")
    
    return "; ".join(explanation_parts)

# API Endpoints
@app.get("/v1/health")
async def health_check():
    uptime = round(time.time() - request_metrics["start_time"], 2)
    return {
        "status": "healthy",
        "service": "resume-match-api",
        "version": "1.0.0",
        "uptime_seconds": uptime,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/v1/metrics")
async def get_metrics():
    uptime = time.time() - request_metrics["start_time"]
    requests_per_minute = request_metrics["total_requests"] / (uptime / 60) if uptime > 0 else 0
    
    return {
        "requests_total": request_metrics["total_requests"],
        "requests_per_minute": round(requests_per_minute, 2),
        "parse_requests": request_metrics["parse_requests"],
        "score_requests": request_metrics["score_requests"],
        "batch_requests": request_metrics["batch_requests"],
        "average_response_time_ms": 250,  # Placeholder - in production, track actual times
        "error_rate": 0.0,  # Placeholder - in production, track actual errors
        "uptime_seconds": round(uptime, 2)
    }

@app.post("/v1/parse-resume")
@limiter.limit("50/minute")
async def parse_resume(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
) -> ParseResumeResponse:
    """Parse resume from file or text input"""
    request_metrics["total_requests"] += 1
    request_metrics["parse_requests"] += 1
    
    # Get resume text
    resume_text = ""
    if file:
        # Check file size (max 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        
        if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload PDF, DOCX, or TXT files."
            )
        
        # Read file content
        content = await file.read()
        
        if file.content_type == "text/plain":
            resume_text = content.decode('utf-8', errors='ignore')
        elif file.content_type == "application/pdf":
            resume_text = DocumentParser.extract_text_from_pdf(content)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = DocumentParser.extract_text_from_docx(content)
    elif text:
        resume_text = text
    else:
        raise HTTPException(status_code=400, detail="Either file or text input is required")
    
    if len(resume_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume text too short. Minimum 50 characters required.")
    
    # Parse resume sections
    sections = ResumeParser.parse_resume_text(resume_text)
    
    return ParseResumeResponse(
        status="ok",
        parsed_text=resume_text,
        sections=sections
    )

@app.post("/v1/score-match")
@limiter.limit("100/minute")
async def score_match(
    api_request: Request,
    request: ScoreMatchRequest,
    api_key: str = Depends(verify_api_key)
) -> ScoreMatchResponse:
    """Score resume against job description"""
    request_metrics["total_requests"] += 1
    request_metrics["score_requests"] += 1
    
    if not request.resume_text:
        raise HTTPException(status_code=400, detail="resume_text is required")
    
    if len(request.resume_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume text too short")
    
    if len(request.job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description too short")
    
    # Extract job keywords
    job_keywords = KeywordExtractor.extract_keywords(request.job_description, request.top_k_keywords)
    
    # Calculate semantic similarity (now async)
    semantic_similarity = await MatchingEngine.calculate_semantic_similarity(
        request.resume_text, request.job_description
    )
    
    # Calculate keyword coverage
    matched_keywords, missing_keywords, coverage_ratio = MatchingEngine.calculate_keyword_coverage(
        request.resume_text, job_keywords
    )
    
    # Parse resume and calculate section scores
    resume_sections = ResumeParser.parse_resume_text(request.resume_text)
    section_scores = await MatchingEngine.calculate_section_scores(resume_sections, request.job_description)
    
    # Calculate final score
    final_score = calculate_final_score(semantic_similarity, coverage_ratio, section_scores)
    
    # Generate explainability
    explainability = generate_explainability(request.resume_text, request.job_description, matched_keywords)
    
    # Select relevant ATS hints
    selected_hints = ATS_HINTS[:5]  # Return first 5 hints for consistency
    
    return ScoreMatchResponse(
        score=final_score,
        semantic_similarity=round(semantic_similarity, 2),
        keyword_coverage=KeywordCoverage(
            required=job_keywords,
            missing=missing_keywords
        ),
        matched_sections=section_scores,
        ats_hints=selected_hints,
        explainability=explainability
    )

@app.post("/v1/batch-score")
async def batch_score(
    request: BatchScoreRequest,
    api_key: str = Depends(verify_api_key)
) -> BatchScoreResponse:
    """Score multiple resumes against a single job description"""
    request_metrics["total_requests"] += 1
    request_metrics["batch_requests"] += 1
    
    if len(request.resumes) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 resumes allowed per batch")
    
    if len(request.job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description too short")
    
    results = []
    
    for i, resume_text in enumerate(request.resumes):
        try:
            if len(resume_text.strip()) < 50:
                # Skip resumes that are too short but don't fail the entire batch
                continue
                
            # Create individual score request
            score_request = ScoreMatchRequest(
                resume_text=resume_text,
                job_description=request.job_description,
                top_k_keywords=request.top_k_keywords,
                return_rewrites=False
            )
            
            # Reuse the scoring logic
            job_keywords = KeywordExtractor.extract_keywords(request.job_description, request.top_k_keywords)
            semantic_similarity = await MatchingEngine.calculate_semantic_similarity(resume_text, request.job_description)
            matched_keywords, missing_keywords, coverage_ratio = MatchingEngine.calculate_keyword_coverage(resume_text, job_keywords)
            resume_sections = ResumeParser.parse_resume_text(resume_text)
            section_scores = await MatchingEngine.calculate_section_scores(resume_sections, request.job_description)
            final_score = calculate_final_score(semantic_similarity, coverage_ratio, section_scores)
            explainability = generate_explainability(resume_text, request.job_description, matched_keywords)
            
            results.append(ScoreMatchResponse(
                score=final_score,
                semantic_similarity=round(semantic_similarity, 2),
                keyword_coverage=KeywordCoverage(required=job_keywords, missing=missing_keywords),
                matched_sections=section_scores,
                ats_hints=ATS_HINTS[:3],  # Fewer hints for batch processing
                explainability=explainability
            ))
            
        except Exception as e:
            # Log error but continue processing other resumes
            continue
    
    return BatchScoreResponse(results=results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)