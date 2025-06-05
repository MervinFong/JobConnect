import os
import json
import subprocess
from typing import List, Dict
import spacy
from textblob import TextBlob
import sys
from pathlib import Path

# ✅ Load spaCy model only when needed
def get_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# ✅ Predefined synonym mapping
SYNONYM_MAP = {
    "software": ["developer", "engineer", "coder", "programmer"],
    "analyst": ["data analyst", "researcher", "insight specialist"],
    "technician": ["operator", "mechanic", "machinist", "repairman"],
    "marketing": ["advertiser", "promoter", "campaign manager", "brand manager"],
    "finance": ["accountant", "auditor", "banker", "financial analyst"],
    "sales": ["sales rep", "salesperson", "associate", "consultant"],
    "education": ["teacher", "tutor", "lecturer", "instructor"],
    "healthcare": ["nurse", "doctor", "medical officer", "therapist"],
    "admin": ["clerk", "secretary", "administrator", "office assistant"]
}

KNOWN_JOBS = list(SYNONYM_MAP.keys()) + [j for subs in SYNONYM_MAP.values() for j in subs]

# ✅ Scraper runner with cloud-safe logging
def run_scraper(keyword: str, location: str) -> List[Dict]:
    try:
        script_path = os.path.join(Path(__file__).parent, "jobstreet_scraper_cloud.py")
        python_path = sys.executable
        process = subprocess.run(
            [python_path, script_path, keyword, location],
            capture_output=True,
            text=True,
            check=False
        )

        if process.stderr:
            print(f"⚠️ STDERR from scraper: {process.stderr.strip()}")

        if not process.stdout.strip():
            print(f"⚠️ No STDOUT returned from scraper for keyword: {keyword}")
            return []

        print("🔍 RAW OUTPUT:", process.stdout.strip())

        jobs = json.loads(process.stdout.strip())
        return jobs

    except json.JSONDecodeError as e:
        print("❌ Failed to parse scraped job results:", e)
    except Exception as e:
        print("❌ Unknown error occurred while running scraper:", e)

    return []

# ✅ Main job fetch function with NLP fallback
def fetch_scraped_jobs(keyword: str = "Software Engineer", location: str = "Malaysia") -> List[Dict]:
    tried = set()
    corrected = str(TextBlob(keyword).correct())

    print(f"📝 Spellcheck: '{keyword}' → '{corrected}'")
    jobs = run_scraper(corrected, location)
    if jobs:
        return jobs
    tried.add(corrected.lower())

    nlp = get_spacy_model()
    doc = nlp(corrected)
    fallback_nouns = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}]

    for noun in fallback_nouns:
        if noun in tried:
            continue
        tried.add(noun)
        print(f"🔁 Retrying with fallback noun: {noun}")
        jobs = run_scraper(noun, location)
        if jobs:
            return jobs

    keyword_doc = nlp(corrected)
    best_match = None
    best_score = 0

    for job_term in KNOWN_JOBS:
        sim = keyword_doc.similarity(nlp(job_term))
        if sim > best_score:
            best_score = sim
            best_match = job_term

    if best_match and best_match not in tried and best_score > 0.75:
        print(f"🧠 Similarity fallback: '{corrected}' ≈ '{best_match}' (score={best_score:.2f})")
        jobs = run_scraper(best_match, location)
        if jobs:
            return jobs
        tried.add(best_match)

    for key, synonyms in SYNONYM_MAP.items():
        if key in corrected.lower():
            for alt in synonyms:
                if alt not in tried:
                    print(f"🔁 Trying synonym fallback: {alt}")
                    tried.add(alt)
                    jobs = run_scraper(alt, location)
                    if jobs:
                        return jobs

    return []
