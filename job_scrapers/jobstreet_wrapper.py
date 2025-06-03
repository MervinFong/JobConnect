import os
import json
import subprocess
from typing import List, Dict
import spacy
from textblob import TextBlob

# Load spaCy model for fallback keyword matching
nlp = spacy.load("en_core_web_sm")

# Predefined synonym map
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

# Known job keywords for similarity matching
KNOWN_JOBS = list(SYNONYM_MAP.keys()) + [j for subs in SYNONYM_MAP.values() for j in subs]

def fetch_scraped_jobs(keyword: str = "Software Engineer", location: str = "Malaysia") -> List[Dict]:
    def run_scraper(keyword: str, location: str) -> List[Dict]:
        try:
            script_path = os.path.join(os.path.dirname(__file__), "jobstreet_scraper_cloud.py")
            python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", "Scripts", "python.exe")

            process = subprocess.run(
                [python_path, script_path, keyword, location],
                capture_output=True,
                text=True,
                check=True
            )

            print("🔍 RAW SCRAPER OUTPUT:")
            print(process.stdout)

            jobs = json.loads(process.stdout)
            return jobs

        except subprocess.CalledProcessError as e:
            print("❌ Scraper subprocess error:", e.stderr)
        except json.JSONDecodeError as e:
            print("❌ Failed to parse scraped job results:", e)
        except Exception as e:
            print("❌ Unknown error occurred while scraping:", e)

        return []

    tried = set()

    # 1. Spellcheck using TextBlob
    corrected = str(TextBlob(keyword).correct())
    print(f"📝 Spellcheck: '{keyword}' → '{corrected}'")
    jobs = run_scraper(corrected, location)
    if jobs:
        return jobs
    tried.add(corrected.lower())

    # 2. Fallback noun using spaCy
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

    # 3. Similarity match to known job types
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

    # 4. Synonym expansion
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