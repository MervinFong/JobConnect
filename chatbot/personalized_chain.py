import torch
import spacy
import datetime
import os
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from firebase_admin import firestore
from transformers import T5Tokenizer, T5ForConditionalGeneration, DistilBertTokenizer, DistilBertForSequenceClassification
from job_scrapers.jobstreet_wrapper import fetch_scraped_jobs
from chatbot.vector_store import load_vector_store
from chatbot.resume_analyzer import analyze_resume
from fpdf import FPDF
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load NLP + Models ===
nlp = spacy.load("en_core_web_sm")
retriever = load_vector_store().as_retriever()
chat_model = ChatOpenAI(temperature=0)
conversational_chain = ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=retriever, return_source_documents=True)

# === Load T5 Model ===
T5_MODEL_PATH = "t5model_v4"
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH).to(torch_device)

# === Load DistilBERT Model ===
DISTILBERT_PATH = "distilbert_resume_classifier_v2"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH).to(torch_device)
label_map = {
    0: "Education", 1: "Engineering", 2: "Finance", 3: "Healthcare", 4: "Human Resources (HR)",
    5: "Information Technology (IT)", 6: "Marketing", 7: "Operations", 8: "Others", 9: "Sales"
}

# === Memory and Intents ===
session_memory = {}
intent_model = SentenceTransformer("all-MiniLM-L6-v2")
intent_examples = {
    "refine_resume": ["refine my resume", "rewrite my CV"],
    "analyze_resume": ["analyze my resume", "review my resume"],
    "mbti": ["what is my mbti"],
    "interview_tips": ["give me interview tips"],
    "help_request": ["what can you do"],
    "greeting": ["hello"]
}
intent_embeddings = {
    label: intent_model.encode(samples, convert_to_tensor=True)
    for label, samples in intent_examples.items()
}

def detect_intent(user_input):
    query_vec = intent_model.encode(user_input, convert_to_tensor=True)
    best_intent = "unknown"
    max_score = 0
    for label, examples in intent_embeddings.items():
        sim = util.cos_sim(query_vec, examples).max().item()
        if sim > max_score:
            max_score = sim
            best_intent = label
    return best_intent if max_score > 0.65 else "unknown"

def predict_category(text):
    tokens = distilbert_tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(torch_device)
    with torch.no_grad():
        outputs = distilbert_model(**tokens)
        pred_id = outputs.logits.argmax(dim=1).item()
    return label_map[pred_id]

def run_t5_refiner(text):
    input_text = "Refine this resume: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(torch_device)
    with torch.no_grad():
        outputs = t5_model.generate(input_ids, max_new_tokens=512)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_internal_jobs(db):
    jobs = []
    internal_docs = db.collection("job_listings").where("approved", "==", True).where("active", "==", True).stream()
    for doc in internal_docs:
        data = doc.to_dict()
        title = data.get("job_title")
        company = data.get("company_name")
        if title and company:
            jobs.append(f"- {title} at {company}")
    return jobs

def get_personalized_chain(db, email):
    if email not in session_memory:
        session_memory[email] = {}
    if "jobstreet_links" not in session_memory[email]:
        session_memory[email]["jobstreet_links"] = []
    if "chat_history" not in session_memory[email]:
        session_memory[email]["chat_history"] = []

    def run_chain(user_input: str):
        resume_text = ""
        resume_docs = db.collection("resume_uploads").where("email", "==", email).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
        for doc in resume_docs:
            resume_text = doc.to_dict().get("resume_text", "")
            break

        if not user_input.strip() and resume_text:
            category = predict_category(resume_text)
            refined = run_t5_refiner(resume_text)
            return {
                "output_text": f"📄 I’ve analyzed your uploaded resume.\n\nPredicted Category: **{category}**\n\nHere’s a suggested refined version:\n\n```\n{refined}\n```",
                "input_documents": [],
                "resume_pdf": None
            }

        intent = detect_intent(user_input)
        resume_category = predict_category(resume_text) if resume_text else None

        if intent == "mbti":
            mbti_doc = db.collection("mbti_results").document(email).get()
            if mbti_doc.exists:
                mbti_data = mbti_doc.to_dict()
                mbti_type = mbti_data.get("mbti")
                if mbti_type:
                    traits = "Strategic, independent, analytical"
                    careers = "Software Engineer, Data Scientist, Project Manager"
                    return {
                        "output_text": f"🧠 Your MBTI type is **{mbti_type}**.\n\nKey Traits: {traits}\nRecommended Roles: {careers}",
                        "input_documents": [],
                        "resume_pdf": None
                    }
            return {
                "output_text": "❌ No MBTI result found. Please take the MBTI test in the system first.",
                "input_documents": [],
                "resume_pdf": None
            }

        if intent == "refine_resume" and resume_text:
            refined = run_t5_refiner(resume_text)
            return {
                "output_text": f"✅ Your resume has been refined.\n\nHere’s the improved version:\n\n```\n{refined}\n```",
                "input_documents": [],
                "resume_pdf": None
            }

        if intent == "analyze_resume" and resume_text:
            analysis = analyze_resume(resume_text)
            return {
                "output_text": f"🔍 Resume Analysis Result:\n{analysis}",
                "input_documents": [],
                "resume_pdf": None
            }

        if any(kw in user_input.lower() for kw in ["job", "career", "vacancy", "opening", "apply"]):
            scraped_links = fetch_scraped_jobs(keyword=user_input, location="Malaysia")  
            internal_jobs = get_internal_jobs(db)

            response_text = """🗂️ Internal Job Postings:
{}

🔗 Additional JobStreet Listings:
{}""".format(
                "\n".join(internal_jobs) or "- No internal listings available.",
                "\n".join(f"- {link}" for link in scraped_links[:5]) or "- No JobStreet results found."
            )

            return {
                "output_text": response_text,
                "input_documents": [],
                "resume_pdf": None
            }

        history = session_memory[email]["chat_history"]
        response = conversational_chain.invoke({
            "question": user_input,
            "chat_history": history
        })
        session_memory[email]["chat_history"].append((user_input, response.get("answer", "")))
        response["output_text"] = response.get("answer", "")
        response.setdefault("resume_pdf", None)

        return response

    return run_chain
