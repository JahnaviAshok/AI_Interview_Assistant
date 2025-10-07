# core/db.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import os
import streamlit as st
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from bson import ObjectId
from gridfs import GridFS

def _now(): return datetime.now(timezone.utc)

@st.cache_resource(show_spinner=False)
def get_mongo() -> tuple[MongoClient, str]:
    uri = (getattr(st.secrets, "MONGODB_URI", None) or os.getenv("MONGODB_URI"))
    if not uri: raise RuntimeError("MONGODB_URI not set")
    db_name = (getattr(st.secrets, "MONGODB_DB", None) or os.getenv("MONGODB_DB") or "interview_genaie")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    return client, db_name

@dataclass
class MongoStore:
    client: MongoClient
    db_name: str
    def __post_init__(self):
        self.db = self.client[self.db_name]
        self.users: Collection   = self.db["users"]
        self.resumes: Collection = self.db["resumes"]
        self.sessions: Collection= self.db["sessions"]
        self.answers: Collection = self.db["answers"]
        self.feedback: Collection= self.db["feedback"]
        self.demo_qa: Collection = self.db["demo_qa"]
        self.fs = GridFS(self.db)
        # indexes
        self.users.create_index([("email", ASCENDING)], unique=False)
        self.users.create_index([("created_at", DESCENDING)])
        self.resumes.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.sessions.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.answers.create_index([("session_id", ASCENDING), ("q_index", ASCENDING)], unique=True)
        self.feedback.create_index([("session_id", ASCENDING)], unique=True)
        self.demo_qa.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])

    # --- users ---
    def upsert_user(self, *, email: Optional[str], name: Optional[str]) -> str:
        email = (email or "").strip().lower() or None
        name  = (name or "").strip() or None
        if email:
            found = self.users.find_one({"email": email}, {"_id":1})
            if found:
                self.users.update_one({"_id": found["_id"]}, {"$set": {"name": name, "updated_at": _now()}})
                return str(found["_id"])
        _id = self.users.insert_one({"email": email, "name": name, "created_at": _now(), "updated_at": _now()}).inserted_id
        return str(_id)

    # --- resumes ---
    def save_resume(self, *, user_id: str, file_name: str, file_hash: str, text: str, raw_bytes: Optional[bytes], meta: Dict[str, Any]) -> str:
        fs_id = None
        if raw_bytes:
            fs_id = self.fs.put(raw_bytes, filename=file_name, contentType="application/octet-stream")
        doc = {
            "user_id": ObjectId(user_id),
            "file_name": file_name,
            "file_hash": file_hash,
            "text": text,
            "fs_id": fs_id,          # GridFS id if stored
            "meta": meta or {},
            "length_chars": len(text or ""),
            "created_at": _now(),
        }
        return str(self.resumes.insert_one(doc).inserted_id)

    # --- sessions ---
    def start_session(self, *, user_id: str, round_name: str, questions: List[str], is_demo: bool, model: Optional[str], temperature: Optional[float], candidate_name: Optional[str]) -> str:
        doc = {
            "user_id": ObjectId(user_id),
            "round_name": round_name,
            "questions": questions,
            "num_questions": len(questions or []),
            "is_demo": bool(is_demo),
            "model": model,
            "temperature": temperature,
            "candidate_name": candidate_name,
            "status": "in_progress",
            "created_at": _now(),
        }
        return str(self.sessions.insert_one(doc).inserted_id)

    # --- answers ---
    def add_answer(self, *, session_id: str, q_index: int, question: str, answer_text: Optional[str], transcript_text: Optional[str], audio_bytes: Optional[bytes], audio_mime: str = "audio/wav") -> None:
        audio_fs_id = None
        if audio_bytes:
            audio_fs_id = self.fs.put(audio_bytes, filename=f"{session_id}_q{q_index}.wav", contentType=audio_mime)
        doc = {
            "session_id": ObjectId(session_id),
            "q_index": int(q_index),
            "question": question,
            "answer_text": (answer_text or transcript_text or "")[:20000],
            "transcript_text": transcript_text,
            "audio_fs_id": audio_fs_id,
            "created_at": _now(),
        }
        self.answers.update_one({"session_id": doc["session_id"], "q_index": doc["q_index"]}, {"$set": doc}, upsert=True)

    # --- feedback ---
    def save_feedback(self, *, session_id: str, feedback: Dict[str, Any]) -> None:
        self.feedback.update_one(
            {"session_id": ObjectId(session_id)},
            {"$set": {"session_id": ObjectId(session_id), "feedback": feedback, "created_at": _now()}},
            upsert=True,
        )
        self.sessions.update_one({"_id": ObjectId(session_id)}, {"$set": {"status": "completed", "completed_at": _now()}})

    # --- demo Q&A ---
    def save_demo_qa(self, *, user_id: str, round_name: str, qa_items: List[Dict[str,str]], model: Optional[str], temperature: Optional[float], candidate_name: Optional[str], resume_len: int) -> str:
        doc = {
            "user_id": ObjectId(user_id),
            "round_name": round_name,
            "items": qa_items,
            "is_demo": True,
            "model": model,
            "temperature": temperature,
            "candidate_name": candidate_name,
            "resume_len": resume_len,
            "created_at": _now(),
        }
        return str(self.demo_qa.insert_one(doc).inserted_id)

def get_store() -> MongoStore:
    client, db = get_mongo()
    return MongoStore(client=client, db_name=db)
