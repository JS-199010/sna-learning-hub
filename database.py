import os
import json
import hashlib
import datetime
from supabase import create_client, Client

# Initialize Supabase client
def get_supabase_client() -> Client:
    url = ""
    key = ""
    if os.path.exists("secrets.json"):
        with open("secrets.json", "r", encoding="utf-8") as f:
            secrets = json.load(f)
            url = secrets.get("SUPABASE_URL", "")
            key = secrets.get("SUPABASE_KEY", "")
    else:
        # Fallback for Streamlit Cloud
        import streamlit as st
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        except Exception:
            url = os.environ.get("SUPABASE_URL", "")
            key = os.environ.get("SUPABASE_KEY", "")
            
    if not url or not key:
        print("Warning: Supabase credentials not found. Some features may not work.")
        # Dummy initialization to prevent crashing on import, though it will fail on actual queries
        url = url or "https://placeholder.supabase.co"
        key = key or "placeholder"
        
    return create_client(url, key)

supabase = get_supabase_client()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    # Tables are created manually in Supabase SQL Editor.
    pass

def verify_user(username, password):
    hashed = hash_password(password)
    res = supabase.table("users").select("role").eq("username", username).eq("password", hashed).execute()
    if res.data:
        return res.data[0]["role"]
    
    # Fallback for plaintext
    res = supabase.table("users").select("role").eq("username", username).eq("password", password).execute()
    if res.data:
        supabase.table("users").update({"password": hashed}).eq("username", username).execute()
        return res.data[0]["role"]
        
    return None

def register_user(username, password, role):
    try:
        supabase.table("users").insert({
            "username": username,
            "password": hash_password(password),
            "role": role
        }).execute()
        return True
    except Exception as e:
        return False

def save_score(username, test_name, score):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    supabase.table("scores").insert({
        "username": username,
        "test_name": test_name,
        "score": score,
        "date": date_str
    }).execute()

def get_scores(username=None, limit=100):
    query = supabase.table("scores").select("*").order("id", desc=True).limit(limit)
    if username:
        query = query.eq("username", username)
    res = query.execute()
    return [(r['id'], r['username'], r['test_name'], r['score'], r['date']) for r in res.data]

def save_file(filename, uploader):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    supabase.table("files").upsert({
        "filename": filename,
        "uploader": uploader,
        "upload_date": date_str
    }).execute()

def get_files():
    res = supabase.table("files").select("*").order("upload_date", desc=True).execute()
    return [(r['filename'], r['uploader'], r['upload_date']) for r in res.data]

def delete_file(filename):
    supabase.table("files").delete().eq("filename", filename).execute()
    supabase.table("document_chunks").delete().eq("filename", filename).execute()

def save_chunks(filename, chunks, embeddings):
    batch_size = 100
    supabase.table("document_chunks").delete().eq("filename", filename).execute()
    
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "filename": filename,
            "chunk_text": chunk,
            "embedding": json.dumps(embeddings[i])
        })
        
    for i in range(0, len(records), batch_size):
        supabase.table("document_chunks").insert(records[i:i+batch_size]).execute()

def get_chunks_for_file(filename):
    res = supabase.table("document_chunks").select("chunk_text, embedding").eq("filename", filename).execute()
    return [(r['chunk_text'], r['embedding']) for r in res.data]

def get_all_chunks():
    res = supabase.table("document_chunks").select("filename, chunk_text, embedding").execute()
    return [(r['filename'], r['chunk_text'], r['embedding']) for r in res.data]

def add_bookmark(username, filename, chunk_text):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    supabase.table("bookmarks").insert({
        "username": username,
        "filename": filename,
        "chunk_text": chunk_text,
        "date": date_str
    }).execute()

def get_bookmarks(username):
    res = supabase.table("bookmarks").select("*").eq("username", username).order("id", desc=True).execute()
    return [(r['id'], r['filename'], r['chunk_text'], r['date']) for r in res.data]

def delete_bookmark(bookmark_id):
    supabase.table("bookmarks").delete().eq("id", bookmark_id).execute()

def clear_score(score_id):
    supabase.table("scores").update({"score": None}).eq("id", score_id).execute()

def get_users():
    res = supabase.table("users").select("username, role").order("username").execute()
    return [(r['username'], r['role']) for r in res.data]

def update_user_password(username, new_password):
    hashed = hash_password(new_password)
    supabase.table("users").update({"password": hashed}).eq("username", username).execute()

def update_user_role(username, new_role):
    supabase.table("users").update({"role": new_role}).eq("username", username).execute()

def delete_user(username):
    supabase.table("users").delete().eq("username", username).execute()

def load_ai_rules():
    try:
        res = supabase.storage.from_("materials").download("ai_rules.json")
        return json.loads(res.decode("utf-8"))
    except Exception:
        return []

def save_ai_rules(rules_list):
    json_bytes = json.dumps(rules_list, ensure_ascii=False).encode("utf-8")
    try:
        # Try to update first
        supabase.storage.from_("materials").update("ai_rules.json", json_bytes, file_options={"content-type": "application/json"})
    except Exception:
        try:
            # If update fails, try to upload
            supabase.storage.from_("materials").upload("ai_rules.json", json_bytes, file_options={"content-type": "application/json"})
        except Exception as e:
            print(f"Failed to save AI rules: {e}")
