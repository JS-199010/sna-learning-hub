import sqlite3
import os
import hashlib
import datetime

DB_PATH = 'sna_platform.db'

def get_connection():
    return sqlite3.connect(DB_PATH)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Users table (with hashed password)
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    
    # Scores table
    c.execute('''CREATE TABLE IF NOT EXISTS scores 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, test_name TEXT, score INTEGER, date TEXT)''')
    
    # Files table
    c.execute('''CREATE TABLE IF NOT EXISTS files 
                 (filename TEXT PRIMARY KEY, uploader TEXT, upload_date TEXT)''')
    
    # Document chunks table for RAG
    c.execute('''CREATE TABLE IF NOT EXISTS document_chunks 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, chunk_text TEXT, embedding TEXT)''')
    
    # Bookmarks table for student revision
    c.execute('''CREATE TABLE IF NOT EXISTS bookmarks 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, filename TEXT, chunk_text TEXT, date TEXT)''')
    
    # Insert default users if not exists
    c.execute("SELECT * FROM users WHERE username = 'teacher'")
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ('teacher', hash_password('1234'), 'teacher'))
        
    c.execute("SELECT * FROM users WHERE username = 'student1'")
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ('student1', hash_password('1234'), 'student'))
        
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    # Check hashed password
    hashed = hash_password(password)
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, hashed))
    result = c.fetchone()
    if not result:
        # Fallback check for plaintext just in case of old database migration
        c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
        result = c.fetchone()
        if result:
            # Upgrade password to hashed
            c.execute("UPDATE users SET password=? WHERE username=?", (hashed, username))
            conn.commit()
    conn.close()
    return result[0] if result else None

def register_user(username, password, role):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, hash_password(password), role))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def save_score(username, test_name, score):
    conn = get_connection()
    c = conn.cursor()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO scores (username, test_name, score, date) VALUES (?, ?, ?, ?)",
              (username, test_name, score, date_str))
    conn.commit()
    conn.close()

def get_scores(username=None, limit=100):
    conn = get_connection()
    c = conn.cursor()
    if username:
        c.execute("SELECT id, username, test_name, score, date FROM scores WHERE username=? ORDER BY id DESC LIMIT ?", (username, limit))
    else:
        c.execute("SELECT id, username, test_name, score, date FROM scores ORDER BY id DESC LIMIT ?", (limit,))
    results = c.fetchall()
    conn.close()
    return results

def save_file(filename, uploader):
    conn = get_connection()
    c = conn.cursor()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", (filename, uploader, date_str))
    conn.commit()
    conn.close()

def get_files():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT filename, uploader, upload_date FROM files ORDER BY upload_date DESC")
    results = c.fetchall()
    conn.close()
    return results

def delete_file(filename):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM files WHERE filename=?", (filename,))
    c.execute("DELETE FROM document_chunks WHERE filename=?", (filename,))
    conn.commit()
    conn.close()

def save_chunks(filename, chunks, embeddings):
    import json
    conn = get_connection()
    c = conn.cursor()
    # Clean previous chunks of this file
    c.execute("DELETE FROM document_chunks WHERE filename=?", (filename,))
    
    for i, chunk in enumerate(chunks):
        emb_json = json.dumps(embeddings[i])
        c.execute("INSERT INTO document_chunks (filename, chunk_text, embedding) VALUES (?, ?, ?)",
                  (filename, chunk, emb_json))
    conn.commit()
    conn.close()

def get_chunks_for_file(filename):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT chunk_text, embedding FROM document_chunks WHERE filename=?", (filename,))
    results = c.fetchall()
    conn.close()
    return results

def get_all_chunks():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT filename, chunk_text, embedding FROM document_chunks")
    results = c.fetchall()
    conn.close()
    return results

def add_bookmark(username, filename, chunk_text):
    conn = get_connection()
    c = conn.cursor()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO bookmarks (username, filename, chunk_text, date) VALUES (?, ?, ?, ?)",
              (username, filename, chunk_text, date_str))
    conn.commit()
    conn.close()

def get_bookmarks(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, filename, chunk_text, date FROM bookmarks WHERE username=? ORDER BY id DESC", (username,))
    results = c.fetchall()
    conn.close()
    return results

def delete_bookmark(bookmark_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM bookmarks WHERE id=?", (bookmark_id,))
    conn.commit()
    conn.close()
