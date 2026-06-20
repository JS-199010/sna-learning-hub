import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import datetime
import database
import rag_engine
import ui_components

# Initialize directories & database
UPLOAD_DIR = "uploaded_materials"
os.makedirs(UPLOAD_DIR, exist_ok=True)
database.init_db()

# Configure the API Key securely in the backend
# This key is completely hidden from the browser client UI.
# It attempts to load from a local ignored secrets.json file, or fallback to environmental secrets (Streamlit Cloud Secrets)
def load_api_key():
    secrets_path = os.path.join(os.path.dirname(__file__), "secrets.json")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "GEMINI_API_KEY" in data and data["GEMINI_API_KEY"].strip():
                    return data["GEMINI_API_KEY"].strip()
        except Exception:
            pass
    return os.environ.get("GEMINI_API_KEY", "")

API_KEY = load_api_key()
api_key = API_KEY  # Defined for backwards compatibility in downstream code blocks
st.session_state['api_key'] = API_KEY

# Page layout config
st.set_page_config(
    page_title="SNA 臨床麻醉學習中樞 V3 (RAG)", 
    page_icon="🩺", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Apply CSS injection
ui_components.inject_custom_css()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'role' not in st.session_state:
    st.session_state['role'] = ""
if 'exam_state' not in st.session_state:
    st.session_state['exam_state'] = 'setup'
if 'exam_questions' not in st.session_state:
    st.session_state['exam_questions'] = []
if 'user_answers' not in st.session_state:
    st.session_state['user_answers'] = {}
if 'current_exam_name' not in st.session_state:
    st.session_state['current_exam_name'] = ""
if 'current_retrieved_chunks' not in st.session_state:
    st.session_state['current_retrieved_chunks'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'study_mode' not in st.session_state:
    st.session_state['study_mode'] = False
if 'study_question_index' not in st.session_state:
    st.session_state['study_question_index'] = 0
if 'study_submitted' not in st.session_state:
    st.session_state['study_submitted'] = False
if 'study_user_answer' not in st.session_state:
    st.session_state['study_user_answer'] = None
if 'study_score' not in st.session_state:
    st.session_state['study_score'] = 0
if 'flashcard_index' not in st.session_state:
    st.session_state['flashcard_index'] = 0
if 'flashcard_flipped' not in st.session_state:
    st.session_state['flashcard_flipped'] = False

# ==========================================
# 1. LOGIN & SIGN-UP SCREEN
# ==========================================
if not st.session_state['logged_in']:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #F8FAFC; line-height: 1.4;'>📚 輔大醫院SNA學習平台<br><span style='font-size: 0.6em; color: #94A3B8;'>by CHS</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 24px;'>安全的身分驗證與學習管理系統</p>", unsafe_allow_html=True)
    
    login_tab, signup_tab = st.tabs(["🔐 帳號登入", "📝 註冊新帳號"])
    
    with login_tab:
        login_user = st.text_input("學員/教師 帳號", key="login_user")
        login_pass = st.text_input("密碼", type="password", key="login_pass")
        if st.button("立即登入", key="login_btn"):
            if not login_user or not login_pass:
                st.error("請填寫帳號及密碼！")
            else:
                role = database.verify_user(login_user, login_pass)
                if role:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = login_user
                    st.session_state['role'] = role
                    st.success("登入成功！頁面導向中...")
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤，請重試！")
                    
    with signup_tab:
        new_user = st.text_input("欲註冊帳號", key="new_user")
        new_pass = st.text_input("設定密碼", type="password", key="new_pass")
        new_role = st.selectbox("註冊角色", ["student", "teacher"], key="new_role")
        if st.button("註冊帳號", key="signup_btn"):
            if not new_user or not new_pass:
                st.error("請完整填寫註冊資料！")
            else:
                success = database.register_user(new_user, new_pass, new_role)
                if success:
                    st.success("註冊成功！請直接切換至登入分頁。")
                else:
                    st.error("該帳號已被使用，請更換其他名稱。")
                    
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 0px;'>🩺 SNA 學習平台</h3>", unsafe_allow_html=True)
    st.caption("Clinical Anesthesia Hub v3.5")
    st.markdown("---")
    
    # Display user identity
    role_emoji = "🎓" if st.session_state['role'] == 'teacher' else "🧑‍⚕️"
    role_name = "臨床指導教師" if st.session_state['role'] == 'teacher' else "住院學員 (SNA)"
    st.markdown(f"👤 **學員帳號**：{st.session_state['username']}")
    st.markdown(f"🏷️ **權限身分**：{role_emoji} {role_name}")
    st.markdown("---")
    
    # Navigation menu
    menu = [
        "📚 AI 學習與精準測驗",
        "💬 SNA 臨床麻醉 AI 導師",
        "📂 知識教材庫管理",
        "📊 個人學習成果",
        "🔖 重點字卡 (Flashcards)"
    ]
    if st.session_state['role'] == 'teacher':
        menu.append("📊 全院學員統計")
        
    choice = st.radio("功能導航", menu)
    
    st.markdown("---")
    if st.button("📴 安全登出", use_container_width=True):
        # Clear states
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==========================================
# 3. PAGE LOGIC IMPLEMENTATION
# ==========================================

# --- PAGE 1: 📂 知識教材庫管理 ---
if choice == "📂 知識教材庫管理":
    ui_components.render_header(
        "📂 知識教材庫管理", 
        "上傳、解析及管理您的臨床麻醉教科書，打造您的向量知識大腦。"
    )
    
    # Grid layout: left for upload, right for list and delete
    col_upload, col_list = st.columns([1, 1.2])
    
    with col_upload:
        ui_components.draw_card_start("📤 上傳新教材 (自動切割向量化)")
        st.info("💡 系統支援 **PDF**, **PPTX**, **TXT**, **MD** 等格式。檔案上傳後會自動分割並生成嵌入向量(Embeddings)，約需 1-3 分鐘。")
        
        uploaded_file = st.file_uploader("選擇您的電子書或簡報", type=["pdf", "pptx", "txt", "md"])
        if uploaded_file:
            if not api_key:
                st.warning("⚠️ 請先在左側欄位輸入 Gemini API Key 以啟用向量儲存運算。")
            else:
                if st.button("⚡ 開始進行深度記憶 (向量化)", use_container_width=True):
                    try:
                        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                        
                        with st.spinner("⏳ 步驟 1/3: 正在讀取教材並萃取內容..."):
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            text = rag_engine.extract_text(save_path)
                            
                        if not text.strip():
                            st.error("萃取失敗：找不到有效的文字內容。")
                        else:
                            with st.spinner("⏳ 步驟 2/3: 正在將內容切割為語意區段..."):
                                chunks = rag_engine.get_chunks(text)
                                
                            with st.spinner(f"⏳ 步驟 3/3: 正在為 {len(chunks)} 個碎片生成向量並記憶中..."):
                                embeddings = rag_engine.get_embeddings_batch(chunks, api_key)
                                # Save database entries
                                database.save_file(uploaded_file.name, st.session_state['username'])
                                database.save_chunks(uploaded_file.name, chunks, embeddings)
                                
                            st.success(f"🎉 成功！《{uploaded_file.name}》已完全向量化，共生成 {len(chunks)} 個記憶碎片！")
                            st.balloons()
                    except Exception as e:
                        st.error(f"❌ 向量化記憶失敗：{e}")
        ui_components.draw_card_end()
        
    with col_list:
        ui_components.draw_card_start("📚 現有教材清單")
        files = database.get_files()
        if not files:
            st.write("目前尚無上傳之教材庫。請使用左側區塊上傳。")
        else:
            for f in files:
                filename, uploader, upload_date = f
                # Get chunk count
                conn = database.get_connection()
                chunk_cnt = conn.execute("SELECT count(*) FROM document_chunks WHERE filename=?", (filename,)).fetchone()[0]
                conn.close()
                
                # Render file block
                st.markdown(
                    f"""
                    <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <div style='font-weight: 600; color: #F8FAFC;'>📖 {filename}</div>
                            <div style='font-size: 11px; color: #94A3B8; margin-top: 4px;'>上傳者: {uploader} | 日期: {upload_date} | 向量碎片: {chunk_cnt} 段</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                # st.button does not render well inside HTML, so we place it right after
                if st.button(f"🗑️ 刪除 {filename}", key=f"del_{filename}", type="secondary"):
                    database.delete_file(filename)
                    # Remove local file too
                    local_path = os.path.join(UPLOAD_DIR, filename)
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    st.toast(f"已刪除教材: {filename}")
                    st.rerun()
        ui_components.draw_card_end()

# --- PAGE 2: 📚 AI 學習與精準測驗 (RAG 驅動) ---
elif choice == "📚 AI 學習與精準測驗":
    ui_components.render_header(
        "📚 AI 學習與精準測驗 (RAG)", 
        "根據您選擇的教材庫範圍，指定主題即可由 AI 設計專屬的練習或計分考試。"
    )
    
    # Check if there are files in library
    files_list = database.get_files()
    if not files_list:
        st.warning("⚠️ 知識庫中尚無教材。請先至「📂 知識教材庫管理」上傳您的第一本書籍！")
        st.stop()
        
    # Phase 1: Setup test parameters
    if st.session_state['exam_state'] == 'setup':
        st.markdown("<h3 class='section-title'>⚙️ 第一步：設定您的學習主題與模式</h3>", unsafe_allow_html=True)
        
        col_setup_l, col_setup_r = st.columns([1, 1])
        
        with col_setup_l:
            ui_components.draw_card_start("📝 測驗組態設定")
            
            # Select files
            options = ["All Files"] + [f[0] for f in files_list]
            selected_file = st.selectbox("1. 選擇教材檔案範圍：", options)
            
            # Select learning topic
            topic_query = st.text_input(
                "2. 輸入想測驗的主題（請輸入完整的臨床情境或具體主題敘述，以利精準檢索）：", 
                placeholder="例如：請出題考我『TCI 藥理機轉與其副作用』、或『困難插管的評估指引』。建議輸入完整句子...",
                help="💡 提示：RAG 語意檢索非常依賴完整語意，輸入完整問題或敘述（而非單一關鍵字）可以讓系統從原文書中找出更精準的對應段落！"
            )
            st.caption("ℹ️ **檢索提示**：建議輸入**完整的臨床情境或完整問句**（如：『我想練習全身麻醉誘導的生理變化與插管評估』），以達到最佳的出題效果。")
            
            # Mode toggle
            mode = st.radio("3. 選擇學習模式：", ["📖 互動學習模式 (有即時詳解與提問)", "⏱️ 模擬考試模式 (交卷後才計分與檢視)"])
            
            # Question count
            q_count = st.slider("4. 產生題數：", 3, 10, 5)
            
            if st.button("🚀 開始檢索並生成題目", use_container_width=True):
                if not api_key:
                    st.error("請在左側輸入 Gemini API Key！")
                elif not topic_query.strip():
                    st.error("請填寫您想學習的主題！")
                else:
                    with st.spinner("🔍 正在從教材庫進行語意檢索中..."):
                        try:
                            # 1. Retrieve related chunks
                            top_chunks = rag_engine.search_related_chunks(
                                query=topic_query,
                                filename=selected_file,
                                api_key=api_key,
                                top_k=5
                            )
                            
                            if not top_chunks:
                                st.error("在教材庫中找不到與您輸入主題相關的內容，請嘗試更換主題或上傳更相關的書籍。")
                            else:
                                st.session_state['current_retrieved_chunks'] = top_chunks
                                context_text = "\n\n---\n\n".join([c["text"] for c in top_chunks])
                                
                                # 2. Generate questions using context
                                with st.spinner("🤖 AI 正在精準出題中..."):
                                    questions = rag_engine.generate_questions(
                                        context_text=context_text,
                                        topic_query=topic_query,
                                        api_key=api_key,
                                        num_questions=q_count
                                    )
                                    
                                    st.session_state['exam_questions'] = questions
                                    st.session_state['study_mode'] = "互動學習模式" in mode
                                    
                                    # Reset runtime states
                                    st.session_state['exam_state'] = 'testing'
                                    st.session_state['user_answers'] = {}
                                    st.session_state['study_question_index'] = 0
                                    st.session_state['study_submitted'] = False
                                    st.session_state['study_user_answer'] = None
                                    st.session_state['study_score'] = 0
                                    st.session_state['current_exam_name'] = f"{selected_file} - {topic_query}"
                                    st.rerun()
                        except Exception as e:
                            st.error(f"出題失敗：{e}")
            ui_components.draw_card_end()
            
        with col_setup_r:
            ui_components.draw_card_start("ℹ️ SNA 學習提示")
            st.markdown(
                """
                **什麼是 RAG 驅動出題？**
                
                RAG (Retrieval-Augmented Generation) 是一種將語意搜尋與生成式 AI 結合的技術。
                
                1. 系統會把您的**主題關鍵字**進行向量運算。
                2. 在資料庫中尋找**相似度最高的前 5 段教材內容**。
                3. 將這些真實存在的教材文字作為「參考書」，提供給 Gemini 進行精準出題。
                
                > [!TIP]
                > 這能有效防止 AI 產生「幻覺」胡亂編造答案，確保每一題的考點都有文獻或教科書為依據！
                """
            )
            ui_components.draw_card_end()

    # Phase 2: Testing mode - Study Mode (Practice one by one)
    elif st.session_state['exam_state'] == 'testing' and st.session_state['study_mode']:
        st.markdown(f"### 📖 互動學習單元：{st.session_state['current_exam_name']}")
        
        q_idx = st.session_state['study_question_index']
        questions = st.session_state['exam_questions']
        total_q = len(questions)
        
        if q_idx >= total_q:
            st.session_state['exam_state'] = 'finished'
            st.rerun()
            
        q = questions[q_idx]
        
        # Display progress bar
        st.progress((q_idx) / total_q)
        st.write(f"**第 {q_idx + 1} 題 / 共 {total_q} 題**")
        
        ui_components.draw_card_start(f"Q: {q['question']}")
        
        # Select answer
        selected_option = st.radio("選擇您的答案：", q['options'], key=f"study_q_{q_idx}", index=None if not st.session_state['study_submitted'] else q['options'].index(st.session_state['study_user_answer']))
        
        st.write("---")
        
        # Form buttons
        col_btn1, col_btn2 = st.columns([1, 5])
        
        with col_btn1:
            if not st.session_state['study_submitted']:
                if st.button("確認提交", use_container_width=True):
                    if not selected_option:
                        st.warning("請先選擇一個選項！")
                    else:
                        st.session_state['study_submitted'] = True
                        st.session_state['study_user_answer'] = selected_option
                        if selected_option == q['answer']:
                            st.session_state['study_score'] += 1
                        st.rerun()
            else:
                if st.button("下一題 ➡️", use_container_width=True):
                    st.session_state['study_submitted'] = False
                    st.session_state['study_user_answer'] = None
                    st.session_state['study_question_index'] += 1
                    if st.session_state['study_question_index'] >= total_q:
                        # Save score directly
                        final_pct = int((st.session_state['study_score'] / total_q) * 100)
                        database.save_score(st.session_state['username'], st.session_state['current_exam_name'], final_pct)
                        st.session_state['exam_state'] = 'finished'
                    st.rerun()
                    
        # Feedback details shown upon submission
        if st.session_state['study_submitted']:
            user_ans = st.session_state['study_user_answer']
            correct_ans = q['answer']
            
            with col_btn2:
                if user_ans == correct_ans:
                    st.success(f"🎉 回答正確！答案就是：{correct_ans}")
                else:
                    st.error(f"❌ 回答錯誤！您的選擇：{user_ans} | 正確答案：{correct_ans}")
                    
            st.markdown(f"**💡 AI 導師詳解：**\n{q['explanation']}")
            
            # Bookmark Option
            if st.button("🔖 將此教材考點加入重點書籤", key=f"bookmark_btn_{q_idx}"):
                # Store the question + explanation context as bookmark
                bookmark_text = f"【問題】\n{q['question']}\n\n【正確答案】\n{q['answer']}\n\n【詳解】\n{q['explanation']}"
                database.add_bookmark(st.session_state['username'], st.session_state['current_exam_name'], bookmark_text)
                st.toast("已成功存入您的「教材重點書籤」！")
                
            # Follow-up interactive chat under the practice question
            st.markdown("💬 **針對此題有疑惑？向 AI 導師發問：**")
            follow_up_input = st.text_input("輸入您的追問問題...", key=f"follow_up_{q_idx}")
            if st.button("送出追問", key=f"send_follow_up_{q_idx}"):
                if not follow_up_input:
                    st.warning("請填寫問題內容。")
                else:
                    with st.spinner("AI 導師思考中..."):
                        try:
                            # Context for follow-up
                            snippet = f"問題: {q['question']}\n正確答案: {q['answer']}\n原詳解: {q['explanation']}"
                            # Build a single prompt
                            genai.configure(api_key=api_key.strip())
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            fu_prompt = f"""
                            你是一位臨床麻醉導師。正在為學員解答關於以下題目的疑惑。
                            
                            相關考點脈絡：
                            {snippet}
                            
                            學員針對這題的提問是：
                            「{follow_up_input}」
                            
                            請用繁體中文給出簡明、準確且具臨床實務指導意義的解答。
                            """
                            response = model.generate_content(fu_prompt)
                            st.info(f"💡 **AI 導師的補充回覆：**\n\n{response.text}")
                        except Exception as e:
                            st.error(f"追問失敗：{e}")
        ui_components.draw_card_end()
        
    # Phase 2: Testing mode - Timed formal quiz
    elif st.session_state['exam_state'] == 'testing' and not st.session_state['study_mode']:
        st.markdown(f"### ⏱️ 模擬計分測驗中：{st.session_state['current_exam_name']}")
        
        with st.form("formal_exam_form"):
            for i, q in enumerate(st.session_state['exam_questions']):
                ui_components.draw_card_start(f"Q{i+1}: {q['question']}")
                st.session_state['user_answers'][str(i)] = st.radio(
                    "請選擇最適當之選項：", 
                    q['options'], 
                    key=f"formal_q_{i}", 
                    index=None
                )
                ui_components.draw_card_end()
                st.write("")
                
            submitted = st.form_submit_button("📤 提交試卷，計算成績")
            if submitted:
                # Check that all answered
                unanswered = [i for i in range(len(st.session_state['exam_questions'])) if st.session_state['user_answers'].get(str(i)) is None]
                if unanswered:
                    st.warning("⚠️ 您還有題目尚未作答！請完成所有題目後再行提交。")
                else:
                    # Calculate score
                    score = 0
                    total = len(st.session_state['exam_questions'])
                    for i, q in enumerate(st.session_state['exam_questions']):
                        if st.session_state['user_answers'].get(str(i)) == q['answer']:
                            score += 1
                            
                    final_pct = int((score / total) * 100)
                    # Save to DB
                    database.save_score(st.session_state['username'], st.session_state['current_exam_name'], final_pct)
                    st.session_state['study_score'] = score
                    st.session_state['exam_state'] = 'finished'
                    st.rerun()

    # Phase 3: Results & Analysis
    elif st.session_state['exam_state'] == 'finished':
        st.subheader("🎯 測驗結果報告與回顧")
        
        questions = st.session_state['exam_questions']
        total = len(questions)
        
        if st.session_state['study_mode']:
            correct_cnt = st.session_state['study_score']
        else:
            correct_cnt = st.session_state['study_score']
            
        score_pct = int((correct_cnt / total) * 100)
        
        # Display large metrics
        col_score1, col_score2 = st.columns([1, 2])
        with col_score1:
            ui_components.render_metric_card(
                "測驗得分", 
                f"{score_pct} / 100", 
                delta=f"答對 {correct_cnt} / {total} 題", 
                trend_up=score_pct >= 60
            )
            if st.button("🔄 重新進行新測驗", use_container_width=True):
                st.session_state['exam_state'] = 'setup'
                st.rerun()
        
        with col_score2:
            ui_components.draw_card_start("📈 分數短評")
            if score_pct == 100:
                st.success("🎉 太棒了！您完全掌握了這個單元的教材精髓。")
            elif score_pct >= 80:
                st.info("👍 表現優異！對大部分概念已有扎實理解，可透過下方錯題進行複習。")
            elif score_pct >= 60:
                st.warning("🤝 合格！但仍有些許盲點，建議重新閱讀下方參考教材片段。")
            else:
                st.error("🔬 尚待加強。此主題在臨床麻醉非常關鍵，請務必重新熟讀教材。")
            ui_components.draw_card_end()
            
        # Display Detailed Explanations
        st.markdown("<h3 class='section-title'>🔍 逐題詳解回顧</h3>", unsafe_allow_html=True)
        for i, q in enumerate(questions):
            ui_components.draw_card_start(f"Q{i+1}: {q['question']}")
            
            # Show answers depending on mode
            if st.session_state['study_mode']:
                # Study mode doesn't record static st.session_state['user_answers'] since it was live
                st.markdown(f"**📖 正確答案：** `{q['answer']}`")
            else:
                user_ans = st.session_state['user_answers'].get(str(i))
                correct_ans = q['answer']
                if user_ans == correct_ans:
                    st.markdown(f"🟢 **您的答案：** `{user_ans}` (正確)")
                else:
                    st.markdown(f"🔴 **您的答案：** `{user_ans}` | **正確答案：** `{correct_ans}`")
                    
            st.markdown(f"💡 **AI 詳解：**\n{q['explanation']}")
            ui_components.draw_card_end()
            st.write("")
            
        # RAG Source Inspector (Visual proof of text snippets)
        with st.expander("🔍 檢視本次出題所使用的教材出處 (RAG 溯源稽核)"):
            st.markdown("以下為系統從您的資料庫中搜尋出相似度最高的前 5 段教材片段，AI 是依據此內容出題與解答的：")
            for idx, chunk in enumerate(st.session_state['current_retrieved_chunks']):
                ui_components.render_rag_chunk(
                    text=chunk["text"],
                    score=chunk["score"],
                    filename=chunk["filename"],
                    index=idx + 1
                )

# --- PAGE 3: 💬 SNA 臨床麻醉 AI 導師 ---
elif choice == "💬 SNA 臨床麻醉 AI 導師":
    ui_components.render_header(
        "💬 SNA 臨床麻醉 AI 導師", 
        "向 AI 導師提出您的臨床疑問，導師將即時在您的專屬教科書大腦中檢索答案，並附帶原文出處。"
    )
    
    if not api_key:
        st.warning("⚠️ 請先在左側欄位輸入 Gemini API Key 以啟用 AI 導師對話服務。")
        st.stop()
        
    # File selection context
    files_list = database.get_files()
    options = ["All Uploaded Textbooks"] + [f[0] for f in files_list]
    chat_context = st.selectbox("📚 設定解答參考教材範圍：", options)
    
    # Initialize message list
    if not st.session_state['chat_history']:
        st.session_state['chat_history'] = [
            {"role": "assistant", "content": "你好！我是你的 SNA 臨床麻醉專屬導師。你可以向我詢問任何關於麻醉藥理學、氣道管理、生理監測或臨床指引的問題，我將結合你上傳的教材庫為你精準解答。"}
        ]
        
    # Clear chat helper
    col_clear, _ = st.columns([1, 6])
    with col_clear:
        if st.button("🧹 清除對話紀錄", use_container_width=True):
            st.session_state['chat_history'] = [
                {"role": "assistant", "content": "你好！對話紀錄已清除。我們開始新的討論吧！"}
            ]
            st.rerun()
            
    # Render chat container
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "source_chunks" in msg and msg["source_chunks"]:
                with st.expander("📖 檢視本回答的參考教材來源"):
                    for idx, chunk in enumerate(msg["source_chunks"]):
                        ui_components.render_rag_chunk(
                            text=chunk["text"],
                            score=chunk["score"],
                            filename=chunk["filename"],
                            index=idx + 1
                        )
                        
    # Chat input
    if prompt := st.chat_input("請輸入您的麻醉學疑問..."):
        # Display user message
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Get AI Response
        with st.chat_message("assistant"):
            with st.spinner("🔍 檢索教材並彙整回答中..."):
                try:
                    # 1. RAG retrieval
                    top_chunks = rag_engine.search_related_chunks(
                        query=prompt,
                        filename=None if chat_context == "All Uploaded Textbooks" else chat_context,
                        api_key=api_key,
                        top_k=4
                    )
                    
                    # Construct context
                    context_text = ""
                    if top_chunks:
                        context_text = "\n\n---\n\n".join([c["text"] for c in top_chunks])
                        
                    # 2. Get LLM response
                    answer = rag_engine.ask_ai_tutor(
                        chat_history=st.session_state['chat_history'],
                        context_text=context_text,
                        query=prompt,
                        api_key=api_key
                    )
                    
                    st.write(answer)
                    
                    # Show references
                    if top_chunks:
                        with st.expander("📖 檢視本回答的參考教材來源"):
                            for idx, chunk in enumerate(top_chunks):
                                ui_components.render_rag_chunk(
                                    text=chunk["text"],
                                    score=chunk["score"],
                                    filename=chunk["filename"],
                                    index=idx + 1
                                )
                                
                    # Save to chat history session
                    st.session_state['chat_history'].append({
                        "role": "assistant", 
                        "content": answer,
                        "source_chunks": top_chunks
                    })
                except Exception as e:
                    st.error(f"抱歉，解答發生錯誤：{e}")

# --- PAGE 4: 📊 個人學習成果 ---
elif choice == "📊 個人學習成果":
    ui_components.render_header(
        "📊 個人學習成果與歷程", 
        "追蹤並視覺化您的測驗表現、成長趨勢以及需要特別加強的麻醉知識主題。"
    )
    
    # Retrieve user scores
    scores = database.get_scores(username=st.session_state['username'])
    
    if not scores:
        st.info("💡 您尚未進行任何測驗。趕緊到「📚 AI 學習與精準測驗」生成您的第一份練習題吧！")
    else:
        df = pd.DataFrame(scores, columns=["id", "username", "test_name", "score", "date"])
        
        # Calculate metric values
        total_exams = len(df)
        valid_scores = df["score"].dropna()
        avg_score = int(valid_scores.mean()) if not valid_scores.empty else 0
        max_score = df["score"].max() if not valid_scores.empty else 0
        valid_count = len(valid_scores)
        pass_rate = int((len(df[df["score"] >= 60]) / valid_count) * 100) if valid_count > 0 else 0
        
        # Display Metrics Grid
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            ui_components.render_metric_card("累積測驗次數", f"{total_exams} 次", delta="持續學習中", trend_up=True)
        with col_m2:
            ui_components.render_metric_card("歷史平均分數", f"{avg_score} 分", delta="+1.2%" if avg_score >= 60 else "-2.1%", trend_up=avg_score >= 60)
        with col_m3:
            ui_components.render_metric_card("單次最高分數", f"{max_score} 分", delta="完美掌握" if max_score == 100 else "繼續加油", trend_up=max_score >= 80)
        with col_m4:
            ui_components.render_metric_card("合格比率 (>=60)", f"{pass_rate} %", delta="合格表現", trend_up=pass_rate >= 80)
            
        st.write("---")
        
        # Visual Charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            fig_line = ui_components.plot_score_history(df)
            if fig_line:
                st.plotly_chart(fig_line, use_container_width=True)
                
        with col_c2:
            fig_bar = ui_components.plot_topic_performance(df)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
                
        st.write("---")
        
        # Historical Data Table
        st.subheader("📋 歷次測驗詳細紀錄")
        df_display = df[["date", "test_name", "score"]].copy()
        df_display.columns = ["測驗時間", "測驗單元/主題", "分數"]
        df_display["分數"] = df_display["分數"].apply(lambda x: "🚫 已由教師清除" if pd.isna(x) else f"{int(x)} 分")
        st.dataframe(df_display, use_container_width=True)

# --- PAGE 5: 🔖 重點字卡 (Flashcards) ---
elif choice == "🔖 重點字卡 (Flashcards)":
    ui_components.render_header(
        "🔖 重點字卡 (Flashcards)", 
        "利用間隔重複原理，翻轉字卡來複習您收藏的經典考題與詳解。"
    )
    
    bookmarks = database.get_bookmarks(st.session_state['username'])
    
    if not bookmarks:
        st.info("💡 目前尚無收藏的重點字卡。在「📚 AI 學習與精準測驗」使用互動模式答題時，即可隨時將關鍵考題收藏至此！")
    else:
        total_cards = len(bookmarks)
        
        # Ensure index is within bounds
        if st.session_state['flashcard_index'] >= total_cards:
            st.session_state['flashcard_index'] = 0
            st.session_state['flashcard_flipped'] = False
            
        current_idx = st.session_state['flashcard_index']
        bm_id, filename, content, date = bookmarks[current_idx]
        
        # Parse content
        # Format is usually:
        # 【問題】\n...\n\n【正確答案】\n...\n\n【詳解】\n...
        question_text = content
        answer_text = ""
        
        if "【問題】" in content and "【正確答案】" in content:
            parts = content.split("【正確答案】")
            question_text = parts[0].replace("【問題】", "").strip()
            answer_text = "【正確答案】" + parts[1]
        
        st.markdown(f"### 卡片 {current_idx + 1} / {total_cards}")
        st.caption(f"🔖 來源: {filename} | 收藏時間: {date}")
        
        # Flashcard UI
        ui_components.draw_card_start("❓ 問題")
        st.markdown(f"#### {question_text}")
        ui_components.draw_card_end()
        
        st.write("")
        
        if st.session_state['flashcard_flipped']:
            ui_components.draw_card_start("💡 答案與詳解")
            st.markdown(answer_text)
            ui_components.draw_card_end()
            st.write("")
            
        # Controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⬅️ 上一張", use_container_width=True):
                st.session_state['flashcard_index'] = (current_idx - 1) % total_cards
                st.session_state['flashcard_flipped'] = False
                st.rerun()
                
        with col2:
            if st.button("🔄 翻轉卡片", type="primary", use_container_width=True):
                st.session_state['flashcard_flipped'] = not st.session_state['flashcard_flipped']
                st.rerun()
                
        with col3:
            if st.button("下一張 ➡️", use_container_width=True):
                st.session_state['flashcard_index'] = (current_idx + 1) % total_cards
                st.session_state['flashcard_flipped'] = False
                st.rerun()
                
        with col4:
            if st.button("🗑️ 移除此卡片", use_container_width=True):
                database.delete_bookmark(bm_id)
                st.toast("已成功移除字卡。")
                st.session_state['flashcard_flipped'] = False
                st.rerun()

# --- PAGE 6: 📊 全院學員統計 (Teacher only) ---
elif choice == "📊 全院學員統計":
    ui_components.render_header(
        "📊 全院學員統計與帳號管理", 
        "供指導教師即時掌握全體學員的學習狀態、進行分數管理，並對系統帳號與權限進行設定。"
    )
    
    # Check permission
    if st.session_state['role'] != 'teacher':
        st.error("🔒 權限不足，此頁面僅供臨床指導教師瀏覽。")
        st.stop()
        
    tab1, tab2 = st.tabs(["📈 全院學員成績管理", "👥 系統帳號密碼管理"])
    
    with tab1:
        all_scores = database.get_scores()
        
        if not all_scores:
            st.info("目前全院尚無學員的測驗紀錄。")
        else:
            df_all = pd.DataFrame(all_scores, columns=["id", "username", "test_name", "score", "date"])
            
            # Metrics aggregations
            total_tests = len(df_all)
            valid_scores_all = df_all["score"].dropna()
            avg_score_all = int(valid_scores_all.mean()) if not valid_scores_all.empty else 0
            active_students = df_all["username"].nunique()
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                ui_components.render_metric_card("全院累積測驗數", f"{total_tests} 次", delta="全體學員活躍度", trend_up=True)
            with col_t2:
                ui_components.render_metric_card("全體平均分數", f"{avg_score_all} 分", delta="合格標準: 60分", trend_up=avg_score_all >= 60)
            with col_t3:
                ui_components.render_metric_card("已測驗學員數", f"{active_students} 人", delta="持續新增中", trend_up=True)
                
            st.write("---")
            
            # Group stats
            st.subheader("🧑‍⚕️ 個別學員表現排名")
            student_summary = df_all.groupby("username").agg(
                平均分數=("score", "mean"),
                最高分數=("score", "max"),
                測驗次數=("date", "count")
            ).reset_index().round(1).sort_values(by="平均分數", ascending=False)
            
            student_summary["平均分數"] = student_summary["平均分數"].apply(lambda x: "無分數" if pd.isna(x) else f"{x} 分")
            student_summary["最高分數"] = student_summary["最高分數"].apply(lambda x: "無分數" if pd.isna(x) else f"{x} 分")
            st.dataframe(student_summary, use_container_width=True)
            
            st.write("---")
            
            # Interactive Score Clearance list
            st.subheader("📋 歷次測驗明細與分數管理 (可刪除分數但保留紀錄)")
            st.caption("💡 提示：點擊「清除分數」會將該次測驗分數設為『已清除』，不列入平均計算，但仍會保留該名學生有在該日期進行此測驗的歷程紀錄。")
            
            for row in all_scores[:50]:  # Show latest 50 tests
                score_id, uname, tname, val, dt = row
                
                col_row1, col_row2, col_row3, col_row4, col_row5 = st.columns([1.5, 3, 1.2, 1.2, 1])
                with col_row1:
                    st.write(f"🧑‍⚕️ **{uname}**")
                with col_row2:
                    st.write(f"📖 {tname}")
                with col_row3:
                    st.write(f"📅 {dt}")
                with col_row4:
                    if val is None or pd.isna(val):
                        st.markdown("<span style='color: #F87171; font-weight: 600;'>🚫 分數已清除</span>", unsafe_allow_html=True)
                    else:
                        st.write(f"🎯 **{int(val)} 分**")
                with col_row5:
                    if val is not None and not pd.isna(val):
                        if st.button("🗑️ 清除分數", key=f"clear_s_{score_id}", type="secondary", use_container_width=True):
                            database.clear_score(score_id)
                            st.toast(f"已清除 {uname} 的測驗分數！")
                            st.rerun()
            
            st.write("---")
            
            # Download reports
            st.subheader("📥 匯出全院學員成績總表")
            csv_data = df_all[["username", "test_name", "score", "date"]].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 下載 CSV 成績報表",
                data=csv_data,
                file_name=f"SNA_All_Students_Scores_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with tab2:
        st.subheader("👥 系統帳號密碼與權限管理台")
        
        user_list = database.get_users()
        df_users = pd.DataFrame(user_list, columns=["帳號名稱", "角色權限"])
        
        col_u_l, col_u_r = st.columns([1.2, 1])
        
        with col_u_l:
            st.markdown("##### 🔑 現有註冊帳號清單")
            st.dataframe(df_users, use_container_width=True)
            
        with col_u_r:
            st.markdown("##### ⚙️ 帳號管理控制板")
            
            usernames = [u[0] for u in user_list]
            selected_username = st.selectbox("選擇要管理的帳號：", usernames)
            
            # Action 1: Reset Password
            st.markdown("---")
            st.markdown("**1. 重設帳號密碼**")
            new_pwd = st.text_input("輸入新密碼：", value="1234", type="password", key="reset_pwd_input")
            if st.button("🔧 立即重設密碼", use_container_width=True):
                if selected_username and new_pwd:
                    database.update_user_password(selected_username, new_pwd)
                    st.success(f"帳號 `{selected_username}` 的密碼已更新成功！")
                    st.toast(f"已重設密碼：{selected_username}")
                    
            # Action 2: Change Role
            st.markdown("---")
            st.markdown("**2. 變更帳號角色**")
            current_role = next((u[1] for u in user_list if u[0] == selected_username), "student")
            role_options = ["student", "teacher"]
            new_role_val = st.selectbox("設定新角色身分：", role_options, index=role_options.index(current_role))
            if st.button("🔄 變更身分權限", use_container_width=True):
                if selected_username:
                    database.update_user_role(selected_username, new_role_val)
                    st.success(f"帳號 `{selected_username}` 的角色已更新為 `{new_role_val}`！")
                    st.toast(f"已更變角色：{selected_username}")
                    st.rerun()
                    
            # Action 3: Delete Account
            st.markdown("---")
            st.markdown("**3. 移除帳號**")
            st.warning("⚠️ 注意：刪除帳號將永久移除該名學員的登入資格。")
            if st.button("❌ 刪除此帳號", use_container_width=True, type="primary"):
                if selected_username == st.session_state['username']:
                    st.error("您無法刪除目前登入中的帳號！")
                elif selected_username:
                    database.delete_user(selected_username)
                    st.success(f"帳號 `{selected_username}` 已成功從系統中刪除！")
                    st.toast(f"已刪除帳號：{selected_username}")
                    st.rerun()
