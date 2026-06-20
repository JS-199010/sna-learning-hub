import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def inject_custom_css():
    """Injects high-end professional CSS for a premium glassmorphic UI look."""
    css = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* Overall Font Reset */
    html, body, .stMarkdown, p, li, label {
        font-family: 'Plus Jakarta Sans', 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Main Background Gradient styling */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(26, 32, 48, 1) 0%, rgba(14, 18, 28, 1) 90.2%);
        color: #E2E8F0;
    }
    
    /* Card Styles */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }
    
    /* Diagnostic Card for RAG Chunks */
    .rag-chunk-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .rag-score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 8px;
    }
    
    .rag-source-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        color: #94A3B8;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
    }
    
    /* Login & Setup Forms Custom Layout */
    .login-container {
        max-width: 450px;
        margin: 80px auto;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    
    /* Premium Header Container */
    .premium-header {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 28px;
    }
    
    /* Custom Scrollbar for Chat */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.3);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }
    
    /* Buttons Customization */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.25s ease !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    }
    
    /* Custom subheader with accent bar */
    .section-title {
        border-left: 4px solid #6366F1;
        padding-left: 12px;
        margin-top: 24px;
        margin-bottom: 16px;
        color: #F8FAFC;
    }
    
    /* Feedback Containers */
    .success-alert {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #34D399;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    .error-alert {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #F87171;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_header(title, subtitle, badge_text=None):
    """Renders a beautifully styled top header bar."""
    badge_html = f'<span style="background-color: rgba(99,102,241,0.2); color: #818CF8; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 20px; border: 1px solid rgba(99,102,241,0.3); margin-left: 12px;">{badge_text}</span>' if badge_text else ''
    
    header_html = f"""
    <div class="premium-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {title}{badge_html}
                </h1>
                <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 1rem; font-weight: 400;">{subtitle}</p>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def draw_card_start(title=None):
    """Draws a premium cards starter tag."""
    title_html = f'<h4 style="margin: 0 0 12px 0; color: #F8FAFC; font-size: 1.2rem;">{title}</h4>' if title else ''
    st.markdown(f'<div class="glass-card">{title_html}', unsafe_allow_html=True)

def draw_card_end():
    """Closes the card wrapper."""
    st.markdown('</div>', unsafe_allow_html=True)

def render_metric_card(label, value, delta=None, trend_up=True):
    """Displays a custom styled metric card."""
    delta_color = "#34D399" if trend_up else "#F87171"
    delta_symbol = "▲" if trend_up else "▼"
    delta_html = f'<span style="color: {delta_color}; font-size: 13px; font-weight: 600; margin-left: 8px;">{delta_symbol} {delta}</span>' if delta else ''
    
    html = f"""
    <div class="glass-card" style="padding: 18px; text-align: center; margin-bottom: 12px;">
        <div style="color: #94A3B8; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">{label}</div>
        <div style="font-size: 28px; font-weight: 700; color: #F8FAFC; display: inline-flex; align-items: baseline;">
            {value}
            {delta_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_rag_chunk(text, score, filename, index):
    """Renders a retrieved chunk card with source info and similarity score."""
    score_percentage = f"{score * 100:.1f}%"
    html = f"""
    <div class="rag-chunk-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
                <span class="rag-score-badge">相似度 {score_percentage}</span>
                <span class="rag-source-badge">📖 {filename} (碎片 #{index})</span>
            </div>
        </div>
        <div style="color: #E2E8F0; font-size: 13.5px; line-height: 1.6; white-space: pre-wrap; font-family: monospace; background: rgba(0,0,0,0.15); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.03);">
{text}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def plot_score_history(scores_df):
    """Creates a beautiful line plot of test scores over time using Plotly."""
    if scores_df.empty:
        return None
        
    scores_df = scores_df.copy()
    scores_df['date_parsed'] = pd.to_datetime(scores_df['date'])
    scores_df = scores_df.sort_values(by='date_parsed')
    
    fig = px.line(
        scores_df,
        x='date',
        y='score',
        hover_data=['test_name', 'username'],
        markers=True,
        labels={'score': '分數', 'date': '測試日期', 'test_name': '測試單元'},
        title="📈 個人測驗分數趨勢"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94A3B8',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 105]),
        title_font=dict(size=16, family="Outfit", color="#F8FAFC"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    fig.update_traces(
        line=dict(color='#6366F1', width=3),
        marker=dict(size=8, color='#818CF8', line=dict(color='#E2E8F0', width=1.5))
    )
    return fig

def plot_topic_performance(scores_df):
    """Creates a horizontal bar plot showing average performance per test topic."""
    if scores_df.empty:
        return None
        
    # Group and clean test names. Example: "book.pdf - TCI" -> "TCI" or similar
    df = scores_df.copy()
    df['topic'] = df['test_name'].apply(lambda x: x.split(" - ")[-1] if " - " in x else x)
    
    avg_scores = df.groupby('topic')['score'].mean().reset_index().sort_values(by='score')
    
    fig = px.bar(
        avg_scores,
        x='score',
        y='topic',
        orientation='h',
        labels={'score': '平均分數', 'topic': '主題單元'},
        title="🎯 各單元掌握度分析"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94A3B8',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 105]),
        yaxis=dict(showgrid=False),
        title_font=dict(size=16, family="Outfit", color="#F8FAFC"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    fig.update_traces(
        marker=dict(
            color='#8B5CF6',
            line=dict(color='rgba(139, 92, 246, 0.5)', width=1)
        )
    )
    return fig
