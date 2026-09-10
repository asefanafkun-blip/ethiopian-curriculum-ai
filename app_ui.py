import os
import streamlit as st
import requests

# Set page configuration for mobile and web responsiveness
st.set_page_config(
    page_title="Ethiopian Curriculum AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished mobile UI and Android APK presentation
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Live Cloud Render Backend API Endpoint
BACKEND_URL = os.getenv("BACKEND_URL", "https://ethiopian-curriculum-ai.onrender.com/api/chat")

st.title("📚 Ethiopian Curriculum AI Assistant")
st.caption("Grade 9–12 Textbook Vector Search & Citation Engine")

# Sidebar Controls & Information
with st.sidebar:
    st.header("🔍 Search Filters")
    selected_grade = st.selectbox(
        "Grade Level",
        [None, 9, 10, 11, 12],
        format_func=lambda x: "All Grades (9–12)" if x is None else f"Grade {x}"
    )
    selected_subject = st.selectbox(
        "Subject",
        [None, "Physics", "Chemistry", "Biology", "Mathematics", "English"],
        format_func=lambda x: "All Subjects" if x is None else x
    )
    
    st.divider()
    
    # Sample Test Prompts for Students
    st.markdown("💡 **Quick Practice Questions**")
    if st.button("❓ What is Newton's Second Law?"):
        st.session_state.preset_query = "Explain Newton's Second Law of Motion with mathematical formulas."
    if st.button("❓ Define Photosynthesis process"):
        st.session_state.preset_query = "What is photosynthesis and why is it important in plant biology?"
    if st.button("❓ What is an Ionic Bond?"):
        st.session_state.preset_query = "Explain ionic bonding and chemical structure."

    st.divider()
    
    # Reset Chat Session
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.info("ℹ️ **Textbook Citations**: Verified sources link directly to PDF page targets (`#page=N`).")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle preset queries clicked from sidebar buttons
preset_prompt = st.session_state.pop("preset_query", None)

# Display Past Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 Verified Textbook Sources"):
                for src in msg["sources"]:
                    file_url = src.get("file_url", "")
                    page_num = src.get("page_number", 1)
                    target_url = src.get("pdf_link") or (f"{file_url}#page={page_num}" if file_url else "")
                    
                    st.write(f"**Grade {src.get('grade', 'N/A')} {src.get('subject', 'General')} — Page {page_num}**")
                    st.caption(f'"{src.get("snippet", "")}"')
                    if target_url:
                        st.markdown(f"👉 [📄 Open PDF at Page {page_num}]({target_url})")

# User Input capturing (either typed or from quick buttons)
user_input = st.chat_input("Ask any question from Grade 9-12 textbooks...")
prompt = preset_prompt if preset_prompt else user_input

if prompt:
    # Append user question to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Live FastAPI Backend on Render
    with st.chat_message("assistant"):
        with st.spinner("Searching textbook embeddings & generating answer..."):
            try:
                payload = {
                    "query": prompt,
                    "grade": selected_grade,
                    "subject": selected_subject,
                    "top_k": 4
                }
                
                # Increased timeout to 120s to account for Render free-tier cold boot
                res = requests.post(BACKEND_URL, json=payload, timeout=120)

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer generated.")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📖 Verified Textbook Sources"):
                            for src in sources:
                                file_url = src.get("file_url", "")
                                page_num = src.get("page_number", 1)
                                target_url = src.get("pdf_link") or (f"{file_url}#page={page_num}" if file_url else "")

                                st.write(f"**Grade {src.get('grade', 'N/A')} {src.get('subject', 'General')} — Page {page_num}**")
                                st.caption(f'"{src.get("snippet", "")}"')
                                if target_url:
                                    st.markdown(f"👉 [📄 Open PDF at Page {page_num}]({target_url})")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                else:
                    st.error(f"Error from server ({res.status_code}): {res.text}")

            except requests.exceptions.Timeout:
                st.warning("⏱️ The backend server is spinning up after inactivity. Please retry in 20 seconds.")
            except Exception as e:
                st.error(f"Failed to connect to backend service: {e}")