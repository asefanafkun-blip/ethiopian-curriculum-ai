import streamlit as st
import requests

st.set_page_config(page_title="Ethiopian Curriculum AI", page_icon="📚", layout="wide")

st.title("📚 Ethiopian Curriculum AI Assistant")
st.caption("Grade 9–12 Textbook Vector Search & Citation Engine")

# Sidebar Filters
with st.sidebar:
    st.header("Search Filters")
    selected_grade = st.selectbox("Grade Level", [None, 9, 10, 11, 12], format_func=lambda x: "All Grades" if x is None else f"Grade {x}")
    selected_subject = st.selectbox("Subject", [None, "Physics", "Chemistry", "Biology", "Mathematics", "English"], format_func=lambda x: "All Subjects" if x is None else x)
    st.divider()
    st.info("Direct PDF links open the exact split textbook part scrolled to the specific page target `#page=N`.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📖 Verified Textbook Sources"):
                for src in msg["sources"]:
                    target_url = f"{src['file_url']}#page={src['page_number']}"
                    st.write(f"**Grade {src['grade']} {src['subject']} — Page {src['page_number']}**")
                    st.caption(f'"{src["snippet"]}"')
                    st.markdown(f"👉 [📄 Open PDF at Page {src['page_number']}]({target_url})")

# User Input
if prompt := st.chat_input("Ask any question from Grade 9-12 textbooks..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching textbook embeddings & generating answer..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/api/chat",
                    json={
                        "query": prompt,
                        "grade": selected_grade,
                        "subject": selected_subject,
                        "top_k": 4
                    },
                    timeout=60
                )
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📖 Verified Textbook Sources"):
                            for src in sources:
                                target_url = f"{src['file_url']}#page={src['page_number']}"
                                st.write(f"**Grade {src['grade']} {src['subject']} — Page {src['page_number']}**")
                                st.caption(f'"{src["snippet"]}"')
                                st.markdown(f"👉 [📄 Open PDF at Page {src['page_number']}]({target_url})")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                else:
                    st.error(f"Error from server: {res.text}")

            except Exception as e:
                st.error(f"Failed to connect to backend service: {e}")