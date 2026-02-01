import streamlit as st

st.set_page_config(page_title="VNU Summarizer", layout="wide")

# Chèn JavaScript để thay đổi tiêu đề ngay lập tức
st.markdown(
    """
    <script>
        document.title = "VNU Summarizer";
    </script>
    """,
    unsafe_allow_html=True
)

import fitz  
from docx import Document 

# Cấu hình tiêu đề trang ngay từ đầu

@st.cache_resource
def get_summarizer():
    from summarization import MultiDocSummarizationAPI
    return MultiDocSummarizationAPI


# Ẩn footer "Made with Streamlit"
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.image("./Logo_UET.png", width=150)

def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Lỗi khi xử lý PDF: {e}")
    return pdf_text

def extract_text_from_docx(uploaded_file):
    try:
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Lỗi khi xử lý DOCX: {e}")
        return ""
    
def add_text_area():
    st.session_state.additional_texts.append("")

def remove_text_area(index):
    st.session_state.additional_texts.pop(index)
    
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False

if "additional_texts" not in st.session_state:
    st.session_state.additional_texts = []
    
st.markdown("<h1>Hệ thống tóm tắt đa văn bản tiếng Việt</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### ✍️ Nhập văn bản")
    input_method = st.radio("Phương thức nhập liệu:", ["Nhập văn bản", "Kéo thả tệp"], horizontal=True)
    texts = []

    if input_method == "Nhập văn bản":
        if st.button("➕ Thêm vùng nhập văn bản"):
            add_text_area()
        for i, text in enumerate(st.session_state.additional_texts):
            with st.expander(f"📌 Văn bản {i + 1}", expanded=True):
                col_expander = st.columns([13, 0.5])
                with col_expander[0]:
                    updated_text = st.text_area("", text, height=200, key=f"text_{i}")
                    st.session_state.additional_texts[i] = updated_text
                with col_expander[1]:
                    if st.button("🗑", key=f"delete_{i}", help="Xóa văn bản"):
                        remove_text_area(i)
                        # st.experimental_rerun()
            texts.append(st.session_state.additional_texts[i])

    else:
        uploaded_files = st.file_uploader(
            "📂 Kéo thả tệp văn bản:", 
            type=["txt", "pdf", "docx"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                all_texts = ""
                if uploaded_file.type == "text/plain":
                    all_texts = uploaded_file.getvalue().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    all_texts = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    all_texts = extract_text_from_docx(uploaded_file)

                texts.append(all_texts)

    # st.markdown("### 🎯 Nhập tóm tắt mẫu")
    # golden_ext = st.text_area("📑 Tóm tắt tóm lược", height=100)
    # golden_abs = st.text_area("📝 Tóm tắt trích rút", height=100)

    
with col2:
    st.markdown("### ⚙️ Tuỳ chọn tóm tắt")
    summary_method = st.selectbox("Chọn phương thức rút gọn:", ["Số câu", "Tỷ lệ"])

    if summary_method == "Tỷ lệ":
        compress_ratio = st.slider("🔽 Chọn tỷ lệ rút gọn:", 0, 50, 15, step=1, format="%d%%") / 100
    else:
        compress_ratio = st.number_input("🔢 Số câu đầu ra:", min_value=1, max_value=20, value=5, step=1)

    if st.button("🚀 Tóm tắt") and any(texts):
        with st.spinner("⏳ Đang tóm tắt..."):
            summarizer = get_summarizer()
            summary_results = summarizer(texts, compress_ratio)
        
        st.session_state.extractive_summary = summary_results.get("extractive_summ", "Không có kết quả")
        st.session_state.abstractive_summary = summary_results.get("abstractive_summ", "Không có kết quả")
        st.session_state.rouge_ext = summary_results.get("score_ext", ("None", "None", "None"))
        st.session_state.rouge_abs = summary_results.get("score_abs", ("None", "None", "None"))
        st.session_state.show_summary = True
        # st.experimental_rerun()

if st.session_state.get("show_summary", False):
    col_summary = st.columns(2)
    rouge_ext = st.session_state.rouge_ext if st.session_state.rouge_ext is not None else ("None", "None", "None")
    rouge_abs = st.session_state.rouge_abs if st.session_state.rouge_abs is not None else ("None", "None", "None")

    with col_summary[0]:
        st.markdown("### 📑 Tóm tắt tóm lược")
        # st.markdown(f"**🔹 ROUGE 1:** {rouge_ext[0]}")
        # st.markdown(f"**🔹 ROUGE 2:** {rouge_ext[1]}")
        # st.markdown(f"**🔹 ROUGE L:** {rouge_ext[2]}")
        st.text_area("📑 Tóm tắt trích lược:", st.session_state.extractive_summary, height=250)

    # with col_summary[1]:
    #     st.markdown("### 📝 Tóm tắt trích rút")
    #     st.markdown(f"**🔹 ROUGE 1:** {rouge_abs[0]}")
    #     st.markdown(f"**🔹 ROUGE 2:** {rouge_abs[1]}")
    #     st.markdown(f"**🔹 ROUGE L:** {rouge_abs[2]}")
    #     st.text_area("Văn bản tóm tắt trích rút:", st.session_state.abstractive_summary, height=250)
