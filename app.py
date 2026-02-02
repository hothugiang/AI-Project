import streamlit as st
import base64

st.set_page_config(page_title="VNU Summarizer", layout="wide")
st.markdown("""
<style>
/* Xóa padding mặc định phía trên */
.block-container {
    padding-top: 0rem !important;
}
            
/* ===== RESPONSIVE ===== */
@media (max-width: 1000px) {
    /* Ẩn cột gạch phân cách */
    .vertical-line {
        display: none !important;
    }

    /* Giảm padding tổng thể cho mobile */
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Toàn bộ row của st.columns */
    div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 24px;
    }

    /* Mỗi column chiếm full width */
    div[data-testid="column"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* ===== MOBILE: FILE UPLOADER FULL WIDTH ===== */
    div[data-testid="stFileUploader"] {
        width: 100% !important;
    }

    /* Drop zone bên trong */
    div[data-testid="stFileUploader"] section {
        width: 100% !important;
    }
}
        
@media (max-width: 1400px) {
    .vertical-line {
        height: 60vh;
    }
}

@media (max-width: 768px) {
    button:has(span:contains("Tóm tắt")) {
        width: 100% !important;
    }
}
        
/* ===== RESPONSIVE MODE SWITCH ===== */
@media (max-width: 900px) {

    /* Ẩn cột trống */
    .mode-switch div[data-testid="column"]:nth-of-type(3) {
        display: none !important;
    }

    /* ÉP 2 cột đầu GIÃN RA */
    .mode-switch div[data-testid="column"]:nth-of-type(1),
    .mode-switch div[data-testid="column"]:nth-of-type(2) {
        flex: 1 1 0% !important;
        max-width: 50% !important;
    }
}         
}
            
</style>
""", unsafe_allow_html=True)

    
# Hàm nạp ảnh local và chuyển sang Base64
def load_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = load_image_base64("Logo_UET.png")   # ← ảnh local

# Hiển thị logo + tiêu đề
st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: flex-start;
        align-items: center;
        margin-top: 50px;
    ">
        <img src="data:image/png;base64,{logo_base64}" width="60" style="margin-right: 17px;">
        <h1 style="
            margin: 0;
            font-size: 32px;
            font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
            font-weight: 700;
            color: #003366;
        ">
            Hệ thống tóm tắt đa văn bản Tiếng Việt
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)
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

@st.cache_resource(show_spinner=False)
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
    st.session_state.additional_texts = [""]
    


col1, col_line, col2 = st.columns([1.05, 0.05, 0.9]) # 0.05 là độ rộng cột chứa vạch

with col1:

    # --- 1. QUẢN LÝ STATE ---
    if "mode" not in st.session_state:
        st.session_state.mode = "Nhập văn bản"

    # --- 2. ĐỊNH NGHĨA MÀU ---
    primary_color = "#1a4d8f"
    white_color = "#ffffff"

    if st.session_state.mode == "Nhập văn bản":
        btn1_bg, btn1_text = primary_color, white_color
        btn2_bg, btn2_text = white_color, primary_color
    else:
        btn1_bg, btn1_text = white_color, primary_color
        btn2_bg, btn2_text = primary_color, white_color

    # --- 3. CSS TÙY CHỈNH KÍCH THƯỚC & VỊ TRÍ ---
    st.markdown(f"""
    <style>
    /* LƯU Ý QUAN TRỌNG: 
       Code Python bên dưới sẽ chia cột theo tỉ lệ [2, 2, 6].
       CSS này sẽ ép độ rộng nút theo pixel để đều tăm tắp.
    */

    /* Chọn button trong khối chia cột này */
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] button {{
        width: 170px !important;    /* <--- CHỈNH ĐỘ RỘNG NÚT TẠI ĐÂY */
        height: 40px !important;    /* Chỉnh chiều cao */
        border: 2px solid {primary_color} !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0px !important;
        transition: all 0.2s !important;
    }}
    
    /* Hiệu ứng Hover */
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    /* Nút 1: Nhập văn bản */
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(1) button {{
        background-color: {btn1_bg} !important;
        color: {btn1_text} !important;
    }}

    /* Nút 2: Kéo thả tệp */
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(2) button {{
        background-color: {btn2_bg} !important;
        color: {btn2_text} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # --- 4. GIAO DIỆN (ĐÃ SỬA ĐỂ SÁT TRÁI) ---
    def rerun():
        try: st.rerun()
        except: st.experimental_rerun()

    # MẸO: Chia 3 cột. 
    # Cột 1 & 2 nhỏ (chiếm 2 phần) để chứa nút.
    # Cột 3 to (chiếm 6 phần) là khoảng trắng để đẩy 2 nút kia sang trái.
    st.markdown('<div class="mode-switch">', unsafe_allow_html=True)
    colA, colB = st.columns([1,1]) 

    with colA:
        if st.button("Nhập văn bản", key="btn_text", use_container_width=True): 
            st.session_state.mode = "Nhập văn bản"
            rerun()

    with colB:
        if st.button("Kéo thả tệp", key="btn_file", use_container_width=True):
            st.session_state.mode = "Kéo thả tệp"
            rerun()
    
    # col_space bỏ trống hoàn toàn

    st.write("---")
    input_warning = st.empty() # Warning
    texts = []

    # --- NHẬP VĂN BẢN (ĐOẠN ĐÃ SỬA) ---
    st.markdown("""
    <style>
    /* --- STYLE 1: Style mặc định cho các nút khác (giữ nguyên) --- */
    div.stButton > button {
        background-color: #324569 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #143d72 !important;
    }
    /* ============================================================
   3. STYLE Ô NHẬP VĂN BẢN (TEXT AREA)
   ============================================================ */
    /* A. Style cho cái KHUNG bao ngoài (nơi chứa viền thực sự) */
    div[data-testid="stExpander"] .stTextArea div[data-baseweb="base-input"] {
        border: 1px solid #1a4d8f !important; /* Viền xanh mặc định */
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* B. Style khi CLICK chuột vào (Focus) - Dùng :focus-within */
    div[data-testid="stExpander"] .stTextArea div[data-baseweb="base-input"]:focus-within {
        border-color: #1a4d8f !important;         /* Giữ nguyên màu viền xanh */
    
    }

    /* C. Style cho chữ bên trong */
    div[data-testid="stExpander"] .stTextArea textarea {
        color: #000000 !important;       /* Chữ đen */
        font-weight: 700 !important;     /* In đậm */
        caret-color: #000000 !important; /* Màu con trỏ nháy cũng đen luôn */
    }
    /* Selector này nhắm vào ô nhập liệu bên trong Expander */
    div[data-testid="stExpander"] textarea {
        border: 1px solid #1a4d8f !important; /* Viền xanh đậm */
        background-color: #ffffff !important; /* Nền trắng */
        color: #000000 !important;            /* Chữ đen tuyệt đối */
        font-weight: 700 !important;          /* In đậm */
        border-radius: 8px !important;        /* Bo góc mềm mại */
    }

    /* --- FIX LỖI VIỀN ĐỎ TẠI ĐÂY --- */
    /* Trạng thái khi click chuột vào (Focus) */
    div[data-testid="stExpander"] textarea:focus {
        border-color: #1a4d8f !important;          /* Giữ nguyên màu viền xanh */
        box-shadow: 0 0 0 2px #1a4d8f !important;  /* Đè bóng mờ đỏ bằng bóng xanh */
        outline: none !important;                  /* Xóa outline mặc định của trình duyệt */
    }
                

                
    /* --- STYLE 4: STYLE RIÊNG CHO NÚT XÓA TRONG EXPANDER --- */
    /* Selector này sẽ ghi đè style của nút to (width 140px) phía trên */
    div[data-testid="stExpander"] div[data-testid="stHorizontalBlock"] button {
        /* 1. Reset kích thước */
        width: 32px !important;     
        height: 32px !important;
        min-width: 0px !important; /* QUAN TRỌNG: Để nút co nhỏ lại được */
        
        /* 2. Reset màu sắc và viền */
        border: none !important;
        background-color: transparent !important;
        color: #ff4b4b !important;
        
        /* 3. Căn chỉnh icon */
        padding: 0px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        margin: 0 auto !important;
        # margin-top: 45px !important;
        margin-top: clamp(2px, 2vw, 45px) !important;
    }

    /* Hover cho nút xoá */
    div[data-testid="stExpander"] div[data-testid="stHorizontalBlock"] button:hover {
        background-color: rgba(255, 75, 75, 0.1) !important;
        color: #324569 !important;
        transform: none !important;
        border: 1px solid #ff4b4b !important;
    }
    div[data-testid="stExpander"] div[data-testid="stHorizontalBlock"] button:hover p {
    color: #ff0000 !important;
}
    
    /* Tinh chỉnh khoảng cách input text */
    div[data-testid="stExpander"] .stTextArea {
        margin-bottom = 0 px !important;
        
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.mode == "Nhập văn bản":
        if st.button("Thêm vùng nhập văn bản"):
            add_text_area()

        for i, text in enumerate(st.session_state.additional_texts):
            with st.expander(f"📌 Văn bản {i + 1}", expanded=True):
                col_expander = st.columns([13, 0.8])

                with col_expander[0]:
                    updated_text = st.text_area("", text, height=200, key=f"text_{i}")
                    st.session_state.additional_texts[i] = updated_text

                with col_expander[1]:
                    if i > 0:
                        if st.button("🗑", key=f"delete_{i}", help="Xóa văn bản"):
                            remove_text_area(i)
                            st.experimental_rerun()

            # texts.append(st.session_state.additional_texts[i])
            texts = st.session_state.additional_texts.copy()


    # --- KÉO THẢ TỆP ---
    else:
        uploaded_files = st.file_uploader(
            "📂 Kéo thả tệp văn bản:",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "text/plain":
                    all_texts = uploaded_file.getvalue().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    all_texts = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    all_texts = extract_text_from_docx(uploaded_file)

                texts.append(all_texts)

st.markdown("""
<style>
/* CSS cho vạch kẻ dọc */
.vertical-line {
    border-left: 2px solid #e0e0e0;  /* Màu của vạch kẻ */
    height: 80vh;                    /* Chiều cao (80% màn hình) */
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

with col_line:
    # Chèn thẻ div đã style thành vạch kẻ
    st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)

st.markdown("""
<style>
.summary-title {
        background-color: #324569;
        color: white;
        font-weight: 700;
        padding: 5px 20px;
        display: flex;
        align-items: center;
        white-space: nowrap;
        font-size: 18px;
        border: 2px solid #324569;
        border-top-left-radius: 15px;
        border-bottom-left-radius: 15px;
    }
            

</style>
""", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="summary-wrapper">', unsafe_allow_html=True)
    title, select = st.columns([1.5, 2])
    with title:
        st.markdown("""
            <div class="summary-title">
                ⚙️ Tuỳ chọn tóm tắt
            </div>
        """, unsafe_allow_html=True)
    with select:
        summary_method = st.selectbox("", ["Số câu", "Tỷ lệ"], label_visibility="collapsed")

    if summary_method == "Tỷ lệ":
        compress_ratio = st.slider("🔽 Chọn tỷ lệ rút gọn:", 0, 50, 15, step=1, format="%d%%") / 100
    else:
        compress_ratio = st.number_input("🔢 Số câu đầu ra:", min_value=1, max_value=20, value=5, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== BUTTON + SPINNER (CENTER, INLINE) =====
    left, center, right = st.columns([3, 2, 3])

    with center:
        valid_texts = [t for t in texts if t.strip()]
        
        if st.button("🚀 Tóm tắt"):
            if len(valid_texts) < 2:
                input_warning.error("❌ Cần thêm ít nhất 2 văn bản")
            else:
                input_warning.empty()  # xoá cảnh báo cũ
                with st.spinner("⏳ Tóm tắt..."):
                    summarizer = get_summarizer()
                    summary_results = summarizer(texts, compress_ratio)


                st.session_state.extractive_summary = summary_results.get(
                    "extractive_summ", "Không có kết quả"
                )
                st.session_state.abstractive_summary = summary_results.get(
                    "abstractive_summ", ""
                )
                st.session_state.rouge_ext = summary_results.get(
                    "score_ext", ("None", "None", "None")
                )
                st.session_state.rouge_abs = summary_results.get(
                    "score_abs", ("None", "None", "None")
                )
                st.session_state.show_summary = True

    # ===== KẾT QUẢ ĐẦU RA (CARD) =====
    if st.session_state.get("show_summary", False):
        st.markdown("""
            <h3>📄 Kết quả đầu ra</h3>
        """, unsafe_allow_html=True)

        st.text_area(
            "",
            st.session_state.extractive_summary,
            height=260
        )

        st.markdown("</div>", unsafe_allow_html=True)