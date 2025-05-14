# VNU Summarizer - Hệ thống tóm tắt đa văn bản tiếng Việt

![Logo UET](./Logo_UET.png)

## Giới thiệu

VNU Summarizer là một ứng dụng web được phát triển nhằm cung cấp giải pháp tóm tắt đa văn bản cho tiếng Việt. Hệ thống này được xây dựng trên nền tảng Streamlit, cung cấp giao diện trực quan và dễ sử dụng cho người dùng.

## Mục tiêu

- Tạo các bản tóm tắt chất lượng cao từ nhiều tài liệu đầu vào
- Hỗ trợ cả hai phương pháp tóm tắt: trích lược (extractive) và trích rút (abstractive)
- Cung cấp công cụ đánh giá chất lượng tóm tắt dựa trên các chỉ số ROUGE
- Tạo giao diện người dùng thân thiện, dễ sử dụng

## Chức năng chính

### 1. Nhập liệu đa dạng
- **Nhập văn bản trực tiếp**: Người dùng có thể thêm nhiều vùng nhập văn bản
- **Tải lên tệp**: Hỗ trợ nhiều định dạng tệp phổ biến (txt, pdf, docx)

### 2. Phương pháp tóm tắt
- **Tóm tắt trích lược (Extractive Summarization)**: Trích xuất các câu quan trọng từ văn bản gốc
- **Tóm tắt trích rút (Abstractive Summarization)**: Tạo ra bản tóm tắt mới với cách diễn đạt riêng

### 3. Tùy chỉnh tham số
- **Tỷ lệ rút gọn**: Người dùng có thể chọn tỷ lệ rút gọn từ 0-50%
- **Số câu đầu ra**: Người dùng có thể chỉ định số câu cần xuất ra trong bản tóm tắt

### 4. Đánh giá chất lượng
- **Chỉ số ROUGE**: Hệ thống cung cấp các chỉ số ROUGE-1, ROUGE-2, ROUGE-L để đánh giá chất lượng tóm tắt
- **Tóm tắt mẫu**: Người dùng có thể nhập tóm tắt mẫu để so sánh với kết quả tóm tắt của hệ thống

## Cách sử dụng

1. **Nhập văn bản**:
   - Chọn phương thức nhập liệu (nhập trực tiếp hoặc tải tệp lên)
   - Nếu nhập trực tiếp, sử dụng nút "Thêm vùng nhập văn bản" để thêm nhiều văn bản
   - Nếu tải tệp, kéo thả các tệp vào vùng quy định

2. **Nhập tóm tắt mẫu** (không bắt buộc):
   - Nhập bản tóm tắt mẫu cho phương pháp trích lược
   - Nhập bản tóm tắt mẫu cho phương pháp trích rút

3. **Cấu hình tóm tắt**:
   - Chọn phương thức rút gọn (tỷ lệ hoặc số câu)
   - Điều chỉnh tỷ lệ rút gọn hoặc số câu đầu ra theo nhu cầu

4. **Xem kết quả**:
   - Nhấn nút "Tóm tắt" để xem kết quả
   - Kết quả sẽ hiển thị cả hai phương pháp tóm tắt cùng các chỉ số đánh giá ROUGE

## Cấu trúc mã nguồn

Ứng dụng được xây dựng dựa trên các thành phần chính sau:
- `streamlit`: Framework để xây dựng giao diện web
- `api.summarization.MultiDocSummarizationAPI`: API chính để xử lý tóm tắt đa văn bản
- `fitz`: Thư viện xử lý tệp PDF
- `docx`: Thư viện xử lý tệp Word

## Yêu cầu hệ thống

- Python 3.11
- Streamlit
- PyMuPDF (fitz)
- python-docx
- Các thư viện phụ thuộc khác được liệt kê trong tệp requirements.txt

## Cài đặt và chạy

```bash
# Clone repository
git clone <repository-url>

# Di chuyển vào thư mục dự án
cd vnu-summarizer

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run app.py
