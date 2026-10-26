# AI Showroom Ô Tô (Local AI Assistant)

Dự án trợ lý AI tư vấn xe ô tô cho Showroom, hỗ trợ:
- Tư vấn thông số kỹ thuật, giá bán và chương trình khuyến mãi (Toyota Vios, VinFast VF 7, Mazda CX-5).
- Đặt lịch hẹn lái thử xe (`dat_lich_lai_thu`).
- Tạo đơn đặt cọc xe và thanh toán tự động qua mã VietQR (`tao_don_dat_coc_xe`).
- Hủy lịch hẹn lái thử (`huy_lich_lai_thu`).
- Yêu cầu xét duyệt hoàn tiền cọc (`yeu_cau_hoan_tien_coc`).

Dự án hoạt động **100% Offline / Local** sử dụng **Ollama (Qwen 2.5:3B)** kết hợp **ChromaDB (RAG)** và **FastAPI**.

---

## 🛠️ Yêu Cầu Tiên Quyết (Prerequisites)

1. **Python 3.10+**: Đã cài đặt trên máy.
2. **Ollama**: Đã cài đặt và đang chạy ([Tải Ollama tại đây](https://ollama.com/download)).
3. **Các mô hình AI cục bộ**:
   Mở Terminal / Command Prompt và tải 2 mô hình sau:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án

### Bước 1: Clone dự án về máy
```bash
git clone <URL_REPO_CỦA_BẠN>
cd ai-showroom-local
```

### Bước 2: Tạo và kích hoạt môi trường ảo Python
- **Trên Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Nạp dữ liệu tài liệu xe vào Vector DB (ChromaDB)
Chạy script nạp tài liệu từ file `tai_lieu_xe.txt`:
```bash
python ingest_local.py
```
*Sau khi chạy xong, hệ thống sẽ báo: "AI đã học xong tài liệu xe của bạn."*

### Bước 5: Khởi động Backend Server (FastAPI)
```bash
uvicorn main_local:app --reload --port 8000
```
Backend API sẽ chạy tại: `http://127.0.0.1:8000`.

### Bước 6: Khởi chạy Giao diện (Frontend)
- Mở file `index.html` trực tiếp bằng trình duyệt (Chrome, Edge, Firefox...), hoặc:
- Sử dụng tiện ích mở rộng **Live Server** trên VS Code để mở file `index.html`.

---

## 📂 Cấu Trúc Thư Mục

```text
ai-showroom-local/
├── main_local.py          # Backend chính (FastAPI, LangChain Agent, Tools, State Machine)
├── ingest_local.py        # Script nhúng vector tài liệu xe vào ChromaDB
├── tai_lieu_xe.txt        # Tài liệu dữ liệu về các dòng xe của showroom
├── du_lieu_khach_hang.json# CSDL lưu trữ lịch hẹn, đơn cọc và trạng thái hoàn/hủy
├── index.html             # Giao diện chat trực quan tích hợp VietQR
├── requirements.txt       # Danh sách thư viện Python cần cài đặt
├── .gitignore             # Loại bỏ venv và file tạm khi push Git
└── README.md              # Hướng dẫn chi tiết
```
