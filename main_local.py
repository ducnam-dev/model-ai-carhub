import json
import random
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=0)
vectorstore = Chroma(
    collection_name="showroom_xe",
    persist_directory="./chroma_db_local", 
    embedding_function=embeddings
)

pending_ui_action = {}
DB_FILE = "du_lieu_khach_hang.json"

def save_record(record):
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(record)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

@tool
def tra_cuu_thong_tin_xe(tu_khoa: str = "", mau_xe: str = "") -> str:
    """Chỉ dùng khi khách hỏi CHI TIẾT thông số kỹ thuật, khuyến mãi, trả góp, so sánh xe hoặc tư vấn FAQ trong showroom. TUYỆT ĐỐI KHÔNG gọi khi khách chỉ hỏi danh sách xe hoặc chào hỏi."""
    query = (tu_khoa or mau_xe or "").strip().lower()
    
    # Nếu câu hỏi là về danh sách xe chung, chỉ trả lời đúng 3 xe kèm khoảng giá (không lấy thông số/khuyến mãi)
    if any(k in query for k in ["danh sách", "danh sach", "có xe gì", "co xe gi", "những xe nào", "các xe", "mẫu xe", "showroom"]) or query in ["", "xe", "o to", "ô tô"]:
        return (
            "Showroom hiện có đúng 3 dòng xe sau:\n"
            "- Toyota Vios 2026: 458 - 545 triệu VNĐ\n"
            "- VinFast VF 7: 760 - 999 triệu VNĐ\n"
            "- Mazda CX-5 2026: 749 - 829 triệu VNĐ"
        )
    
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return "Không tìm thấy thông tin trong tài liệu."
    return "\n\n".join([d.page_content for d in docs])

def extract_phone(text: str):
    clean = re.sub(r'[\s\.\-]', '', text)
    m = re.search(r'(0[35789]\d{8}|0\d{9})', clean)
    return m.group(0) if m else None

def extract_order_code(text: str):
    m = re.search(r'\b(LT\d{4}|COC\d{4})\b', text, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    m2 = re.search(r'\b([A-Za-z]{2,4}\d{4,6})\b', text)
    if m2:
        return m2.group(0).upper()
    return None

def extract_refund_reason(text: str, target_id: str = "") -> str:
    cleaned = text
    if target_id:
        cleaned = re.sub(re.escape(target_id), "", cleaned, flags=re.IGNORECASE)
    for phrase in [
        "yêu cầu hoàn tiền cọc xe", "yêu cầu hoàn tiền cọc", "yêu cầu hoàn cọc",
        "hoàn tiền cọc xe", "hoàn tiền cọc", "hoàn cọc xe", "hoàn tiền", "hoàn cọc",
        "rút cọc xe", "rút cọc", "hủy cọc xe", "hủy cọc", "hủy đơn cọc",
        "tôi muốn", "cho tôi", "em ơi", "showroom ơi", "giúp tôi", "mã đơn", "mã cọc", "sđt", "số điện thoại", "sdt"
    ]:
        cleaned = re.sub(phrase, "", cleaned, flags=re.IGNORECASE)
    m_reason = re.search(r'(?:lý do|ly do|vì|do|bởi vì)\s*[:\-]?\s*(.+)', cleaned, re.IGNORECASE)
    if m_reason:
        return m_reason.group(1).strip(" .!?,;")
    cleaned = re.sub(r'[\:\-\,]', ' ', cleaned).strip()
    words = cleaned.split()
    if len(words) >= 2:
        return cleaned.strip(" .!?,;")
    return ""

STOP_NAME_WORDS = {
    "tôi", "muốn", "đó", "luôn", "lấy", "mua", "bán", "cọc", "đặt", "thêm", "nữa", 
    "xe", "con", "chiếc", "cùng", "lúc", "được", "không", "nhé", "nha", "ạ", "dạ", 
    "cho", "bạn", "cần", "hỏi", "tư", "vấn", "giá", "tiền", "nào", "gì", 
    "sao", "thế", "này", "mai", "nay", "hôm", "sáng", "chiều", "tối", "giờ", "lịch", "hẹn",
    "vios", "vinfast", "mazda", "toyota", "vf7", "vf 7", "cx5", "cx-5", "một", "hai", "ba"
}

def is_valid_person_name(name: str) -> bool:
    if not name: return False
    words = [w.lower() for w in name.split()]
    if not (1 <= len(words) <= 4): return False
    if any(w in STOP_NAME_WORDS for w in words):
        return False
    if not re.match(r'^[A-ZÀ-Ỹa-zà-ỹ\s]+$', name):
        return False
    return True

def extract_cars(text: str) -> list:
    t = text.lower()
    cars = []
    if any(k in t for k in ['vf 7', 'vf7', 'vinfast']):
        cars.append('VinFast VF 7')
    if any(k in t for k in ['vios', 'toyota']):
        cars.append('Toyota Vios 2026')
    if any(k in t for k in ['cx-5', 'cx5', 'mazda']):
        cars.append('Mazda CX-5 2026')
    return cars

def extract_car(text: str):
    cars = extract_cars(text)
    return cars[0] if cars else None

def extract_quantity(text: str) -> int:
    t = text.lower()
    m = re.search(r'(\d+)\s*(?:xe|chiếc|con)', t)
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            pass
    if any(k in t for k in ["hai xe", "2 xe", "hai chiếc", "2 chiếc", "cả 2", "cả hai", "2 con"]):
        return 2
    if any(k in t for k in ["ba xe", "3 xe", "ba chiếc", "3 chiếc", "cả 3", "cả ba"]):
        return 3
    cars = extract_cars(text)
    if len(cars) >= 2:
        return len(cars)
    return 1

def extract_name(text: str):
    m = re.search(r'(?:(?:tên\s+(?:của\s+)?(?:tôi|em|anh|chị|mình)|(?:tôi|em|anh|chị|mình)?\s*tên)\s*(?:là)?|mình\s+là|tôi\s+là|họ\s*tên)\s*[:\-]?\s*([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|-|\.|\s+sđt|\s+sdt|\s+số|\s+ngày|\s+lúc|$)', text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if is_valid_person_name(name):
            return name.title()
    
    parts = re.split(r'[,;\n]', text)
    if len(parts) > 1:
        for part in parts:
            p_strip = part.strip()
            if not p_strip: continue
            if extract_phone(p_strip) or extract_cars(p_strip) or extract_time(p_strip): continue
            cleaned_p = re.sub(r'^(?:tôi|em|anh|chị|mình|tên|là|họ)\s+', '', p_strip, flags=re.IGNORECASE).strip()
            if is_valid_person_name(cleaned_p):
                return cleaned_p.title()

    p = extract_phone(text)
    if p:
        cleaned = text.replace(p, '')
        t = extract_time(cleaned)
        if t: cleaned = cleaned.replace(t, '')
        for c in extract_cars(cleaned): cleaned = cleaned.replace(c, '')
        cleaned = re.sub(r'(?:đặt cọc|đặt lịch|lái thử|thử xe|đặt xe|xe|vios|vf\s*7|cx-?5|toyota|vinfast|mazda|sdt|sđt|số điện thoại|ngày đặt|ngày hẹn|ngày|hôm|lúc|thời gian)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'[0-9:\/\-\.,;?!]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if is_valid_person_name(cleaned):
            return cleaned.title()
    return None

def extract_time(text: str):
    m_pref = re.search(r'(?:ngày\s*đặt|ngay\s*dat|ngày\s*hẹn|ngay\s*hen|thời\s*gian|thoi\s*gian|lịch\s*hẹn|lich\s*hen)\s*[:\-]?\s*([A-Za-z0-9À-ỹ\s\/\:\-]+?)(?:,|-|\.|$)', text, re.IGNORECASE)
    if m_pref and len(m_pref.group(1).strip()) >= 2:
        return m_pref.group(1).strip()
    patterns = [
        r'(?:sáng|trưa|chiều|tối)?\s*(?:ngày\s*mai|mai|hôm\s*mai|hôm\s*nay|hôm\s*sau)?\s*(?:lúc\s*)?\d{1,2}(?:h|g|:\d{2}|h\d{2})?\s*(?:sáng|trưa|chiều|tối)?',
        r'(?:sáng|trưa|chiều|tối|ngày\s*mai|mai|hôm\s*mai|hôm\s*nay)\s*(?:lúc\s*)?\d{1,2}(?:h|g|:\d{2})?',
        r'(?:ngày|hôm)?\s*\d{1,2}[/-]\d{1,2}(?:[/-]\d{4})?',
        r'(?:ngày|hôm)\s*\d{1,2}',
        r'(?:thứ\s*[2-7]|chủ\s*nhật|t[2-7]|cn)(?:\s+tuần\s+sau)?',
        r'(?:sáng|trưa|chiều|tối)\s*(?:mai|nay)|ngày\s*mai|hôm\s*mai|hôm\s*nay|hôm\s*sau'
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m and len(m.group(0).strip()) >= 3:
            return m.group(0).strip()
    return None

@tool
def dat_lich_lai_thu(ho_ten: str = "", so_dien_thoai: str = "", mau_xe: str = "", ngay_gio_hen: str = "") -> str:
    """Gọi khi khách muốn ĐẶT LỊCH LÁI THỬ XE."""
    missing = []
    if not mau_xe: missing.append("mẫu xe")
    if not ho_ten: missing.append("Họ tên")
    if not so_dien_thoai: missing.append("Số điện thoại")
    if not ngay_gio_hen: missing.append("Ngày giờ hẹn")
    
    if missing:
        return f"Dạ để showroom chuẩn bị xe chu đáo nhất, anh/chị vui lòng cho em xin: {', '.join(missing)} nhé ạ!"

    ma_lich = f"LT{random.randint(1000, 9999)}"
    save_record({
        "ma_don": ma_lich,
        "loai": "LAI_THU",
        "ho_ten": ho_ten,
        "so_dien_thoai": so_dien_thoai,
        "mau_xe": mau_xe,
        "ngay_gio_hen": ngay_gio_hen,
        "trang_thai": "DA_XAC_NHAN"
    })
    return f"Đã đặt lịch lái thử thành công cho quý khách {ho_ten}! Mã lịch hẹn: {ma_lich}, Xe: {mau_xe}, Thời gian: {ngay_gio_hen}. Showroom rất hân hạnh được đón tiếp anh/chị!"

@tool
def tao_don_dat_coc_xe(ho_ten: str = "", so_dien_thoai: str = "", mau_xe: str = "", so_tien_coc: int = 20000000, so_luong: int = 1) -> str:
    """Gọi khi khách muốn ĐẶT CỌC MUA XE (Tiêu chuẩn 20.000.000 VNĐ / xe)."""
    missing = []
    if not mau_xe: missing.append("mẫu xe muốn đặt cọc")
    if not ho_ten: missing.append("Họ tên người đặt cọc")
    if not so_dien_thoai: missing.append("Số điện thoại liên hệ")
    
    if missing:
        return f"Dạ để showroom lập phiếu cọc và sinh mã QR chuẩn xác, anh/chị vui lòng cung cấp: {', '.join(missing)} giúp em nhé ạ!"

    if so_luong <= 0:
        so_luong = 1
    if so_tien_coc <= 0 or (so_luong > 1 and so_tien_coc == 20000000):
        so_tien_coc = so_luong * 20000000

    ma_don = f"COC{random.randint(1000, 9999)}"
    ngan_hang_id = "MB"
    so_tai_khoan = "0799014365"
    ten_chu_tk = "SHOWROOM O TO"
    qr_url = (
        f"https://img.vietqr.io/image/{ngan_hang_id}-{so_tai_khoan}-compact2.png"
        f"?amount={so_tien_coc}&addInfo={ma_don}&accountName={ten_chu_tk}"
    )
    
    save_record({
        "ma_don": ma_don,
        "loai": "DAT_COC",
        "ho_ten": ho_ten,
        "so_dien_thoai": so_dien_thoai,
        "mau_xe": mau_xe,
        "so_luong": so_luong,
        "so_tien_coc": so_tien_coc,
        "trang_thai": "CHO_THANH_TOAN",
        "qr_url": qr_url
    })
    
    pending_ui_action["current"] = {
        "type": "SHOW_QR",
        "order_id": ma_don,
        "customer_name": ho_ten,
        "car_model": mau_xe,
        "amount": so_tien_coc,
        "qr_url": qr_url
    }
    
    if so_luong > 1:
        return (
            f"Đã tạo đơn đặt cọc thành công mã {ma_don} cho khách hàng {ho_ten} "
            f"(Gồm {so_luong} xe: {mau_xe}, Tổng số tiền cọc: {so_tien_coc:,} VNĐ - tiêu chuẩn 20.000.000 VNĐ/xe). "
            f"Anh/chị vui lòng quét mã QR thanh toán hiển thị ngay bên dưới để hoàn tất thủ tục nhé ạ!"
        )
    else:
        return (
            f"Đã tạo đơn đặt cọc thành công mã {ma_don} cho khách hàng {ho_ten} "
            f"(Xe: {mau_xe}, Số tiền cọc: {so_tien_coc:,} VNĐ). "
            f"Anh/chị vui lòng quét mã QR thanh toán hiển thị ngay bên dưới để hoàn tất thủ tục nhé ạ!"
        )

@tool
def huy_lich_lai_thu(ma_lich_hoac_sdt: str = "") -> str:
    """Gọi khi khách hàng muốn HỦY LỊCH HẸN LÁI THỬ XE. Nhận mã lịch hẹn (LTxxxx) hoặc số điện thoại."""
    query = (ma_lich_hoac_sdt or "").strip()
    if not query:
        return "Dạ để hủy lịch hẹn lái thử, anh/chị vui lòng cung cấp Mã lịch hẹn (ví dụ: LT1234) hoặc Số điện thoại đăng ký giúp em nhé ạ!"
    
    clean_query = re.sub(r'[\s\.\-]', '', query).lower()
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
        
    found_item = None
    for item in reversed(data):
        if item.get("loai") == "LAI_THU":
            item_ma = str(item.get("ma_don", "")).strip().lower()
            item_sdt = re.sub(r'[\s\.\-]', '', str(item.get("so_dien_thoai", ""))).lower()
            if (clean_query == item_ma) or (len(clean_query) >= 9 and clean_query in item_sdt) or (len(item_sdt) >= 9 and item_sdt in clean_query):
                found_item = item
                break
                
    if not found_item:
        return f"Dạ showroom không tìm thấy lịch hẹn lái thử nào khớp với thông tin '{query}'. Anh/chị vui lòng kiểm tra lại Mã lịch hẹn hoặc Số điện thoại giúp em nhé ạ!"
        
    if found_item.get("trang_thai") == "DA_HUY":
        return f"Lịch hẹn lái thử {found_item.get('ma_don')} của quý khách {found_item.get('ho_ten')} (xe {found_item.get('mau_xe')}) trước đó đã được hủy rồi ạ."

    found_item["trang_thai"] = "DA_HUY"
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        return f"Lỗi khi cập nhật hủy lịch: {str(e)}"
        
    return (
        f"Đã hủy thành công lịch hẹn lái thử {found_item.get('ma_don')} của quý khách {found_item.get('ho_ten')} "
        f"(Mẫu xe: {found_item.get('mau_xe')}, Thời gian hẹn: {found_item.get('ngay_gio_hen')}). "
        f"Showroom rất tiếc vì sự bất tiện này và hy vọng có cơ hội được đón tiếp anh/chị trong thời gian tới!"
    )

@tool
def yeu_cau_hoan_tien_coc(ma_don_hoac_sdt: str = "", ly_do_hoan: str = "") -> str:
    """Gọi khi khách hàng YÊU CẦU HOÀN TIỀN CỌC XE. Nhận mã đơn cọc (COCxxxx) hoặc số điện thoại và lý do hoàn tiền."""
    query = (ma_don_hoac_sdt or "").strip()
    if not query:
        return "Dạ để yêu cầu hoàn tiền cọc xe, anh/chị vui lòng cung cấp Mã đơn cọc (ví dụ: COC1234) hoặc Số điện thoại đặt cọc và lý do giúp em nhé ạ!"
    
    clean_query = re.sub(r'[\s\.\-]', '', query).lower()
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
        
    found_item = None
    for item in reversed(data):
        if item.get("loai") == "DAT_COC":
            item_ma = str(item.get("ma_don", "")).strip().lower()
            item_sdt = re.sub(r'[\s\.\-]', '', str(item.get("so_dien_thoai", ""))).lower()
            if (clean_query == item_ma) or (len(clean_query) >= 9 and clean_query in item_sdt) or (len(item_sdt) >= 9 and item_sdt in clean_query):
                found_item = item
                break
                
    if not found_item:
        return f"Dạ showroom không tìm thấy đơn đặt cọc nào khớp với thông tin '{query}'. Anh/chị vui lòng kiểm tra lại Mã đơn cọc hoặc Số điện thoại giúp em nhé ạ!"
        
    if found_item.get("trang_thai") == "CHO_XET_DUYET_HOAN_TIEN":
        return (
            f"Yêu cầu hoàn tiền cọc cho đơn {found_item.get('ma_don')} của quý khách {found_item.get('ho_ten')} hiện đang ở trạng thái 'CHO_XET_DUYET_HOAN_TIEN'. "
            f"Showroom cam kết hoàn 100% tiền cọc trong 24 giờ làm việc. Bộ phận kế toán sẽ liên hệ theo SĐT {found_item.get('so_dien_thoai')} để hoàn tất thủ tục ạ."
        )

    found_item["trang_thai"] = "CHO_XET_DUYET_HOAN_TIEN"
    if ly_do_hoan:
        found_item["ly_do_hoan"] = ly_do_hoan
        
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        return f"Lỗi khi cập nhật yêu cầu hoàn tiền cọc: {str(e)}"
        
    so_tien = found_item.get("so_tien_coc", 20000000)
    return (
        f"Dạ đã tiếp nhận yêu cầu hoàn tiền cọc cho đơn {found_item.get('ma_don')} của quý khách {found_item.get('ho_ten')} "
        f"(Mẫu xe: {found_item.get('mau_xe')}, Số tiền cọc: {so_tien:,} VNĐ"
        f"{', Lý do: ' + ly_do_hoan if ly_do_hoan else ''}). "
        f"Trạng thái đơn đã chuyển sang 'CHO_XET_DUYET_HOAN_TIEN'. "
        f"Showroom cam kết hoàn 100% tiền cọc trong 24 giờ làm việc. "
        f"Bộ phận chăm sóc khách hàng sẽ liên hệ với anh/chị qua số {found_item.get('so_dien_thoai')} để thực hiện thủ tục hoàn tiền nhé ạ!"
    )

tools = [tra_cuu_thong_tin_xe, dat_lich_lai_thu, tao_don_dat_coc_xe, huy_lich_lai_thu, yeu_cau_hoan_tien_coc]
tools_map = {t.name: t for t in tools}

llm = ChatOllama(model="qwen2.5:3b", temperature=0.1, num_gpu=0, num_thread=8, num_predict=300)
llm_with_tools = llm.bind_tools(tools)

chat_sessions = {}
chat_states = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    pending_ui_action["current"] = None
    
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = [
            SystemMessage(content=(
                "Bạn là trợ lý tư vấn Showroom Ô tô. Trả lời cực kỳ ngắn gọn, súc tích, đi thẳng vào ý chính, tuyệt đối không trả lời thừa.\n\n"
                "Showroom chỉ kinh doanh 3 dòng xe:\n"
                "1. Toyota Vios 2026: 458 - 545 triệu VNĐ\n"
                "2. VinFast VF 7: 760 - 999 triệu VNĐ\n"
                "3. Mazda CX-5 2026: 749 - 829 triệu VNĐ\n\n"
                "CÁC NGUYÊN TẮC BẮT BUỘC:\n"
                "1. Khi khách chào hỏi ('hi', 'hello', 'chào'): CHỈ chào lại trong 1 câu ngắn. Tuyệt đối KHÔNG liệt kê danh sách xe.\n"
                "2. Khi khách hỏi 'danh sách xe', 'có xe gì', 'showroom có những xe nào', 'giá các xe': CHỈ trả lời đúng duy nhất 3 dòng xe trên và khoảng giá (tuyệt đối KHÔNG ghi thông số kỹ thuật, KHÔNG ghi khuyến mãi hay chính sách trả góp).\n"
                "3. Khi khách hỏi thông số kỹ thuật xe nào: Gọi 'tra_cuu_thong_tin_xe', chỉ trả lời đúng thông số của chính xe đó dạng gạch đầu dòng ngắn gọn (VinFast VF 7 là xe điện: SUV 5 chỗ, công suất 349 mã lực, pin 431 km/lần sạc; Toyota Vios: máy xăng 1.5L, 5.7L/100km; Mazda CX-5: máy xăng 2.0L, 6 cấp, 10 loa Bose). Tuyệt đối KHÔNG nhầm lẫn thông số giữa các xe, KHÔNG ghi giá bán, KHÔNG ghi khuyến mãi hay đặt cọc.\n"
                "4. Khi khách hỏi khuyến mãi / ưu đãi xe nào: Gọi 'tra_cuu_thong_tin_xe', chỉ nêu đúng các khuyến mãi của xe đó theo tài liệu (VF 7: hỗ trợ vay 80% trong 8 năm, miễn phí sạc 2 năm; Vios: giảm 50% trước bạ, tặng bảo hiểm & dán phim; CX-5: tặng thảm sàn & camera hành trình). Tuyệt đối KHÔNG ghi thông số kỹ thuật, và kèm đúng câu: 'Anh/chị có muốn đặt cọc 20 triệu để giữ ưu đãi này không ạ?'\n"
                "5. Các câu hỏi khác (tư vấn tầm giá 500tr/800tr, mua trả góp, thủ tục, so sánh xe, sạc pin, bảo hành...): Bắt buộc gọi 'tra_cuu_thong_tin_xe' để tra cứu tài liệu showroom và trả lời ngắn gọn trong 1-2 câu đúng trọng tâm.\n"
                "6. Lái thử: Yêu cầu đủ Họ tên, SĐT, Mẫu xe, Ngày giờ hẹn. Chưa đủ thì hỏi thông tin còn thiếu. Chỉ gọi 'dat_lich_lai_thu' khi khách cung cấp đầy đủ.\n"
                "7. Đặt cọc: Yêu cầu đủ Họ tên, SĐT, Mẫu xe. Chưa đủ thì hỏi thông tin còn thiếu, không tự tạo đơn khi chưa có thông tin từ khách.\n"
                "8. Hủy lịch lái thử: Khi khách muốn hủy lịch lái thử, bắt buộc hỏi Mã lịch hẹn (LTxxxx) hoặc Số điện thoại trước khi gọi tool 'huy_lich_lai_thu'.\n"
                "9. Hoàn tiền cọc: Khi khách muốn hoàn tiền cọc xe, bắt buộc hỏi Mã đơn cọc (COCxxxx) hoặc Số điện thoại (và lý do nếu có) trước khi gọi tool 'yeu_cau_hoan_tien_coc'."
            ))
        ]
    
    msg_raw = req.message.strip()
    msg_lower = msg_raw.lower().strip("?.!,:;")
    
    # Fast path 1: Chào hỏi đơn giản (phản hồi tức thì <0.01s, tiết kiệm 100% GPU/CPU)
    greetings = ["hi", "hello", "chao", "chào", "chào em", "chào bạn", "chào showroom", "xin chào", "alo", "helo", "hi bot"]
    if msg_lower in greetings:
        reply = "Dạ em chào anh/chị! Em có thể tư vấn dòng xe nào (Toyota Vios, VinFast VF 7, Mazda CX-5), đặt lịch lái thử hay đặt cọc xe cho anh/chị ạ?"
        chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
        chat_sessions[req.session_id].append(AIMessage(content=reply))
        return {"reply": reply, "action": None}
    
    # Fast path 2: Hỏi danh sách xe showroom (trả lời chính xác 3 xe và khoảng giá, không thừa thông số/khuyến mãi)
    car_list_keywords = [
        "danh sách xe", "danh sach xe", "có xe gì", "co xe gi", "showroom có xe gì",
        "có những xe nào", "co nhung xe nao", "showroom có những xe nào", "danh sách xe trong showroom",
        "cac mau xe", "các mẫu xe", "danh sach cac xe", "danh sách các xe", "bảng giá các xe",
        "bang gia cac xe", "xe trong showroom", "nhung xe nao trong showroom", "danh sách các loại xe",
        "cho tôi biết danh sách xe", "cho tôi biết danh sách xe trong showroom"
    ]
    is_asking_list = any(k in msg_lower for k in car_list_keywords)
    is_asking_details = any(k in msg_lower for k in ["thông số", "thong so", "khuyến mãi", "khuyen mai", "ưu đãi", "uu dai", "cọc", "lái thử", "lai thu", "so sánh", "so sanh"])
    
    if is_asking_list and not is_asking_details:
        reply = (
            "Showroom của chúng tôi hiện có các mẫu xe sau:\n\n"
            "1. **Toyota Vios 2026**: 458 - 545 triệu VNĐ\n"
            "2. **VinFast VF 7**: 760 - 999 triệu VNĐ\n"
            "3. **Mazda CX-5 2026**: 749 - 829 triệu VNĐ\n\n"
            "Anh/chị quan tâm đến thông số chi tiết hay chương trình khuyến mãi của mẫu xe nào ạ?"
        )
        chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
        chat_sessions[req.session_id].append(AIMessage(content=reply))
        return {"reply": reply, "action": None}

    # Quản lý trạng thái hội thoại (State Machine) cho Đặt cọc và Đặt lịch lái thử
    if req.session_id not in chat_states:
        chat_states[req.session_id] = {
            "flow": None,
            "mau_xe": None,
            "ho_ten": None,
            "so_dien_thoai": None,
            "ngay_gio_hen": None,
            "saved_ho_ten": None,
            "saved_so_dien_thoai": None,
            "last_flow": None
        }
    state = chat_states[req.session_id]

    # Hủy thao tác nếu khách đang trong form nhập liệu và muốn dừng lại
    if state["flow"] in ["DAT_COC", "LAI_THU", "HUY_LICH", "HOAN_COC"] and any(k in msg_lower for k in [
        "thôi không", "thoi khong", "không cọc nữa", "khong coc nua", 
        "không đặt nữa", "khong dat nua", "dừng lại", "dung lai", "bỏ qua", "bo qua", "thôi em", "thôi bot"
    ]):
        state["flow"] = None
        reply = "Dạ em đã hủy thao tác này. Em có thể hỗ trợ tư vấn dòng xe nào khác cho anh/chị ạ?"
        chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
        chat_sessions[req.session_id].append(AIMessage(content=reply))
        return {"reply": reply, "action": None}

    # Bóc tách thực thể từ tin nhắn
    cars_found = extract_cars(msg_raw)
    c = cars_found[0] if cars_found else None
    if c: state["mau_xe"] = c
    n = extract_name(msg_raw)
    if n: state["ho_ten"] = n
    p = extract_phone(msg_raw)
    if p: state["so_dien_thoai"] = p
    t = extract_time(msg_raw)
    if t: state["ngay_gio_hen"] = t

    # Tự động kế thừa thông tin khách hàng từ đơn trước nếu khách không nhập lại
    if not state["ho_ten"] and state.get("saved_ho_ten"):
        state["ho_ten"] = state["saved_ho_ten"]
    if not state["so_dien_thoai"] and state.get("saved_so_dien_thoai"):
        state["so_dien_thoai"] = state["saved_so_dien_thoai"]

    # Xử lý chuyên biệt: HỦY LỊCH LÁI THỬ & YÊU CẦU HOÀN TIỀN CỌC
    is_huy_lich_intent = any(k in msg_lower for k in [
        "hủy lịch lái thử", "huy lich lai thu", "hủy lịch hẹn", "huy lich hen",
        "hủy lái thử", "huy lai thu", "hủy hẹn lái thử", "hủy hẹn", "huy hen",
        "hủy lịch", "huy lich", "hủy đơn lái thử", "huy don lai thu"
    ]) and not any(k in msg_lower for k in ["cọc", "coc", "tiền"])

    is_hoan_coc_intent = any(k in msg_lower for k in [
        "hoàn tiền cọc", "hoan tien coc", "hoàn tiền", "hoan tien", 
        "hoàn cọc", "hoan coc", "rút cọc", "rut coc", 
        "trả lại cọc", "tra lai coc", "hủy cọc", "huy coc", 
        "hủy đơn cọc", "huy don coc"
    ])

    order_code = extract_order_code(msg_raw)
    phone_in_msg = extract_phone(msg_raw)
    lookup_id = order_code or phone_in_msg

    if is_huy_lich_intent or state.get("flow") == "HUY_LICH":
        if not lookup_id:
            if re.match(r'^\d{4}$', msg_raw.strip()):
                lookup_id = f"LT{msg_raw.strip()}"
            elif re.match(r'^[A-Za-z0-9]{4,12}$', msg_raw.strip()):
                lookup_id = msg_raw.strip()

        if lookup_id:
            result = huy_lich_lai_thu.invoke({"ma_lich_hoac_sdt": lookup_id})
            state["flow"] = None
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=result))
            return {"reply": result, "action": None}
        else:
            state["flow"] = "HUY_LICH"
            reply = "Dạ để showroom hỗ trợ hủy lịch hẹn lái thử, anh/chị vui lòng cung cấp Mã lịch hẹn (ví dụ: LT1234) hoặc Số điện thoại đã đăng ký giúp em nhé ạ!"
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=reply))
            return {"reply": reply, "action": None}

    if is_hoan_coc_intent or state.get("flow") == "HOAN_COC":
        if not lookup_id:
            if re.match(r'^\d{4}$', msg_raw.strip()):
                lookup_id = f"COC{msg_raw.strip()}"
            elif re.match(r'^[A-Za-z0-9]{4,12}$', msg_raw.strip()):
                lookup_id = msg_raw.strip()

        reason = extract_refund_reason(msg_raw, lookup_id or "")
        if lookup_id:
            result = yeu_cau_hoan_tien_coc.invoke({
                "ma_don_hoac_sdt": lookup_id,
                "ly_do_hoan": reason
            })
            state["flow"] = None
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=result))
            return {"reply": result, "action": None}
        else:
            state["flow"] = "HOAN_COC"
            reply = "Dạ để yêu cầu hoàn tiền cọc xe, anh/chị vui lòng cung cấp Mã đơn cọc (ví dụ: COC1234) hoặc Số điện thoại đặt cọc (kèm lý do hoàn tiền nếu có) giúp em nhé ạ!"
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=reply))
            return {"reply": reply, "action": None}

    # Nhận diện ý định đặt thêm xe / đặt nhiều xe
    is_dat_them_intent = any(k in msg_lower for k in [
        "đặt thêm", "dat them", "thêm xe", "them xe", 
        "đặt tiếp", "dat tiep", "đặt 2 xe", "dat 2 xe", "đặt hai xe", 
        "mua thêm", "mua them", "thêm 1 xe", "thêm một xe", "xe nữa", "xe nua", 
        "đặt nữa", "dat nua", "cọc thêm", "coc them", "lái thử thêm", "đặt 2 chiếc", "cọc 2 xe"
    ]) or bool(re.search(r'đặt\s+(?:thêm|tiếp|\d+)?\s*(?:chiếc\s+|con\s+)?xe', msg_lower))

    # Không kích hoạt quy trình mới nếu khách đang hỏi về lịch hẹn/đơn hàng vừa tạo
    is_inquiry = any(k in msg_lower for k in [
        "vừa đặt", "đã đặt", "vừa cọc", "đã cọc", "chưa em", "chưa bot", 
        "xem lại", "kiểm tra", "của tôi", "gì vậy", "sao rồi", "được chưa", "xong chưa"
    ])

    if not is_inquiry:
        is_coc_intent = any(k in msg_lower for k in [
            "đặt cọc", "dat coc", "cọc xe", "coc xe", 
            "tiền cọc", "tien coc", "đặt coc", "mua cọc", "giữ cọc", "dat cọc"
        ])
        is_lai_thu_intent = any(k in msg_lower for k in [
            "lái thử", "lai thu", "thử xe", "thu xe", 
            "đặt lịch", "dat lich", "lịch hẹn", "lich hen", 
            "hẹn lái", "hen lai", "hẹn ngày", "hen ngay", "ngày hẹn", "ngay hen"
        ])

        # Nếu khách muốn "đặt thêm xe" nhưng chưa chọn xe: hỏi ngay và hướng dẫn
        if is_dat_them_intent and not cars_found:
            reply = "Dạ vâng! Em sẵn sàng hỗ trợ anh/chị đặt thêm xe. Showroom hiện có 3 dòng xe: Toyota Vios 2026, VinFast VF 7 và Mazda CX-5 2026. Anh/chị muốn đặt thêm dòng xe nào và muốn Đặt cọc giữ xe hay Đặt lịch lái thử ạ?"
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=reply))
            return {"reply": reply, "action": None}

        # Nếu có từ khóa đặt thêm xe kèm mẫu xe:
        if is_dat_them_intent and cars_found:
            if any(k in msg_lower for k in ["cọc", "coc", "mua"]):
                is_coc_intent = True
            elif any(k in msg_lower for k in ["lái", "lai", "hẹn", "lịch"]):
                is_lai_thu_intent = True
            else:
                if state.get("last_flow") == "LAI_THU":
                    is_lai_thu_intent = True
                else:
                    is_coc_intent = True

        # Nếu khách chỉ nói "đặt xe": nếu có kèm ngày/giờ/thời gian thì là Đặt lịch lái thử, ngược lại là Đặt cọc
        if not is_coc_intent and not is_lai_thu_intent and ("đặt xe" in msg_lower or "dat xe" in msg_lower):
            if any(k in msg_lower for k in ["ngày", "ngay", "hôm", "mai", "giờ", "hẹn", "lịch", "sáng", "chiều", "tối"]) or extract_time(msg_raw):
                is_lai_thu_intent = True
            else:
                is_coc_intent = True
    else:
        is_coc_intent = False
        is_lai_thu_intent = False

    # XỬ LÝ ĐẶT NHIỀU XE CÙNG LÚC (len(cars_found) >= 2 hoặc có số lượng >= 2)
    qty = extract_quantity(msg_raw)
    if len(cars_found) >= 2:
        qty = max(qty, len(cars_found))

    if (len(cars_found) >= 2 or qty >= 2) and (is_coc_intent or is_lai_thu_intent or state["flow"]):
        target_flow = "LAI_THU" if is_lai_thu_intent or state["flow"] == "LAI_THU" else "DAT_COC"
        
        if len(cars_found) >= 2:
            combined_cars = " và ".join(cars_found)
        elif cars_found:
            combined_cars = f"{qty} xe {cars_found[0]}"
        elif state["mau_xe"]:
            combined_cars = f"{qty} xe {state['mau_xe']}"
        else:
            combined_cars = f"{qty} xe"

        if target_flow == "DAT_COC":
            tong_tien = qty * 20000000
            if not state["ho_ten"] or not state["so_dien_thoai"] or not (cars_found or state["mau_xe"]):
                missing = []
                if not (cars_found or state["mau_xe"]): missing.append("Mẫu xe muốn đặt")
                if not state["ho_ten"]: missing.append("Họ và tên")
                if not state["so_dien_thoai"]: missing.append("Số điện thoại")
                reply = f"Dạ vâng! Để tạo đơn đặt cọc cho {combined_cars} (Tổng tiền cọc: {tong_tien:,} VNĐ - tiêu chuẩn 20.000.000 VNĐ/xe), anh/chị vui lòng cung cấp {', '.join(missing)} giúp em nhé ạ!"
                state["flow"] = "DAT_COC"
                state["so_luong"] = qty
                chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
                chat_sessions[req.session_id].append(AIMessage(content=reply))
                return {"reply": reply, "action": None}
            else:
                result = tao_don_dat_coc_xe.invoke({
                    "ho_ten": state["ho_ten"],
                    "so_dien_thoai": state["so_dien_thoai"],
                    "mau_xe": combined_cars,
                    "so_luong": qty,
                    "so_tien_coc": tong_tien
                })
                act = pending_ui_action.get("current")
                state["saved_ho_ten"] = state["ho_ten"]
                state["saved_so_dien_thoai"] = state["so_dien_thoai"]
                state["last_flow"] = "DAT_COC"
                state["flow"] = None
                state["mau_xe"] = None
                state["ho_ten"] = None
                state["so_dien_thoai"] = None
                state["ngay_gio_hen"] = None
                state["so_luong"] = 1
                
                chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
                chat_sessions[req.session_id].append(AIMessage(content=result))
                return {
                    "reply": result,
                    "action": act
                }
        else: # LAI_THU
            missing = []
            if not (cars_found or state["mau_xe"]): missing.append("Mẫu xe")
            if not state["ho_ten"]: missing.append("Họ và tên")
            if not state["so_dien_thoai"]: missing.append("Số điện thoại")
            if not state["ngay_gio_hen"]: missing.append("Ngày giờ hẹn")
            if missing:
                reply = f"Dạ để showroom chuẩn bị {combined_cars} đón tiếp chu đáo nhất, anh/chị vui lòng cho em xin: {', '.join(missing)} nhé ạ!"
                state["flow"] = "LAI_THU"
                chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
                chat_sessions[req.session_id].append(AIMessage(content=reply))
                return {"reply": reply, "action": None}
            else:
                result = dat_lich_lai_thu.invoke({
                    "ho_ten": state["ho_ten"],
                    "so_dien_thoai": state["so_dien_thoai"],
                    "mau_xe": combined_cars,
                    "ngay_gio_hen": state["ngay_gio_hen"]
                })
                state["saved_ho_ten"] = state["ho_ten"]
                state["saved_so_dien_thoai"] = state["so_dien_thoai"]
                state["last_flow"] = "LAI_THU"
                state["flow"] = None
                state["mau_xe"] = None
                state["ho_ten"] = None
                state["so_dien_thoai"] = None
                state["ngay_gio_hen"] = None
                chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
                chat_sessions[req.session_id].append(AIMessage(content=result))
                return {"reply": reply, "action": None}

    # 1. Xử lý chuyên biệt: Quy trình ĐẶT CỌC XE ĐƠN LẺ
    if is_coc_intent or (state["flow"] == "DAT_COC" and not is_lai_thu_intent):
        state["flow"] = "DAT_COC"
        qty = max(extract_quantity(msg_raw), state.get("so_luong", 1))
        tong_tien = qty * 20000000
        car_name = f"{qty} xe {state['mau_xe']}" if (qty > 1 and state["mau_xe"]) else state["mau_xe"]

        if state["mau_xe"] and state["ho_ten"] and state["so_dien_thoai"]:
            result = tao_don_dat_coc_xe.invoke({
                "ho_ten": state["ho_ten"],
                "so_dien_thoai": state["so_dien_thoai"],
                "mau_xe": car_name,
                "so_luong": qty,
                "so_tien_coc": tong_tien
            })
            state["saved_ho_ten"] = state["ho_ten"]
            state["saved_so_dien_thoai"] = state["so_dien_thoai"]
            state["last_flow"] = "DAT_COC"
            state["flow"] = None
            state["mau_xe"] = None
            state["ho_ten"] = None
            state["so_dien_thoai"] = None
            state["ngay_gio_hen"] = None
            state["so_luong"] = 1
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=result))
            return {"reply": result, "action": pending_ui_action.get("current")}
        elif not state["mau_xe"]:
            reply = "Dạ showroom có 3 dòng xe: Toyota Vios 2026, VinFast VF 7 và Mazda CX-5 2026. Anh/chị muốn đặt cọc mẫu xe nào ạ?"
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=reply))
            return {"reply": reply, "action": None}
        else:
            missing = []
            if not state["ho_ten"]: missing.append("Họ và tên")
            if not state["so_dien_thoai"]: missing.append("Số điện thoại")
            missing_str = ", ".join(missing)
            reply = f"Dạ vâng! Để tạo đơn đặt cọc mẫu xe {car_name} (tiêu chuẩn {tong_tien:,} VNĐ), anh/chị vui lòng cung cấp {missing_str} giúp em nhé ạ!"
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=reply))
            return {"reply": reply, "action": None}

    # 2. Xử lý chuyên biệt: Quy trình ĐẶT LỊCH LÁI THỬ
    if is_lai_thu_intent or (state["flow"] == "LAI_THU" and not is_coc_intent):
        state["flow"] = "LAI_THU"
        # Chỉ nhận msg_raw làm ngày giờ hẹn nếu đây là tin nhắn phản hồi riêng về thời gian (không phải lệnh lái thử ban đầu)
        if not state["ngay_gio_hen"] and not is_lai_thu_intent and not n and not p and not c and len(msg_raw.split()) <= 8:
            state["ngay_gio_hen"] = msg_raw.strip()

        if state["mau_xe"] and state["ho_ten"] and state["so_dien_thoai"] and state["ngay_gio_hen"]:
            result = dat_lich_lai_thu.invoke({
                "ho_ten": state["ho_ten"],
                "so_dien_thoai": state["so_dien_thoai"],
                "mau_xe": state["mau_xe"],
                "ngay_gio_hen": state["ngay_gio_hen"]
            })
            state["saved_ho_ten"] = state["ho_ten"]
            state["saved_so_dien_thoai"] = state["so_dien_thoai"]
            state["last_flow"] = "LAI_THU"
            state["flow"] = None
            state["mau_xe"] = None
            state["ho_ten"] = None
            state["so_dien_thoai"] = None
            state["ngay_gio_hen"] = None
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=result))
            return {"reply": result, "action": None}
        elif not state["mau_xe"]:
            reply = "Dạ anh/chị muốn đăng ký lái thử mẫu xe nào ạ (Toyota Vios, VinFast VF 7 hay Mazda CX-5 2026)?"
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=reply))
            return {"reply": reply, "action": None}
        else:
            missing = []
            if not state["ho_ten"]: missing.append("Họ và tên")
            if not state["so_dien_thoai"]: missing.append("Số điện thoại")
            if not state["ngay_gio_hen"]: missing.append("Ngày giờ hẹn")
            missing_str = ", ".join(missing)
            reply = f"Dạ để showroom chuẩn bị xe {state['mau_xe']} đón tiếp chu đáo nhất, anh/chị vui lòng cho em xin: {missing_str} nhé ạ!"
            chat_sessions[req.session_id].append(HumanMessage(content=msg_raw))
            chat_sessions[req.session_id].append(AIMessage(content=reply))
            return {"reply": reply, "action": None}

    # Giới hạn context history: giữ SystemMessage + 8 tin nhắn gần nhất để CPU luôn phản hồi nhanh
    system_msg = chat_sessions[req.session_id][0]
    recent_history = chat_sessions[req.session_id][1:][-8:]
    chat_sessions[req.session_id] = [system_msg] + recent_history
    
    history = chat_sessions[req.session_id]
    history.append(HumanMessage(content=msg_raw))
    
    try:
        ai_msg = llm_with_tools.invoke(history)
        history.append(ai_msg)
        
        loop_guard = 0
        while ai_msg.tool_calls and loop_guard < 5:
            loop_guard += 1
            for tool_call in ai_msg.tool_calls:
                selected_tool = tools_map[tool_call["name"]]
                tool_output = selected_tool.invoke(tool_call["args"])
                history.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
            ai_msg = llm_with_tools.invoke(history)
            history.append(ai_msg)

        # Chặn lỗi LLM Qwen hallucinate in cú pháp tool dạng raw text ra màn hình
        if ai_msg.content and any(k in ai_msg.content for k in [
            "dat_lich_lai_thu", "Dat_lich_lai_thu", 
            "tao_don_dat_coc_xe", "Tao_don_dat_coc_xe",
            "huy_lich_lai_thu", "Huy_lich_lai_thu",
            "yeu_cau_hoan_tien_coc", "Yeu_cau_hoan_tien_coc"
        ]):
            if any(k in ai_msg.content for k in ["huy_lich_lai_thu", "Huy_lich_lai_thu"]):
                ai_msg.content = "Dạ để hủy lịch hẹn lái thử, anh/chị vui lòng cung cấp Mã lịch hẹn (LTxxxx) hoặc Số điện thoại đã đăng ký giúp em nhé ạ!"
            elif any(k in ai_msg.content for k in ["yeu_cau_hoan_tien_coc", "Yeu_cau_hoan_tien_coc"]):
                ai_msg.content = "Dạ để yêu cầu hoàn tiền cọc xe, anh/chị vui lòng cung cấp Mã đơn cọc (COCxxxx) hoặc Số điện thoại đặt cọc giúp em nhé ạ!"
            elif any(k in msg_lower for k in ["cọc", "coc", "mua"]):
                ai_msg.content = "Dạ vâng! Để tạo đơn đặt cọc, anh/chị vui lòng cho em biết muốn đặt cọc mẫu xe nào (Toyota Vios 2026, VinFast VF 7 hay Mazda CX-5 2026) nhé ạ!"
            else:
                ai_msg.content = "Dạ vâng! Showroom hiện có 3 dòng xe: Toyota Vios 2026, VinFast VF 7 và Mazda CX-5 2026. Anh/chị muốn đặt mẫu xe nào và muốn Đặt cọc giữ xe hay Đặt lịch lái thử ạ?"
            
        return {
            "reply": ai_msg.content,
            "action": pending_ui_action.get("current")
        }
    except Exception as e:
        print(f"Lỗi khi xử lý chat: {e}")
        return {
            "reply": f"Hệ thống gặp sự cố khi xử lý yêu cầu: {str(e)}",
            "action": None
        }

# API xác nhận thanh toán QR
@app.post("/api/payment-webhook/{order_id}")
async def confirm_payment(order_id: str):
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"status": "error", "message": "Chưa có dữ liệu đơn hàng."}

    for item in data:
        if item.get("ma_don") == order_id:
            item["trang_thai"] = "DA_THANH_TOAN_COC"
            with open(DB_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return {
                "status": "success",
                "order_id": order_id,
                "customer_name": item["ho_ten"],
                "car_model": item["mau_xe"],
                "amount": item["so_tien_coc"]
            }
    return {"status": "error", "message": "Không tìm thấy mã đơn."}

# API kiểm tra trạng thái đơn (phục vụ tự động kiểm tra)
@app.get("/api/order-status/{order_id}")
async def check_order_status(order_id: str):
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("ma_don") == order_id:
                return {"status": item.get("trang_thai")}
    except FileNotFoundError:
        pass
    return {"status": "NOT_FOUND"}