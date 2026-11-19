"""Konfigurasi global, leksikon aspek/sentimen, dan kamus preprocessing."""

SEED = 42
SENT_LABELS = ["negative", "neutral", "positive"]
OTHER = "Other/Unclear"

DEFAULT_PATHS = [
    "data/processed/lms_reviews_clean.csv",
    "data/raw/[Data Set] Combined LMS App Review.xlsx",
]

# Nama kanonik -> kandidat nama kolom asli (dicocokkan case-insensitive, tanpa spasi/underscore)
COLUMN_CANDIDATES = {
    "institution": ["institution_name", "institution", "university", "institusi"],
    "app": ["app_name", "app", "application", "aplikasi"],
    "review_text": ["review_body", "review_text", "content", "review", "text", "ulasan"],
    "review_title": ["review_title", "title", "judul"],
    "rating": ["star_rating", "rating", "score", "stars", "bintang"],
    "likes": ["likes", "thumbs_up", "thumbsup", "helpful_count"],
    "date": ["review_created_at", "review_date", "date", "created_at", "at", "tanggal"],
    "version": ["app_version", "version", "review_created_version"],
    "language": ["language", "lang", "bahasa"],
    "group": ["app_scope_flag", "scope", "country", "group", "app_scope", "negara"],
    "review_id": ["review_id", "id", "reviewid"],
}

# ---- Aspek usability: fragmen regex (prefix-word match), bilingual ID/EN ----
ASPECTS = {
    "Usability/Navigation": [
        r"ui\b", r"ux\b", r"tampilan", r"interface", r"navigas", r"menu", r"desain", r"design",
        r"layout", r"user.?friendly", r"ribet", r"rumit", r"membingungkan", r"bingung", r"confus",
        r"(susah|sulit|hard|difficult)( to)? ?(di)?(pakai|gunakan|digunakan|dipahami|cari|use|find|navigate)",
        r"mudah (di)?(pakai|digunakan|dipahami)", r"easy to use", r"simpel", r"simple", r"intuitive",
        r"tombol", r"button", r"halaman", r"user experience",
    ],
    "Login & Authentication": [
        r"login", r"log in", r"log.?out", r"sign.?in", r"masuk", r"password", r"kata sandi", r"sandi",
        r"akun", r"account", r"sso\b", r"otp\b", r"verifikasi", r"verification", r"captcha",
        r"autentikasi", r"authenticat", r"token", r"daftar", r"register", r"registrasi", r"lupa", r"forgot", r"reset",
    ],
    "Performance/Stability": [
        r"error", r"eror", r"lemot", r"lambat", r"loading", r"lag\b", r"ngelag", r"crash", r"force close",
        r"keluar sendiri", r"hang\b", r"freeze", r"macet", r"bug", r"server", r"down\b", r"gangguan",
        r"maintenance", r"slow", r"stuck", r"berat", r"cepat", r"fast", r"responsive", r"gagal", r"failed",
        r"(tidak|gak|ga|nggak|not|cant|can't|cannot) ?(bisa )?(di)?(buka|open|load|work|respon)",
    ],
    "Features/Functionality": [
        r"fitur", r"feature", r"absen", r"presensi", r"tugas", r"assignment", r"kuis", r"quiz", r"ujian",
        r"exam", r"nilai", r"grade", r"jadwal", r"schedule", r"krs\b", r"khs\b", r"download", r"unduh",
        r"upload", r"unggah", r"notifikasi", r"notification", r"forum", r"materi", r"video", r"zoom",
        r"chat", r"kalender", r"calendar", r"submit", r"kumpul",
    ],
    "Accessibility": [
        r"offline", r"kuota", r"dark.?mode", r"mode gelap", r"font", r"huruf", r"tulisan", r"ukuran",
        r"tablet", r"ios\b", r"iphone", r"android", r"compatib", r"kompatibel", r"device", r"perangkat",
        r"layar", r"screen", r"bahasa", r"language", r"aksesibil", r"accessib", r"kontras", r"contrast",
        r"storage", r"memori",
    ],
    "Information/Content": [
        r"informasi", r"information", r"info\b", r"konten", r"content", r"pengumuman", r"announcement",
        r"berita", r"news", r"sinkron", r"sync", r"tidak (sesuai|muncul|tampil)", r"not showing",
        r"missing", r"outdated", r"lengkap", r"detail", r"data\b",
    ],
    "Support/Communication": [
        r"admin", r"support", r"helpdesk", r"help desk", r"respon[s]?\b", r"response", r"komplain",
        r"keluhan", r"kontak", r"contact", r"developer", r"pengembang", r"customer service", r"cs\b",
        r"hubungi", r"feedback", r"laporan", r"report", r"email", r"balas", r"reply", r"dosen",
        r"lecturer", r"komunikasi", r"bantuan", r"help\b",
    ],
}
ASPECT_NAMES = list(ASPECTS.keys())

# ---- Leksikon sentimen (kecil, transparan; untuk ABSA tingkat klausa) ----
POS_WORDS = set("""bagus baik mantap mantab keren mudah membantu bantu helpful good great nice excellent love best
useful lancar cepat fast smooth simpel simple praktis puas recommended rekomen top oke ok suka makasih amazing awesome
easy perfect sempurna friendly responsif bermanfaat memuaskan emo_pos terbaik""".split())
NEG_WORDS = set("""jelek buruk parah lemot lambat error eror bug susah sulit ribet rumit gagal bad worst terrible slow
crash stuck hang lag annoying useless sampah kecewa mengecewakan bingung confusing hate poor broken gangguan down lama
mahal berat capek menyebalkan nyebelin frustrasi frustrating waste failed problem masalah unstable ngelag hilang salah
emo_neg""".split())
NEGATIONS = set("""tidak tak bukan belum jangan tanpa gak ga nggak ngga enggak engga tdk not no never cannot cant can't
dont don't doesnt doesn't isnt isn't wont won't didnt didn't gabisa unable""".split())
CAPABILITY = set("""bisa dibuka buka work working open masuk login load muncul access akses kebuka terbuka tampil
berfungsi respon""".split())

# ---- Preprocessing ----
SLANG = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "engga": "tidak",
    "enggak": "tidak", "tdk": "tidak", "tak": "tidak", "bgt": "sekali", "banget": "sekali", "yg": "yang",
    "udh": "sudah", "sdh": "sudah", "udah": "sudah", "blm": "belum", "dgn": "dengan", "utk": "untuk",
    "krn": "karena", "jd": "jadi", "trs": "terus", "bs": "bisa", "gabisa": "tidak bisa", "gaada": "tidak ada",
    "mantul": "mantap", "bagu": "bagus", "eror": "error", "sy": "saya", "tp": "tapi", "dr": "dari",
    "pake": "pakai", "pls": "please", "plis": "please", "thx": "thanks", "apk": "aplikasi",
    "lgi": "lagi", "kk": "kak", "aja": "saja", "doang": "saja", "bkn": "bukan", "jgn": "jangan",
}
EMOJI_MAP = {
    **{e: " emo_pos " for e in "😀😃😄😁😆😊🙂😍🥰😘👍👏🙏❤💯🔥✨🎉😎🤩👌💕😻"},
    **{e: " emo_neg " for e in "😡😠🤬😤😞😢😭😩😫👎💔😒🙄😖😣😔🤮😱"},
}
STOP_ID = set("""yang dan di ke dari ini itu untuk dengan pada adalah ada saya kami kita akan juga atau sudah sekali saja
nya lagi karena jadi oleh agar para tapi sih dong kok ya""".split())
STOP_EN = set("""the a an and or of to in on at for with is are was were be been it this that i you we they my your its
as by from so but""".split())
ID_HINT = set("""yang dan tidak di untuk ini itu tolong bisa saya sudah dengan karena tapi gak ga banget sekali mau kok masuk
susah sangat bagus tugas kuliah mahasiswa aplikasi nya lagi semua padahal harus kenapa sering belum""".split())
EN_HINT = set("""the and is this to not it app can with for very but please my i you cant works good always when after
every want need why have has too really""".split())
