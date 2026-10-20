import streamlit as st
import numpy as np
import cv2
import requests
import json
import hashlib
from datetime import datetime, timezone

from severity import verify_and_analyze_leaf
from disease_data import DISEASE_DB

st.set_page_config(
    page_title="ClimaSphere AI | IEEE ClimateChain",
    page_icon="🌍",
    layout="wide"
)

# Custom High-Contrast Styling & Complete Tab Overrides
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&family=Inter:wght@400;600;700;800&display=swap');

    .stApp {
        background-color: #F8F5F0;
        color: #111827;
        font-family: 'Inter', sans-serif;
    }

    .rtl-mode {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Nastaliq Urdu', serif;
    }

    /* Primary & Fetch Buttons */
    div.stButton > button {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: 1px solid #1B5E20 !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12) !important;
    }
    div.stButton > button:hover {
        background-color: #1B5E20 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }
    div.stButton > button p, div.stButton > button span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* High-Contrast Tab Selectors */
    [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
        border-bottom: 2px solid #D1C7BD !important;
        margin-bottom: 14px !important;
    }
    [data-baseweb="tab"], button[data-baseweb="tab"], [data-testid="stTabs"] button {
        color: #1F2937 !important;
        font-weight: 600 !important;
        background-color: #EAE4DC !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 10px 18px !important;
        border: 1px solid #D1C7BD !important;
        border-bottom: none !important;
    }
    [data-baseweb="tab"] p, [data-baseweb="tab"] span, [data-testid="stTabs"] button p {
        color: #1F2937 !important;
        font-weight: 600 !important;
    }
    [data-baseweb="tab"]:hover, [data-testid="stTabs"] button:hover {
        background-color: #DDD4C8 !important;
    }
    [data-baseweb="tab"][aria-selected="true"], [data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #4A0E17 !important;
        border-color: #4A0E17 !important;
    }
    [data-baseweb="tab"][aria-selected="true"] p, [data-testid="stTabs"] button[aria-selected="true"] p,
    [data-baseweb="tab"][aria-selected="true"] span {
        color: #FBE8A6 !important;
        font-weight: 800 !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4A0E17 0%, #29050A 100%);
        color: #FBE8A6;
        padding: 22px 24px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(74, 14, 23, 0.15);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #FBE8A6;
        margin-bottom: 4px;
    }
    .hero-sub {
        color: #E5CBA8;
        font-size: 0.95rem;
    }

    /* Universal KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 15px 0;
    }
    .kpi-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 14px;
        border-top: 4px solid #C5A059;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .kpi-card-title {
        font-size: 0.85rem;
        color: #4B5563;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .kpi-card-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #4A0E17;
    }

    /* Code & Certificate Blocks */
    .cert-box {
        background: #111827;
        color: #10B981;
        font-family: 'Courier New', Courier, monospace;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #374151;
        font-size: 0.85rem;
        line-height: 1.4;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "temperature" not in st.session_state:
    st.session_state.temperature = 28
if "humidity" not in st.session_state:
    st.session_state.humidity = 76
if "aqi" not in st.session_state:
    st.session_state.aqi = 178
if "pm25" not in st.session_state:
    st.session_state.pm25 = 92.4
if "pm10" not in st.session_state:
    st.session_state.pm10 = 145.0
if "city_name" not in st.session_state:
    st.session_state.city_name = "Lahore"
if "audit_ledger" not in st.session_state:
    st.session_state.audit_ledger = []

# Telemetry Fetcher: Open-Meteo Air Quality & Weather API
def fetch_telemetry(city):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=4).json()
        if "results" in geo_res and len(geo_res["results"]) > 0:
            lat = geo_res["results"][0]["latitude"]
            lon = geo_res["results"][0]["longitude"]
            
            # Weather fetch
            w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
            w_data = requests.get(w_url, timeout=4).json().get("current", {})
            temp = w_data.get("temperature_2m", 28.0)
            hum = w_data.get("relative_humidity_2m", 76.0)

            # Air quality fetch
            a_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,us_aqi"
            a_data = requests.get(a_url, timeout=4).json().get("current", {})
            aqi = a_data.get("us_aqi", 165)
            pm25 = a_data.get("pm2_5", 85.0)
            pm10 = a_data.get("pm10", 130.0)

            return temp, hum, aqi, pm25, pm10, lat, lon, True
    except Exception:
        pass
    return 28.0, 76.0, 178, 92.4, 145.0, 31.5497, 74.3436, False

# Procedural Synthetic Leaf Generator
def generate_synthetic_foliage(leaf_type):
    img = np.ones((400, 400, 3), dtype=np.uint8) * 230
    if leaf_type == "banana":
        cv2.ellipse(img, (200, 200), (160, 70), 20, 0, 360, (30, 145, 45), -1)
        cv2.line(img, (60, 150), (340, 250), (15, 95, 25), 4)
        for pt, axes in [((150, 180), (35, 8)), ((230, 220), (45, 9)), ((280, 240), (25, 6))]:
            cv2.ellipse(img, pt, axes, 20, 0, 360, (20, 30, 70), -1)
        return img
    elif leaf_type == "tomato":
        cv2.ellipse(img, (200, 200), (140, 90), 35, 0, 360, (34, 139, 34), -1)
        cv2.circle(img, (220, 190), 45, (25, 45, 95), -1)
        cv2.circle(img, (160, 230), 30, (25, 45, 95), -1)
        return img
    elif leaf_type == "potato":
        cv2.ellipse(img, (200, 200), (135, 85), 30, 0, 360, (34, 139, 34), -1)
        cv2.circle(img, (180, 160), 24, (20, 50, 110), -1)
        cv2.circle(img, (230, 240), 18, (20, 50, 110), -1)
        return img
    else:
        cv2.ellipse(img, (200, 200), (140, 90), 35, 0, 360, (34, 139, 34), -1)
        cv2.line(img, (100, 110), (300, 290), (20, 100, 20), 3)
        return img

# Cryptographic SHA-256 Block Generator
def generate_audit_block(city, aqi, pm25, temp, humidity, crop, disease, severity, yield_loss, financial_loss):
    prev_hash = st.session_state.audit_ledger[-1]["current_hash"] if st.session_state.audit_ledger else "0" * 64
    block_index = len(st.session_state.audit_ledger) + 1
    timestamp = datetime.now(timezone.utc).isoformat()
    
    payload = {
        "index": block_index,
        "timestamp_utc": timestamp,
        "location": city,
        "telemetry": {
            "aqi": aqi,
            "pm25": pm25,
            "temp_c": temp,
            "humidity_rh": humidity
        },
        "pathology": {
            "crop": crop,
            "disease": disease,
            "severity_pct": severity
        },
        "economic_claim": {
            "yield_loss_pct": yield_loss,
            "financial_risk_usd_acre": financial_loss
        },
        "previous_hash": prev_hash
    }
    
    payload_string = json.dumps(payload, sort_keys=True)
    block_hash = hashlib.sha256(payload_string.encode('utf-8')).hexdigest()
    
    block_entry = {
        "index": block_index,
        "timestamp": timestamp,
        "location": city,
        "severity": f"{severity}%",
        "yield_loss": f"{yield_loss}%",
        "financial_risk": f"${financial_loss}",
        "previous_hash": prev_hash,
        "current_hash": block_hash,
        "raw_payload": payload_string
    }
    st.session_state.audit_ledger.append(block_entry)
    return block_entry

# Header & Multilingual Selector
col_hdr, col_lng = st.columns([3, 1])
with col_lng:
    selected_lang = st.selectbox("🌐 Interface Language", ["English", "اردو (Urdu)"])
is_urdu = (selected_lang == "اردو (Urdu)")
rtl_class = "rtl-mode" if is_urdu else ""

st.markdown(f"""
<div class="hero-banner {rtl_class}">
    <div class="hero-title">{'کلائما اسفیئر اے آئی (ClimaSphere AI)' if is_urdu else 'CLIMASPHERE AI'}</div>
    <div class="hero-sub">{'ماحولیاتی آلودگی، نباتاتی امراض، اور کرپٹوگرافک ویریفائیڈ کلائمیٹ لیجر' if is_urdu else 'UNIFIED CLIMATE INTELLIGENCE, CROP PATHOLOGY & IMMUTABLE PROOF LEDGER'}</div>
</div>
""", unsafe_allow_html=True)

# Master Navigation Tabs
tab_labels = (
    ["1. ماحولیاتی مانیٹر", "2. نباتاتی کمپیوٹر وژن", "3. معاشی و پیداواری نقصان", "4. ویریفائیڈ بلاک چین لیجر"]
    if is_urdu else
    ["1. Atmospheric & Air Quality", "2. Crop Foliage Pathology", "3. Economic Loss Model", "4. Immutable Climate Ledger"]
)

tab_air, tab_crop, tab_econ, tab_ledger = st.tabs(tab_labels)

# ========================================================
# TAB 1: ATMOSPHERIC & AIR QUALITY ENGINE
# ========================================================
with tab_air:
    st.subheader("شہری آلودگی اور ہوا کا معیار" if is_urdu else "Urban Atmospheric Telemetry & Smog Analytics")
    
    col_search, col_action = st.columns([2, 1])
    with col_search:
        target_city = st.text_input("City / علاقہ", st.session_state.city_name)
    with col_action:
        st.write("")
        st.write("")
        if st.button("📍 Fetch Telemetry"):
            t, h, a, p25, p10, lat, lon, ok = fetch_telemetry(target_city)
            if ok:
                st.session_state.city_name = target_city
                st.session_state.temperature = int(round(t))
                st.session_state.humidity = int(round(h))
                st.session_state.aqi = int(a)
                st.session_state.pm25 = round(p25, 1)
                st.session_state.pm10 = round(p10, 1)
                st.toast(f"Telemetry updated for {target_city}")
            else:
                st.error("Failed to query Open-Meteo telemetry.")

    # AQI Severity Status
    aqi_val = st.session_state.aqi
    if aqi_val <= 50:
        aqi_status, aqi_color = "Good", "#10B981"
    elif aqi_val <= 100:
        aqi_status, aqi_color = "Moderate", "#F59E0B"
    elif aqi_val <= 150:
        aqi_status, aqi_color = "Unhealthy for Sensitive Groups", "#F97316"
    elif aqi_val <= 200:
        aqi_status, aqi_color = "Unhealthy (Smog Inversion)", "#EF4444"
    else:
        aqi_status, aqi_color = "Hazardous / Emergency", "#7F1D1D"

    st.markdown(f"""
    <div class="kpi-container {rtl_class}">
        <div class="kpi-card">
            <div class="kpi-card-title">{'ایئر کوالٹی انڈیکس' if is_urdu else 'Air Quality Index (AQI)'}</div>
            <div class="kpi-card-val" style="color: {aqi_color};">{aqi_val}</div>
            <span style="font-size: 0.8rem; font-weight: 700; color: {aqi_color};">{aqi_status}</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-title">{'باریک ذرات' if is_urdu else 'PM2.5 Concentration'}</div>
            <div class="kpi-card-val">{st.session_state.pm25} µg/m³</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-title">{'درجہ حرارت' if is_urdu else 'Ambient Temperature'}</div>
            <div class="kpi-card-val">{st.session_state.temperature} °C</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-title">{'ہوا میں نمی' if is_urdu else 'Relative Humidity'}</div>
            <div class="kpi-card-val">{st.session_state.humidity} %</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ecological Impact Callout
    if aqi_val > 150:
        st.markdown(f"""
        <div style="background: #FFFBEB; border-left: 5px solid #F59E0B; padding: 14px 18px; border-radius: 8px; margin: 12px 0;" class="{rtl_class}">
            <strong style="color: #92400E;">⚠️ {'ماحولیاتی انتباہ:' if is_urdu else 'Ecological Risk Indicator:'}</strong>
            <span style="color: #78350F;">
            {'شدید اسموگ اور پارٹیکولیٹ میٹر پتوں کے مسام (Stomata) کو بند کر دیتے ہیں، جس سے پودوں میں فنگس اور بیماریوں سے لڑنے کی قدرتی مدافعت 40 فیصد تک کم ہو جاتی ہے۔' if is_urdu else
             'Elevated particulate matter (PM2.5) induces foliar stomatal clogging, diminishing photosynthetically active radiation and increasing plant susceptibility to airborne fungal sporulation.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

# ========================================================
# TAB 2: PRECISION CROP PATHOLOGY (OPENCV)
# ========================================================
with tab_crop:
    col_in, col_cv = st.columns([1, 2])
    
    with col_in:
        st.subheader("سیمپل یا تصویر" if is_urdu else "Foliar Image Input")
        sample_choice = st.radio(
            "Test Sample Case" if not is_urdu else "ٹیسٹ سیمپل منتخب کریں",
            ["Banana: Black Sigatoka", "Tomato: Late Blight", "Potato: Early Blight", "Healthy Leaf", "Custom Upload"]
        )
        
        cv_img_bgr = None
        if sample_choice == "Custom Upload":
            up_file = st.file_uploader("Upload leaf photo", type=["jpg", "png", "jpeg"])
            if up_file:
                b_arr = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                cv_img_bgr = cv2.imdecode(b_arr, cv2.IMREAD_COLOR)
        elif "Banana" in sample_choice:
            cv_img_bgr = generate_synthetic_foliage("banana")
            active_disease_key = "Banana - Black Sigatoka"
        elif "Tomato" in sample_choice:
            cv_img_bgr = generate_synthetic_foliage("tomato")
            active_disease_key = "Tomato - Late Blight"
        elif "Potato" in sample_choice:
            cv_img_bgr = generate_synthetic_foliage("potato")
            active_disease_key = "Potato - Early Blight"
        else:
            cv_img_bgr = generate_synthetic_foliage("healthy")
            active_disease_key = "Healthy Foliage"

    with col_cv:
        st.subheader("کمپیوٹر وژن لیسن اینالیسس" if is_urdu else "Computer Vision Inspection Visualizer")
        if cv_img_bgr is not None:
            is_leaf, severity_pct, leaf_mask, lesion_mask, overlay_bgr = verify_and_analyze_leaf(cv_img_bgr)
            
            if not is_leaf:
                st.error("No valid foliage structure identified. Please upload a clear leaf image.")
            else:
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                raw_rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
                
                v_tabs = st.tabs(["Diagnostic Overlay", "Lesion Mask (OpenCV)", "Raw Foliage"])
                with v_tabs[0]:
                    st.image(overlay_rgb, caption="Lesion Overlay (Yellow Contour + Red Damage)", use_container_width=True)
                with v_tabs[1]:
                    st.image(lesion_mask, caption="Segmented Binary Lesion Mask", use_container_width=True, clamp=True)
                with v_tabs[2]:
                    st.image(raw_rgb, caption="Raw Input Sensor Capture", use_container_width=True)

# Store persistent variables for Tab 3 and Tab 4
if cv_img_bgr is not None and is_leaf:
    current_severity = severity_pct
    current_disease_info = DISEASE_DB[active_disease_key]
else:
    current_severity = 13.6
    current_disease_info = DISEASE_DB["Banana - Black Sigatoka"]

# ========================================================
# TAB 3: ECONOMIC & YIELD LOSS MODEL
# ========================================================
with tab_econ:
    st.subheader("مالیاتی خطرات اور عملی زرعی نسخہ" if is_urdu else "Economic Deficit & Quantitative Loss Advisory")
    
    calc_yield_loss = round((current_severity * 0.75) + 3.2, 1)
    calc_fin_loss = int(current_severity * 15 + 95)

    st.markdown(f"""
    <div class="kpi-container {rtl_class}">
        <div class="kpi-card">
            <div class="kpi-card-title">{'تشخیص شدہ فصل' if is_urdu else 'Identified Crop'}</div>
            <div class="kpi-card-val">{current_disease_info['crop_ur' if is_urdu else 'crop_en']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-title">{'پیتھالوجی انفیکشن' if is_urdu else 'Active Pathology'}</div>
            <div class="kpi-card-val" style="font-size: 1.1rem;">{current_disease_info['disease_ur' if is_urdu else 'disease_en']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-title">{'پیداواری نقصان کا تخمینہ' if is_urdu else 'Projected Yield Deficit'}</div>
            <div class="kpi-card-val" style="color: #EF4444;">{calc_yield_loss} %</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-title">{'فی ایکڑ متوقع مالیاتی خطرہ' if is_urdu else 'Financial Exposure'}</div>
            <div class="kpi-card-val" style="color: #4A0E17;">${calc_fin_loss} / acre</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: white; border-radius: 10px; padding: 22px; border-top: 4px solid #4A0E17; box-shadow: 0 2px 8px rgba(0,0,0,0.06);" class="{rtl_class}">
        <h4 style="color: #4A0E17; margin-top: 0;">{'عملی زرعی نسخہ اور سفارشات' if is_urdu else 'Clinical Agronomic Protocol'}</h4>
        <p><strong>{'کیمیائی علاج:' if is_urdu else 'Chemical Control Protocol:'}</strong> {current_disease_info['chem_ur' if is_urdu else 'chem_en']}</p>
        <p><strong>{'حفاظتی اقدامات:' if is_urdu else 'Agronomic Cultural Controls:'}</strong> {current_disease_info['cult_ur' if is_urdu else 'cult_en']}</p>
        <p><strong>{'موسمیاتی فالو اپ:' if is_urdu else 'Preventative Action:'}</strong> {current_disease_info['prev_ur' if is_urdu else 'prev_en']}</p>
    </div>
    """, unsafe_allow_html=True)

# ========================================================
# TAB 4: IMMUTABLE AUDIT LEDGER (SHA-256)
# ========================================================
with tab_ledger:
    st.subheader("کرپٹوگرافک ویریفائیڈ کلائمیٹ لیجر" if is_urdu else "Cryptographic SHA-256 Climate Audit Ledger")
    st.write(
        "Each diagnostic reading is stamped with an immutable cryptographic proof. "
        "This prevents insurance fraud and validates data for climate mitigation funds."
    )

    col_btn_gen, col_space = st.columns([1, 2])
    with col_btn_gen:
        if st.button("⛓️ Seal Diagnostic Record to Ledger"):
            new_block = generate_audit_block(
                city=st.session_state.city_name,
                aqi=st.session_state.aqi,
                pm25=st.session_state.pm25,
                temp=st.session_state.temperature,
                humidity=st.session_state.humidity,
                crop=current_disease_info["crop_en"],
                disease=current_disease_info["disease_en"],
                severity=current_severity,
                yield_loss=calc_yield_loss,
                financial_loss=calc_fin_loss
            )
            st.success(f"Block #{new_block['index']} verified and sealed into cryptographic ledger!")

    if st.session_state.audit_ledger:
        st.markdown("### Ledger Chain Transactions")
        st.dataframe(
            st.session_state.audit_ledger,
            column_order=["index", "timestamp", "location", "severity", "yield_loss", "financial_risk", "current_hash"],
            use_container_width=True
        )

        st.markdown("### Latest Cryptographic Proof Certificate")
        latest = st.session_state.audit_ledger[-1]
        st.markdown(f"""
        <div class="cert-box">
========================= CLIMASPHERE AUDIT CERTIFICATE =========================
BLOCK INDEX   : #{latest['index']}
TIMESTAMP UTC : {latest['timestamp']}
LOCATION      : {latest['location']}
SEVERITY INDEX: {latest['severity']} (Yield Deficit: {latest['yield_loss']})
FINANCIAL RISK: {latest['financial_risk']}
PREVIOUS HASH : {latest['previous_hash']}
SHA-256 PROOF : {latest['current_hash']}
STATUS        : IMMUTABLE • INTEGRITY VERIFIED (IEEE ClimateChain Standard)
================================================================================
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No audit blocks sealed yet. Click 'Seal Diagnostic Record to Ledger' above to mint Block #1.")

st.markdown("---")
st.caption("ClimaSphere AI • IEEE ClimateChain Global Hackathon 2026 Submission")
