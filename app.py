import logging
import sys
import base64
import io
import time
import requests
import re
import streamlit as st
from openai import OpenAI
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import math

# ─── Logging & OpenAI setup ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)
client = OpenAI(api_key="ADD-KEY-HERE")

# ─── Page & RTL CSS ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="ذكاء التدوير المجتمعي", layout="wide")
st.markdown(
    """<style>
         html, body, [class*="css"] { direction: rtl; unicode-bidi: embed; }
       </style>""",
    unsafe_allow_html=True
)

st.title("🤖♻️ ذكاء التدوير المجتمعي")
st.write(
    "حمّل صورة لتوليد معلومات عن المخاطر وتقنيات التقليل وإعادة الاستخدام والتدوير، واستكشف مراكز التدوير القريبة."
)

# ─── Sidebar radius slider ─────────────────────────────────────────────────────
radius_km = st.sidebar.slider(
    "حدد نصف القطر للبحث عن المراكز (كم)",
    min_value=1,
    max_value=20,
    value=5
)
st.sidebar.markdown("🔄 خريطة OSM عبر Overpass—لا حاجة لمفاتيح خارجية.")

# ─── 1) Upload image ───────────────────────────────────────────────────────────
uploaded = st.file_uploader("📷 حمّل صورتك هنا", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()
img_bytes = uploaded.read()
st.image(img_bytes, caption="صورتك", use_container_width=True)
b64 = base64.b64encode(img_bytes).decode()

# ─── 2) Dangers/Reduce/Reuse/Recycle analysis using multimodal GPT ──────────────
if "analysis" not in st.session_state:
    with st.spinner("جاري تحليل المادة..."):
        system_msg = {
            "role": "system",
            "content": "أنت مساعد بيئي ذكي *يمكنه رؤية الصورة* ويجب أن يعتمد في إجاباتك على ما تراه فيها."
        }
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {
                    "type": "text",
                    "text": (
                        "اعتماداً على ما هو ظاهر في الصورة، قدم أربع أقسام باللغة العربية، "
                        "موسومة تماماً كما يلي:\n\n"
                        "Dangers: أذكر المخاطر البيئية التي قد تنشأ إذا تم التعامل مع هذه المادة بشكل غير سليم.\n"
                        "Reduce: أذكر طرق عملية لتقليل استخدام هذه المادة أو تقليل نفاياتها.\n"
                        "Reuse: بالضبط 3 أفكار مستقلة ومفصلة بالخطوات، اكتب كل فكرة مرقمة بصيغة:\n"
                        "1. الفكرة الأولى...\n"
                        "2. الفكرة الثانية...\n"
                        "3. الفكرة الثالثة...\n"
                        "ولا تذكر أي نص آخر.\n"
                        "Recycle: أذكر خطوات أو طرق لتدوير هذه المادة بشكل صحيح."
                    )
                }
            ]
        }
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_msg, user_msg]
        )
        raw = resp.choices[0].message.content

        # split by labels
        labels = ["Dangers:", "Reduce:", "Reuse:", "Recycle:"]
        positions = [(lbl, raw.find(lbl)) for lbl in labels if raw.find(lbl) != -1]
        positions.sort(key=lambda x: x[1])
        sections = {}
        for idx, (lbl, pos) in enumerate(positions):
            start = pos + len(lbl)
            end = positions[idx+1][1] if idx+1 < len(positions) else len(raw)
            key = lbl[:-1].lower()  # e.g. "dangers"
            sections[key] = raw[start:end].strip()

        st.session_state.analysis = sections

        # extract exactly three reuse ideas by numbering
        reuse_raw = sections.get("reuse", "")
        ideas = []
        for line in reuse_raw.split("\n"):
            m = re.match(r'^\s*(\d+)\.\s*(.+)', line)
            if m:
                ideas.append(m.group(2).strip())
        # fallback: any non-empty
        if len(ideas) < 3:
            ideas = [l.strip() for l in reuse_raw.split("\n") if l.strip()][:3]
        st.session_state.reuse_ideas = ideas[:3]

# ─── 3) Display analysis ───────────────────────────────────────────────────────
st.subheader("⚠️ المخاطر (Dangers)")
st.markdown(st.session_state.analysis.get("dangers", "—"))

st.subheader("➖ التقليل (Reduce)")
st.markdown(st.session_state.analysis.get("reduce", "—"))

st.subheader("🔄 إعادة الاستخدام (Reuse)")
for idea in st.session_state.reuse_ideas:
    st.markdown(f"- {idea}")

st.subheader("♻️ التدوير (Recycle)")
st.markdown(st.session_state.analysis.get("recycle", "—"))

# ─── 4) Generate & cache English scene prompts + images ────────────────────────
if "reuse_images" not in st.session_state:
    st.session_state.reuse_images = []
    with st.spinner("جاري إعداد أوصاف الصور بالإنجليزية..."):
        reuse_english_prompt = (
            "You are an AI image prompt generator. For each of these Arabic reuse ideas, "
            "produce one detailed English scene description of the final upcycled product. "
            "Include colors, materials, scale, and lighting. Do not use commands.\n\n"
            "Arabic ideas:\n" + "\n".join(st.session_state.reuse_ideas)
        )
        resp_desc = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate detailed English image prompts."},
                {"role": "user", "content": reuse_english_prompt}
            ]
        )
        descs = [d.strip() for d in resp_desc.choices[0].message.content.split("\n") if d.strip()][:3]
        for desc in descs:
            with st.spinner(f"جاري توليد صورة لـ: {desc[:30]}..."):
                img_resp = client.images.generate(
                    model="dall-e-2",
                    prompt=desc,
                    n=1,
                    size="1024x1024",
                    response_format="b64_json"
                )
                st.session_state.reuse_images.append(
                    base64.b64decode(img_resp.data[0].b64_json)
                )
                time.sleep(0.1)

st.subheader("📸 صور إعادة الاستخدام")
cols = st.columns(3)
for col, img_data, cap in zip(cols, st.session_state.reuse_images, st.session_state.reuse_ideas):
    with col:
        st.image(io.BytesIO(img_data), caption=cap, use_container_width=True)

# ─── 5) Map via Overpass (true radius) ────────────────────────────────────────
st.subheader("🗺️ مراكز التدوير القريبة")

# determine center coords
lat = lon = None
try:
    info = requests.get("https://ipinfo.io/json", timeout=3).json()
    if loc := info.get("loc"):
        lat, lon = map(float, loc.split(","))
except:
    pass

if lat is None:
    city = st.text_input("أدخل اسم مدينتك للعثور على المراكز:")
    if city:
        geo = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "CommunityRecyclingApp/1.0"},
            timeout=5
        ).json()
        if geo:
            lat, lon = float(geo[0]["lat"]), float(geo[0]["lon"])

if lat is None or lon is None:
    st.error("تعذر تحديد موقعك. تأكد من السماح بالوصول أو كتابة اسم المدينة.")
    st.stop()

# query Overpass for recycling amenities within radius
radius_m = int(radius_km * 1000)
overpass_q = f"""
[out:json][timeout:25];
(
  node(around:{radius_m},{lat},{lon})[amenity=recycling];
  way(around:{radius_m},{lat},{lon})[amenity=recycling];
  rel(around:{radius_m},{lat},{lon})[amenity=recycling];
);
out center;
"""
resp_ov = requests.get(
    "https://overpass-api.de/api/interpreter",
    params={"data": overpass_q},
    timeout=30
).json()
elems = resp_ov.get("elements", [])

markers = []
for el in elems:
    if el["type"] == "node":
        plat, plon = el["lat"], el["lon"]
    else:
        center = el.get("center", {})
        plat, plon = center.get("lat"), center.get("lon")
    name = el.get("tags", {}).get("name", "مركز التدوير")
    markers.append((plat, plon, name))

m = folium.Map(location=[lat, lon], zoom_start=12)
folium.Circle([lat, lon], radius=radius_m, color="blue", fill=False).add_to(m)
cluster = MarkerCluster().add_to(m)
for plat, plon, name in markers:
    folium.Marker([plat, plon], popup=name).add_to(cluster)

st.write(f"تم العثور على {len(markers)} مركز تدوير ضمن {radius_km} كم.")
st_folium(m, width=800, height=500)
