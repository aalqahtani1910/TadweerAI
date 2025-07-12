import logging
import sys
import base64
import io
import requests
import streamlit as st
from openai import OpenAI
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import math
import time

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# ─── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(api_key="sk-proj-jJFAKAQkRxmfY34xg5YdT3BlbkFJ9hGxMN9nMsNzILVuanrB")

# ─── Page/UI setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="ذكاء التدوير المجتمعي", layout="wide")
st.markdown("""
  <style>
    html, body, [class*="css"] { direction: rtl; unicode-bidi: embed; }
  </style>
""", unsafe_allow_html=True)

st.title("🤖♻️ ذكاء التدوير المجتمعي")
st.write("حمّل صورة لتوليد معلومات عن المخاطر وتقنيات التقليل وإعادة الاستخدام والتدوير، واستكشف مراكز التدوير القريبة.")

# ─── Sidebar controls ──────────────────────────────────────────────────────────
radius_km = st.sidebar.slider(
    "اختر نصف القطر للبحث عن مراكز التدوير (كم)",
    min_value=1, max_value=20, value=5
)
st.sidebar.markdown("**أعد تحميل الصفحة** إذا غيرت نصف القطر أو المدينة لتحديث الخريطة.")

# ─── 1) Upload image ───────────────────────────────────────────────────────────
uploaded = st.file_uploader("📷 حمّل صورتك هنا", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

img_bytes = uploaded.read()
st.image(img_bytes, caption="صورتك", use_container_width=True)
b64 = base64.b64encode(img_bytes).decode()

# ─── 2) Generate analysis sections ──────────────────────────────────────────────
if "analysis" not in st.session_state:
    with st.spinner("جاري تحليل المادة..."):
        full_prompt = (
            "أعطني المعلومات التالية للمادة/العناصر الموجودة في الصورة، باللغة العربية،"
            " وبذلك التنسيق:\n\n"
            "Dangers: أذكر مخاطر هذه المادة على البيئة أو الصحة.\n"
            "Reduce: أذكر طرق عملية لتقليل استخدام هذه المادة أو تقليل نفاياتها.\n"
            "Reuse: أعطني 3 أفكار مفصلة بالخطوات لإعادة استخدام المادة الموجودة في الصورة.\n"
            "Recycle: أذكر خطوات أو طرق لتدوير هذه المادة بشكل صحيح.\n\n"
            f"الصورة مقدمة هنا: data:image/jpeg;base64,{b64}\n\n"
            "لا تستخدم أي تنسيق إضافي، فقط النص."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت مساعد ذكي بيئي ومهتم بالتدوير."},
                {"role": "user", "content": full_prompt}
            ]
        )
        text = resp.choices[0].message.content
        # parse into sections
        sections = {}
        current = None
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Dangers:"):
                current = "dangers"; sections[current] = line[len("Dangers:"):].strip()
            elif line.startswith("Reduce:"):
                current = "reduce"; sections[current] = line[len("Reduce:"):].strip()
            elif line.startswith("Reuse:"):
                current = "reuse"; sections[current] = line[len("Reuse:"):].strip()
            elif line.startswith("Recycle:"):
                current = "recycle"; sections[current] = line[len("Recycle:"):].strip()
            elif current:
                sections[current] += "\n" + line
        st.session_state.analysis = sections
        # extract reuse ideas
        reuse_lines = [l for l in sections.get("reuse","").split("\n") if l]
        st.session_state.reuse_ideas = reuse_lines[:3]

# Show analysis
st.subheader("⚠️ المخاطر (Dangers)")
st.markdown(st.session_state.analysis.get("dangers","—"))

st.subheader("➖ التقليل (Reduce)")
st.markdown(st.session_state.analysis.get("reduce","—"))

st.subheader("🔄 إعادة الاستخدام (Reuse)")
for idea in st.session_state.reuse_ideas:
    st.markdown(f"- {idea}")

st.subheader("♻️ التدوير (Recycle)")
st.markdown(st.session_state.analysis.get("recycle","—"))

# ─── 3) Generate & cache descriptive image prompts ─────────────────────────────
if "reuse_images" not in st.session_state:
    st.session_state.reuse_images = []
    # build a descriptive prompt for each idea
    describe_prompt = (
        "أنت مولد أوصاف للصور باستخدام الذكاء الاصطناعي. لكل فكرة من أفكار إعادة الاستخدام التالية،"
        " اكتب وصفًا وصفيًا تفصيليًا للمشهد المتوقع للصورة النهائية، متضمناً الألوان، المواد، الأبعاد الظاهرة،"
        " والإضاءة، دون استخدام صيغة الأمر. لا تستخدم عبارات مثل 'اصنع' أو 'اقم'. فقط وصف تفصيلي.\n\n"
        f"Ideas:\n{chr(10).join(st.session_state.reuse_ideas)}"
    )
    resp_desc = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate detailed AI image prompts."},
            {"role": "user", "content": describe_prompt}
        ]
    )
    desc_text = resp_desc.choices[0].message.content
    # split by lines and take first 3 descriptions
    descs = [d.strip() for d in desc_text.split("\n") if d.strip()][:3]
    # now generate & cache images
    for desc in descs:
        with st.spinner(f"جاري توليد صورة وصفية لـ: {desc}"):
            img_resp = client.images.generate(
                model="dall-e-2",
                prompt=desc,
                n=1,
                size="1024x1024",
                response_format="b64_json"
            )
            b64_img = img_resp.data[0].b64_json
            st.session_state.reuse_images.append(base64.b64decode(b64_img))
            time.sleep(0.1)

# Display reuse images
st.subheader("📸 صور إعادة الاستخدام")
cols = st.columns(3)
for col, img_data, caption in zip(cols, st.session_state.reuse_images, st.session_state.reuse_ideas):
    with col:
        st.image(io.BytesIO(img_data), caption=caption, use_container_width=True)

# ─── 4) Map recycling centers within radius ─────────────────────────────────────
st.subheader("🗺️ مراكز التدوير القريبة")
def haversine(lat1, lon1, lat2, lon2):
    R=6371000; φ1,φ2=math.radians(lat1),math.radians(lat2)
    Δφ,Δλ=math.radians(lat2-lat1),math.radians(lon2-lon1)
    return 2*R*math.atan2(math.sqrt(math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2),
                          math.sqrt(1-(math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2)))

# get center coords
lat=lon=None
try:
    data=requests.get("https://ipinfo.io/json",timeout=3).json()
    loc=data.get("loc",""); lat,lon=map(float,loc.split(",")) if loc else (None,None)
except: pass

if lat is None:
    city=st.text_input("أدخل اسم مدينتك للعثور على مراكز التدوير:")
    if city:
        geo=requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q":city,"format":"json","limit":1},
            headers={"User-Agent":"TadweerAI/1.0"},
            timeout=5
        ).json()
        if geo:
            lat,lon=float(geo[0]["lat"]),float(geo[0]["lon"])

if lat is None:
    st.error("تعذر تحديد موقعك. تأكد من السماح بتحديد الموقع أو كتابة اسم مدينتك.")
    st.stop()

places=requests.get(
    "https://nominatim.openstreetmap.org/search",
    params={"q":"recycling center","format":"json","lat":lat,"lon":lon,"limit":50},
    headers={"User-Agent":"TadweerAI/1.0"},timeout=5
).json()

nearby=[]
for p in places:
    plat,plon=float(p["lat"]),float(p["lon"])
    if haversine(lat,lon,plat,plon)<=radius_km*1000:
        nearby.append((plat,plon,p.get("display_name","")))

m=folium.Map(location=[lat,lon],zoom_start=12)
folium.Circle(location=[lat,lon],radius=radius_km*1000,color="blue",fill=False).add_to(m)
cluster=MarkerCluster().add_to(m)
for plat,plon,name in nearby:
    folium.Marker([plat,plon],popup=name).add_to(cluster)

st.write(f"تم العثور على {len(nearby)} مركز تدوير ضمن {radius_km} كلم.")
st_folium(m,width=800,height=500)
