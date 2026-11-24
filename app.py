import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from io import BytesIO
import hashlib
import zipfile
import qrcode
from deepface import DeepFace

st.set_page_config(page_title="Event Photo Finder", page_icon="📸", layout="wide")
st.title("📸 Event Photo Finder")
st.caption("Upload event photos, take or upload a selfie, and find photos containing a similar face.")

st.sidebar.header("Event Gallery")
st.sidebar.write("Images and cached face embeddings are held in this Streamlit session.")
st.sidebar.warning("Use only photos and selfies you have permission to process. Face matching can make mistakes.")

# Keep a per-session cache of embeddings so a rerun or repeat search does not
# recompute face representations for the same uploaded photo.
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

MAX_IMAGE_SIDE = 1280
JPEG_QUALITY = 85

photos = st.file_uploader("Upload event photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="gallery")
event_url = st.text_input("Public app URL (optional, for QR code)", placeholder="https://your-app-name.streamlit.app")

if event_url.strip():
    st.sidebar.image(qrcode.make(event_url.strip()), caption="Scan to open this event app")


def normalized_image(data, max_side=MAX_IMAGE_SIDE):
    """Load, orient and downscale an image to reduce inference memory/time."""
    with Image.open(BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return im, buf.getvalue()


def cache_key(data):
    return hashlib.sha256(data).hexdigest()


def face_embeddings(data, is_selfie=False):
    """Return all detected face embeddings from a resized image."""
    key = cache_key(data)
    cached = st.session_state.embedding_cache.get(key)
    if cached is not None:
        return cached

    image, _ = normalized_image(data)
    arr = np.asarray(image)
    reps = DeepFace.represent(
        img_path=arr,
        model_name="Facenet512",
        detector_backend="opencv",
        enforce_detection=True,
        align=True,
    )
    if not reps:
        raise ValueError("No face detected")

    # Return all faces, not only the first one, so group/event photos can match.
    output = []
    for rep in reps:
        emb = np.asarray(rep["embedding"], dtype=np.float32)
        area = 0
        facial_area = rep.get("facial_area", {})
        if facial_area:
            area = int(facial_area.get("w", 0)) * int(facial_area.get("h", 0))
        output.append((emb, area))

    if is_selfie:
        # For the query selfie, use the largest detected face to avoid selecting
        # a small background face in a selfie/group image.
        output.sort(key=lambda item: item[1], reverse=True)
        output = output[:1]

    st.session_state.embedding_cache[key] = output
    return output


def cosine_distance(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)

