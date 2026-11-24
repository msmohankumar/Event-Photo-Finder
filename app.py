import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import hashlib
import zipfile
import qrcode
from deepface import DeepFace
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Event Photo Finder", page_icon="📸", layout="wide")
st.title("📸 Event Photo Finder")
st.caption("Upload event photos, scan the event QR, take a selfie, and find photos containing a similar face.")

st.sidebar.header("Event Gallery")
st.sidebar.write("This prototype keeps uploaded images and face embeddings in the current Streamlit session.")
st.sidebar.warning("Use only photos and selfies you have permission to process. Face matching can make mistakes.")

photos = st.file_uploader("Upload event photos", type=["jpg","jpeg","png"], accept_multiple_files=True, key="gallery")
event_url = st.text_input("Public app URL (optional, for QR code)", placeholder="https://your-app-name.streamlit.app")

if event_url.strip():
    qr_img=qrcode.make(event_url.strip())
    st.sidebar.image(qr_img, caption="Scan to open this event app")

def pil_to_bytes(pil_img):
    out=BytesIO()
    pil_img.save(out, format="JPEG", quality=92)
    return out.getvalue()

def load_img(data):
    img=Image.open(BytesIO(data)).convert("RGB")
    return img

def embedding_from_bytes(data):
    # DeepFace returns a facial embedding. enforce_detection=True avoids embedding
    # non-face content as if it were a face.
    arr=np.array(load_img(data))
    result=DeepFace.represent(
        img_path=arr, model_name="Facenet512", detector_backend="opencv",
        enforce_detection=True, align=True
    )
    if not result:
        raise ValueError("No face detected")
    # For a single-face selfie, use the largest detected face if several are found.
    result=sorted(result, key=lambda r: r.get("face_confidence",1), reverse=True)
    return np.array(result[0]["embedding"], dtype=np.float32)

def cosine_distance(a,b):
    denom=(np.linalg.norm(a)*np.linalg.norm(b))
    if denom==0: return 1.0
    return float(1-(np.dot(a,b)/denom))

if photos:
    st.write(f"**{len(photos)} photo(s) in the current gallery**")
    with st.expander("Preview uploaded photos", expanded=False):
        cols=st.columns(4)
        for i,p in enumerate(photos):
            cols[i%4].image(p, caption=p.name, use_container_width=True)

st.divider()
st.header("Find your photos")
st.write("Take a selfie or upload a selfie. The app compares it against the event photos you uploaded above.")

selfie_file=st.camera_input("Take a selfie")
selfie_upload=st.file_uploader("Or upload a selfie image", type=["jpg","jpeg","png"], key="selfie_upload")
selfie_data=selfie_file.getvalue() if selfie_file else (selfie_upload.getvalue() if selfie_upload else None)

threshold=st.slider("Matching sensitivity (cosine distance)", min_value=0.20, max_value=0.70, value=0.40, step=0.01, help="Lower values are stricter. This is a prototype setting; tune using your own sample images.")
run=st.button("🔎 Find my photos", type="primary", disabled=not (photos and selfie_data))

if run:
    with st.spinner("Analyzing selfie..."):
        try:
            query_emb=embedding_from_bytes(selfie_data)
        except Exception as e:
            st.error(f"Couldn't detect a face in the selfie. Try a clearer, front-facing photo. Details: {e}")
            st.stop()

    matches=[]
    progress=st.progress(0)
    status=st.empty()
    for i,p in enumerate(photos):
        status.write(f"Checking {i+1}/{len(photos)}: {p.name}")
        try:
            emb=embedding_from_bytes(p.getvalue())
            dist=cosine_distance(query_emb, emb)
            if dist<=threshold:
                matches.append((p.name,p.getvalue(),dist))
        except Exception:
            # Skip images with no face or processing issues.
            pass
        progress.progress((i+1)/len(photos))
    status.empty()
    progress.empty()
    matches.sort(key=lambda x:x[2])

    st.subheader(f"Found {len(matches)} matching photo(s)")
    if not matches:
        st.info("No matches at this threshold. Try a higher matching sensitivity or a clearer selfie/photo.")
    else:
        cols=st.columns(3)
        for i,(name,data,dist) in enumerate(matches):
            with cols[i%3]:
                st.image(data, caption=f"{name} • distance {dist:.3f}", use_container_width=True)
                st.download_button("⬇️ Download photo", data=data, file_name=name, mime="image/jpeg", key=f"dl_{i}")
        if len(matches)>1:
            zip_buf=BytesIO()
            with zipfile.ZipFile(zip_buf,'w',zipfile.ZIP_DEFLATED) as z:
                for name,data,dist in matches:
                    z.writestr(name,data)
            st.download_button("⬇️ Download all matched photos (ZIP)", data=zip_buf.getvalue(), file_name="my_event_photos.zip", mime="application/zip")

st.divider()
st.caption("Prototype: photos are uploaded to this app session. They are not automatically synced from a photographer's camera or Drive. Face matching is probabilistic, not guaranteed.")
