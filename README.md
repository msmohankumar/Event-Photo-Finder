# 📸 Event Photo Finder

**Find your photos from an event using a selfie!**

Event Photo Finder helps guests find photos of themselves from an event. Upload a selfie, and the application uses facial recognition to search through event photos and identify images that may contain you.

## 🌐 Live Demo

👉 **[Open Event Photo Finder](https://event-photo-finder-8glf2hyh3m8wgul2vxpvsm.streamlit.app/)**

## ✨ Features

- 📷 **Selfie Upload** – Upload your selfie to search for your photos.
- 🤖 **Face Recognition** – Uses AI-based facial recognition to compare faces.
- 🔍 **Photo Matching** – Searches event photos for faces matching your selfie.
- 🖼️ **Matched Photo Results** – Displays photos identified as potential matches.
- 📥 **Download Photos** – Download matched photos.
- 🌐 **Web-Based Application** – Access the app directly through a browser.
- 📱 **Simple Interface** – Designed to be easy for event guests to use.

## 🚀 How to Use

1. Open the [Event Photo Finder](https://event-photo-finder-8glf2hyh3m8wgul2vxpvsm.streamlit.app/).
2. Upload a selfie with your face clearly visible.
3. Upload or select the event photo collection as required by the application.
4. Allow the application to process the images.
5. Review the matching photos.
6. Download the photos you want to keep.

## 🛠️ Technologies Used

- **Python** – Core programming language.
- **Streamlit** – Web application interface.
- **DeepFace** – Facial recognition framework.
- **TensorFlow** – Machine learning framework.
- **OpenCV** – Image processing.
- **NumPy** – Numerical processing.
- **Pillow** – Image handling.
- **QR Code** – QR code functionality.

## 🧠 How It Works

The application uses facial recognition to compare the face in an uploaded selfie against faces detected in event photographs.

1. The user uploads a selfie.
2. The application detects the face in the selfie.
3. Faces are detected in the event photos.
4. Facial representations are compared.
5. Potential matching photographs are displayed.
6. Users can view and download the results.

## 📦 Project Structure

```text
event-photo-finder/
│
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
└── .gitignore
```

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/event-photo-finder.git
cd event-photo-finder
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Deployment

This application is deployed using **Streamlit Community Cloud** and connected to a GitHub repository.

Live application:

https://event-photo-finder-8glf2hyh3m8wgul2vxpvsm.streamlit.app/

## 🔐 Privacy Notice

Facial recognition involves biometric information. Users should upload images only with appropriate consent and use the application responsibly.

This prototype is intended for demonstration and experimentation. Its data handling, storage, security and privacy practices should be reviewed before use at large events or in production.

## 🔮 Future Improvements

- Shared event galleries.
- QR code access for event guests.
- Automatic photo synchronization.
- Faster face-matching algorithms.
- Multiple-person photo matching.
- Cloud-based photo storage.
- Improved privacy and consent management.
- Download matched photos in bulk.

---

**Built with Python, AI and Streamlit.** 📸
