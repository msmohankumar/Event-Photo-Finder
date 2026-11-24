# Event Photo Finder — Streamlit Prototype

A simple prototype: upload event photographs, optionally generate a QR code for the deployed app, take/upload a selfie, compare faces, and download matching photos or a ZIP.

## Important limitations
- This prototype's gallery is populated by photo uploads in the app. It does not automatically read a phone's gallery, local camera folder, or Google Drive.
- On mobile, use the gallery upload picker to select multiple event images. Mobile camera input is used for the selfie.
- Streamlit Cloud may have deployment/runtime constraints because DeepFace and its ML dependencies are large. Try deployment; if build limits are encountered, use a separate inference service or lighter model.
- Face similarity is probabilistic and can return false matches/misses. Use with consent; do not use for identity verification or consequential decisions.
- Images are handled by the Streamlit app runtime and may be retained in runtime memory/cache depending on platform. Do not upload sensitive images to a public deployment.

## Run locally
1. Install Python 3.10 or 3.11 (recommended for compatibility).
2. Download/unzip the project.
3. Open a terminal in the project folder.
4. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
5. Activate it on Windows:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
6. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
7. Start:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this README.
2. Visit https://share.streamlit.io/
3. Sign in with GitHub and choose **New app**.
4. Select the repository, branch and `app.py`.
5. Deploy and wait for dependency installation.
6. Open the deployed app on a phone. Upload event photos and take/upload a selfie to test matching.
7. Enter the public app URL in the QR field to display a QR code for guests.

## Test workflow
- Upload a small test set (5–20 photos) first.
- Ensure faces are clearly visible, not tiny, blurry, or heavily occluded.
- Test with several selfies of the same person and different people.
- Use the sensitivity slider to balance false matches and missed matches.
