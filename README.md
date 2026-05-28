# Insurance Claim Classifier & Smart Router

German-language NLP application for insurance input management in an insurance company. The app classifies incoming customer letters, detects sentiment, assigns routing priority, drafts a formal answer and estimates operational ROI for AI-assisted processing.

## Features

- German claim letter classification for: Schadensmeldung, Beschwerde, Vertragsaenderung, Kuendigung
- Stable portfolio demo mode that runs without loading large transformer models
- Optional Hugging Face model mode for German sentiment and zero-shot classification
- Smart priority scoring, SLA assignment and department routing
- Formal German response draft in Sie-Form
- Streamlit UI with a letter simulator and business metrics dashboard
- Plotly ROI chart for 100,000 incoming letters per month
- Dockerfile for containerized deployment

## Tech Stack

- Python 3.12
- Streamlit
- Hugging Face Transformers
- PyTorch
- Plotly
- Pandas

## Models

The app starts in a lightweight portfolio mode for reliability on local laptops and free cloud tiers. Transformer models can be enabled from the UI toggle or with the environment variable `AKTIVIERE_HF_MODELLE=true`.

- Sentiment: `oliverguhr/german-sentiment-bert`
- Zero-shot classification: `Sahajpreet/german-zero-shot`
- Zero-shot fallback: `Sahajtomar/German_Zeroshot`
- Optional response refinement: `dbmdz/german-gpt2`

## Run Locally

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
```

Open `http://localhost:8501` or `http://127.0.0.1:8501`.

Keep `Hugging-Face-Modelle verwenden` switched off for a stable public demo. Enable it only on a machine with enough RAM.

## Public Demo Deployment

This repository is ready for Streamlit Community Cloud:

1. Open `https://share.streamlit.io`.
2. Connect the GitHub repository `Ilia-Sapan/insurance-claim-classifier-smart-router`.
3. Set the main file path to `app.py`.
4. Deploy the app.

After deployment, Streamlit gives a public URL that can be added to a CV, LinkedIn profile or GitHub README.

## Docker

```powershell
docker build -t insurance-claim-router .
docker run -p 8501:8501 insurance-claim-router
```

## Business Context

Input management teams in insurance companies receive large volumes of unstructured customer correspondence. This prototype shows how NLP can reduce manual triage effort by classifying letters, detecting urgent or negative cases and routing them to the right operational team before a human specialist reviews the case.

## Portfolio Angle

This project demonstrates an end-to-end data science solution: NLP model integration, business rules, user interface, ROI analytics and containerized deployment.