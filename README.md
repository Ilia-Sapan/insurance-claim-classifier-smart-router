# Insurance Claim Classifier & Smart Router

German-language NLP application for insurance input management in an insurance company. The app classifies incoming customer letters, detects sentiment, assigns routing priority, drafts a formal answer and estimates operational ROI for AI-assisted processing.

## Features

- German zero-shot classification for: Schadensmeldung, Beschwerde, Vertragsaenderung, Kuendigung
- German sentiment analysis with `oliverguhr/german-sentiment-bert`
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

- Sentiment: `oliverguhr/german-sentiment-bert`
- Zero-shot classification: `Sahajpreet/german-zero-shot`
- Zero-shot fallback: `Sahajtomar/German_Zeroshot`
- Optional response refinement: `dbmdz/german-gpt2`

## Run Locally

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
```

Open `http://localhost:8501`.

The first analysis can take longer because Hugging Face models are downloaded lazily on first use.

## Docker

```powershell
docker build -t insurance-claim-router .
docker run -p 8501:8501 insurance-claim-router
```

## Business Context

Input management teams in insurance companies receive large volumes of unstructured customer correspondence. This prototype shows how NLP can reduce manual triage effort by classifying letters, detecting urgent or negative cases and routing them to the right operational team before a human specialist reviews the case.

## Portfolio Angle

This project demonstrates an end-to-end data science solution: NLP model integration, business rules, user interface, ROI analytics and containerized deployment.