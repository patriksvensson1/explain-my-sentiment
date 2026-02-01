# explain-my-sentiment
Explain My Sentiment is a web app that analyzes the sentiment of text and shows which words contributed most to the prediction. It uses a RoBERTa sentiment model and SHAP-based token attribution, wrapped behind a simple upload-and-click workflow.

## Tech Stack  
– **Backend**: Python, FastAPI, Hugging Face Transformers (RoBERTa), PyTorch, SHAP  
– **Frontend**: TypeScript, React, Tailwind CSS  
– **Deployment**: Docker

## How to Run
### 1. Clone the repo  
```bash
git clone https://github.com/patriksvensson1/explain-my-sentiment.git
cd explain-my-sentiment
```

### 2. Run with docker
With docker running, navigate to the repo folder and run 
```bash
docker compose up --build
```
Frontend will then run on localhost:3000  
and backend: localhost:8000

### First run note:
The first startup may take a while because the RoBERTa model needs to be downloaded and then loaded into memory. Subsequent starts are faster due to a persisted Hugging Face cache volume (the model files do not need to be re-downloaded).
