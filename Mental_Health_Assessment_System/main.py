from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

stress_model     = joblib.load(os.path.join(BASE, "stress_model.pkl"))
anxiety_model    = joblib.load(os.path.join(BASE, "anxiety_model.pkl"))
depression_model = joblib.load(os.path.join(BASE, "depression_model.pkl"))
le_stress        = joblib.load(os.path.join(BASE, "le_stress.pkl"))
le_anxiety       = joblib.load(os.path.join(BASE, "le_anxiety.pkl"))
le_depression    = joblib.load(os.path.join(BASE, "le_depression.pkl"))

# ─────────────────────────────────────────────
# COLUMN ORDER (must match training order)
# ─────────────────────────────────────────────
STRESS_COLS     = ['Q11 (S)', 'Q1 (S)', 'Q27 (S)', 'Q29 (S)', 'Q8 (S)', 'Q39 (S)', 'Q12 (S)', 'Q33 (S)']
ANXIETY_COLS    = ['Q28 (A)', 'Q36 (A)', 'Q9 (A)', 'Q7 (A)', 'Q20 (A)', 'Q40 (A)', 'Q41 (A)', 'Q4 (A)']
DEPRESSION_COLS = ['Q13 (D)', 'Q34 (D)', 'Q16 (D)', 'Q17 (D)', 'Q21 (D)', 'Q10 (D)', 'Q26 (D)', 'Q24 (D)']

# ─────────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────────
class AssessmentInput(BaseModel):
    # Stress (8 questions)
    Q11_S: int
    Q1_S:  int
    Q27_S: int
    Q29_S: int
    Q8_S:  int
    Q39_S: int
    Q12_S: int
    Q33_S: int
    # Anxiety (8 questions)
    Q28_A: int
    Q36_A: int
    Q9_A:  int
    Q7_A:  int
    Q20_A: int
    Q40_A: int
    Q41_A: int
    Q4_A:  int
    # Depression (8 questions)
    Q13_D: int
    Q34_D: int
    Q16_D: int
    Q17_D: int
    Q21_D: int
    Q10_D: int
    Q26_D: int
    Q24_D: int

# ─────────────────────────────────────────────
# PREDICT ENDPOINT
# ─────────────────────────────────────────────
@app.post("/predict")
def predict(data: AssessmentInput):
    s_input = np.array([[data.Q11_S, data.Q1_S, data.Q27_S, data.Q29_S,
                         data.Q8_S,  data.Q39_S, data.Q12_S, data.Q33_S]])

    a_input = np.array([[data.Q28_A, data.Q36_A, data.Q9_A,  data.Q7_A,
                         data.Q20_A, data.Q40_A, data.Q41_A, data.Q4_A]])

    d_input = np.array([[data.Q13_D, data.Q34_D, data.Q16_D, data.Q17_D,
                         data.Q21_D, data.Q10_D, data.Q26_D, data.Q24_D]])

    s_pred  = stress_model.predict(s_input)[0]
    s_prob  = round(float(stress_model.predict_proba(s_input)[0].max()) * 100, 1)
    s_label = le_stress.inverse_transform([s_pred])[0]

    a_pred  = anxiety_model.predict(a_input)[0]
    a_prob  = round(float(anxiety_model.predict_proba(a_input)[0].max()) * 100, 1)
    a_label = le_anxiety.inverse_transform([a_pred])[0]

    d_pred  = depression_model.predict(d_input)[0]
    d_prob  = round(float(depression_model.predict_proba(d_input)[0].max()) * 100, 1)
    d_label = le_depression.inverse_transform([d_pred])[0]

    return {
        "stress":     {"label": s_label, "confidence": s_prob},
        "anxiety":    {"label": a_label, "confidence": a_prob},
        "depression": {"label": d_label, "confidence": d_prob},
    }

# ─────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=BASE), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(BASE, "index.html"))
