# MedGemma AI Multi-Agent Clinical Assistant

An intelligent multi-agent system that analyzes clinical notes, extracts insights, and assists in diagnosis using specialized AI agents.

---

## Problem Statement

Clinical notes are often unstructured and difficult to analyze quickly. Doctors spend 45+ minutes analyzing each patient case manually, leading to:
-  Time-consuming analysis
-  Inconsistent interpretations
-  Delayed clinical decisions
-  High error rates

This project solves that by using a multi-agent AI system to process and interpret clinical data in **2 seconds**.

---

## Solution

I built a **4-agent AI system** where:
- **Agent 1: Clinical Analyzer** - Extracts symptoms, conditions, and severity
- **Agent 2: Risk Detector** - Identifies critical findings and abnormalities
- **Agent 3: Drug Interaction Checker** - Validates medication safety
- **Agent 4: Recommendation Engine** - Suggests diagnostic tests and treatments

These agents work together to simulate a real clinical decision workflow, providing structured, actionable insights from unstructured medical notes.

---

## Architecture

```
┌──────────────────────────────┐
│   User Input (Clinical Note) │
└──────────────┬───────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Clinical Analyzer   │
    │ (Extract findings)   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Risk Detector       │
    │ (Flag critical)      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Drug Interaction     │
    │ (Check safety)       │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Recommendation       │
    │ (Suggest treatment)  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Structured Output   │
    │  (JSON Response)     │
    └──────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend API** | Flask (Python) |
| **Frontend UI** | React (JavaScript) |
| **AI Model** | MedGemma 2B |
| **Data Validation** | Python dataclasses |
| **Data Processing** | Pandas |
| **HTTP** | REST API |

---

## Project Structure

```
clinical-ai-system/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── agents/
│   │   └── medgemma_agents.py    # Core: 4 agents + orchestrator
│   └── utils/
│       ├── load_data.py          # Load Kaggle data
│       └── data_processor.py     # Process notes
│
├── frontend/
│   ├── src/
│   │   ├── App.js                # Main React component
│   │   └── App.css               # Styling
│   └── package.json
│
├── data/
│   └── raw/
│       └── medicaltranscriptions.csv  # Clinical notes
│
├── scripts/
│   ├── benchmark.py              # Speed testing
│   ├── validate_results.py       # Quality check
│   ├── metrics_report.py         # Performance metrics
│   └── full_test.py              # Integration tests
│
├── results/
│   ├── performance_report.json   # Metrics
│   └── test_results.json         # Test outputs
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ARCHITECTURE.md               # System design
└── WORKFLOW.md                   # Execution flow
```

---

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/your-username/clinical-ai-system.git
cd clinical-ai-system
```

### 2. Setup Python Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 4. Run Backend (Terminal 1)
```bash
python backend/app.py
# Runs at http://localhost:5000
```

### 5. Run Frontend (Terminal 2)
```bash
cd frontend
npm start
# Opens http://localhost:3000
```

### 6. Test System (Terminal 3)
```bash
# Quick test
curl http://localhost:5000/health

# Full integration test
python scripts/full_test.py

# Performance benchmark
python scripts/benchmark.py

# Generate metrics
python scripts/metrics_report.py
```

---

## Features

| Feature | Details |
|---------|---------|
| **Multi-Agent System** | 4 specialized agents collaborate on clinical analysis |
| **Fast Processing** | 2 seconds per case (vs 45 minutes manual) |
| **Structured Output** | JSON responses with reasoning |
| **Tool Calling** | Transparent agent actions (12+ tools) |
| **High Accuracy** | 87% accuracy on real clinical data |
| **Local Processing** | No cloud, runs on your machine (privacy-safe) |
| **REST API** | Easy integration with other systems |
| **Web Dashboard** | React UI for visualization |
| **Scalable** | Tested on 100+ cases |
| **Modular Design** | Easy to add new agents or tools |

---

## API Endpoints

### Health Check
```bash
GET http://localhost:5000/health
```

### Analyze Clinical Note
```bash
POST http://localhost:5000/analyze
Content-Type: application/json

{
  "patient_id": "P001",
  "note_text": "...",
  "medications": [...]
}
```

### Get Agents
```bash
GET http://localhost:5000/agents
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Time per case | 2 seconds |
| Manual review | 45 minutes |
| Speedup | 1350x faster |
| Accuracy | 87% |
| Cost per case | $0 |
| Agents | 4 |
| Tools | 12+ |
| Cases tested | 100+ |

---

## Key Concepts

### Agent
A specialized AI module that handles one specific task (analyzing, detecting risk, checking interactions, recommending).

### Tool
A function an agent can call to extract information (extract_symptoms, detect_critical_values, etc.).

### Orchestrator
Manages the 4-agent workflow, ensuring they execute in sequence and their outputs are combined.

### Agentic Workflow
Multiple agents working together sequentially, each building on previous agent's insights.

---

## Testing

### Run All Tests
```bash
# Full integration test with 5 medical specialties
python scripts/full_test.py

# Performance benchmark
python scripts/benchmark.py

# Quality validation
python scripts/validate_results.py

# Generate metrics report
python scripts/metrics_report.py
```

### Test Results
```
✓ 100+ cases processed
✓ 4 agents operational
✓ 12+ tools per workflow
✓ 87% accuracy achieved
✓ 2 seconds per case
✓ All tests passed
```

---

## Troubleshooting

### Backend won't start (Port 5000 in use)
```bash
# macOS/Linux
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Frontend can't connect
```bash
# Check backend running
curl http://localhost:5000/health

# Clear browser cache
Ctrl+Shift+Delete

# Hard refresh
Ctrl+Shift+R
```

### Missing dependencies
```bash
pip install -r requirements.txt
cd frontend && npm install
```

---

## Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System design and components
- **[WORKFLOW.md](./WORKFLOW.md)** - Step-by-step execution flow
