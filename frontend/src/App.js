import React, { useEffect, useMemo, useState } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

const SEVERITY_COLORS = {
  Critical: '#B91C1C',
  High: '#DC2626',
  Medium: '#F59E0B',
  Low: '#10B981',
  Unknown: '#6B7280',
};

const Dashboard = () => {
  const [disease, setDisease] = useState('');
  const [symptomsText, setSymptomsText] = useState('');
  const [medicationsText, setMedicationsText] = useState('');
  const [noteText, setNoteText] = useState('');

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis');
  const [diseases, setDiseases] = useState([]);
  const [suggestions, setSuggestions] = useState([]);

  // Load disease list once for the autocomplete dropdown.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/diseases`);
        const data = await res.json();
        if (!cancelled && data.success) {
          setDiseases(data.diseases || []);
        }
      } catch (err) {
        // Non-fatal – suggestions will simply be empty.
        console.warn('Could not load disease list:', err.message);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Live fuzzy search through the loaded disease list.
  useEffect(() => {
    const q = disease.trim().toLowerCase();
    if (!q) {
      setSuggestions([]);
      return;
    }
    const matches = diseases
      .filter(d => d.disease.toLowerCase().includes(q))
      .slice(0, 8);
    setSuggestions(matches);
  }, [disease, diseases]);

  const symptoms = useMemo(
    () => symptomsText.split(/[,\n]/).map(s => s.trim()).filter(Boolean),
    [symptomsText]
  );
  const medications = useMemo(
    () => medicationsText.split(/[,\n]/).map(s => s.trim()).filter(Boolean),
    [medicationsText]
  );

  const handleAnalyze = async () => {
    if (!disease.trim() && !symptoms.length && !noteText.trim()) {
      setError('Provide at least a disease name, a symptom, or a free-text note.');
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: 'WEB-001',
          disease: disease.trim(),
          symptoms,
          medications,
          note_text: noteText.trim(),
        }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Unknown error');
      setResults(data.data);
      setActiveTab('analysis');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setResults(null);
    setError(null);
    setActiveTab('analysis');
  };

  const handleExample = (preset) => {
    if (preset === 'chest') {
      setDisease('Coronary Artery Disease');
      setSymptomsText('chest pain, shortness of breath, sweating, palpitations');
      setMedicationsText('Lisinopril 10mg, Aspirin 81mg, Atorvastatin 80mg');
      setNoteText(
        '58-year-old male with crushing chest pain radiating to the left arm, ' +
        'shortness of breath and diaphoresis for the past hour. History of ' +
        'hypertension and hyperlipidemia.'
      );
    } else if (preset === 'diabetes') {
      setDisease('Type 2 Diabetes');
      setSymptomsText('fatigue, frequent urination, excessive thirst, blurred vision');
      setMedicationsText('Metformin 1000mg');
      setNoteText('Patient complains of polyuria, polydipsia and weight changes for 2 months.');
    } else if (preset === 'symptoms_only') {
      setDisease('');
      setSymptomsText('fever, cough, shortness of breath, fatigue');
      setMedicationsText('');
      setNoteText('');
    }
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <span style={styles.badge}>AI-Powered · ML-Driven</span>
          <h1 style={styles.title}>Clinical Sense</h1>
          <p style={styles.subtitle}>
            Multi-agent disease → risk → medication recommender (Random Forest + Gradient Boosting + KMeans)
          </p>
        </div>
      </header>

      <main style={styles.content}>
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Patient Input</h2>
          <p style={styles.sectionSubtitle}>
            Provide a disease, a list of symptoms, or a free-text note. The system loads its knowledge from
            <code style={styles.code}> data/disease_dataset.csv</code> and infers the rest with trained models.
          </p>

          <div style={styles.examplesRow}>
            <span style={styles.examplesLabel}>Try:</span>
            <button onClick={() => handleExample('chest')} style={styles.exampleButton}>
              Chest pain preset
            </button>
            <button onClick={() => handleExample('diabetes')} style={styles.exampleButton}>
              Diabetes preset
            </button>
            <button onClick={() => handleExample('symptoms_only')} style={styles.exampleButton}>
              Symptoms only
            </button>
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Disease (optional, exact name from dataset gives best results)</label>
            <input
              type="text"
              value={disease}
              onChange={(e) => setDisease(e.target.value)}
              style={styles.input}
              placeholder="e.g. Hypertension, Type 2 Diabetes, Pneumonia..."
              autoComplete="off"
            />
            {suggestions.length > 0 && (
              <div style={styles.suggestions}>
                {suggestions.map(s => (
                  <button
                    key={s.disease}
                    style={styles.suggestionItem}
                    onClick={() => {
                      setDisease(s.disease);
                      setSuggestions([]);
                    }}
                  >
                    <span>{s.disease}</span>
                    <span style={styles.suggestionMeta}>
                      {s.specialty} · risk {s.risk_score} · {s.severity}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Symptoms (comma- or newline-separated)</label>
            <textarea
              value={symptomsText}
              onChange={(e) => setSymptomsText(e.target.value)}
              style={styles.textarea}
              placeholder="e.g. fever, cough, shortness of breath, fatigue"
              rows={3}
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Current Medications (comma- or newline-separated)</label>
            <textarea
              value={medicationsText}
              onChange={(e) => setMedicationsText(e.target.value)}
              style={styles.textarea}
              placeholder="e.g. Lisinopril 10mg, Aspirin 81mg"
              rows={2}
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Free-text clinical note (optional, will be parsed for symptoms)</label>
            <textarea
              value={noteText}
              onChange={(e) => setNoteText(e.target.value)}
              style={{ ...styles.textarea, minHeight: '110px' }}
              placeholder="Free-form description of the patient presentation..."
            />
          </div>

          <div style={styles.actionBar}>
            <button onClick={handleAnalyze} disabled={loading} style={styles.button}>
              {loading ? 'Analyzing…' : 'Run Multi-Agent Analysis'}
            </button>
            {results && (
              <button onClick={handleClear} style={styles.secondaryButton}>
                Clear
              </button>
            )}
          </div>

          {error && (
            <div style={styles.errorBox}>
              <strong>Error:</strong> {error}
            </div>
          )}
        </section>

        {results && <ResultsView results={results} activeTab={activeTab} setActiveTab={setActiveTab} />}
      </main>

      <footer style={styles.footer}>
        <p>Trained from <code>data/disease_dataset.csv</code> · Backend: Flask + scikit-learn</p>
      </footer>
    </div>
  );
};

const ResultsView = ({ results, activeTab, setActiveTab }) => {
  const analysis = results.workflow_stage_1_analysis || {};
  const risks = results.workflow_stage_2_risks || {};
  const interactions = results.workflow_stage_3_interactions || {};
  const recs = results.workflow_stage_4_recommendations || {};

  const primary = analysis.primary_disease || {};
  const riskLevel = risks.risk_level || 'Unknown';
  const severityColor = SEVERITY_COLORS[riskLevel] || SEVERITY_COLORS.Unknown;

  return (
    <section style={styles.section}>
      <h2 style={styles.sectionTitle}>Analysis Results</h2>

      <div style={styles.summaryHeader}>
        <SummaryCard
          label="Predicted disease"
          value={primary.disease || '—'}
          sub={primary.specialty || ''}
          color="#10B981"
        />
        <SummaryCard
          label="Risk score"
          value={risks.risk_score != null ? `${risks.risk_score} / 100` : '—'}
          sub={riskLevel}
          color={severityColor}
        />
        <SummaryCard
          label="Urgency"
          value={risks.urgency || '—'}
          sub={risks.severity || ''}
          color={severityColor}
        />
        <SummaryCard
          label="Confidence"
          value={`${Math.round((risks.confidence || 0) * 100)}%`}
          sub="risk detector"
          color="#3B82F6"
        />
      </div>

      <div style={styles.tabs}>
        {[
          { id: 'analysis',   label: 'Clinical Analysis' },
          { id: 'risk',       label: 'Risk Assessment' },
          { id: 'drugs',      label: 'Drug Safety' },
          { id: 'recommend',  label: 'Medication & Tests' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              ...styles.tab,
              borderBottomColor: activeTab === tab.id ? '#10B981' : 'transparent',
              color: activeTab === tab.id ? '#10B981' : '#6B7280',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div style={styles.tabContent}>
        {activeTab === 'analysis' && (
          <Card title="Clinical Analysis">
            <Row label="Symptoms identified" value={
              (analysis.symptoms_identified || []).join(', ') || '—'
            } />
            <Row label="Primary disease" value={
              primary.disease
                ? `${primary.disease} (${primary.specialty})`
                : '—'
            } />
            {primary.description && (
              <Row label="Description" value={primary.description} />
            )}
            <Row label="Model" value={analysis.model_used || '—'} />
            {analysis.alternative_diseases && analysis.alternative_diseases.length > 0 && (
              <div>
                <strong>Alternative hypotheses:</strong>
                <ul style={styles.list}>
                  {analysis.alternative_diseases.map(d => (
                    <li key={d.disease}>
                      {d.disease} — {(d.probability * 100).toFixed(1)}% (risk {d.risk_score})
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysis.cluster_related_diseases && analysis.cluster_related_diseases.length > 0 && (
              <div>
                <strong>Cluster-related diseases (KMeans):</strong>
                <p>{analysis.cluster_related_diseases.join(', ')}</p>
              </div>
            )}
            {analysis.disease_lookup && analysis.disease_lookup.found && (
              <Row label="Lookup" value={`${analysis.disease_lookup.disease} found in dataset (severity ${analysis.disease_lookup.severity})`} />
            )}
            {analysis.disease_lookup && analysis.disease_lookup.found === false && (
              <Row label="Lookup" value={`"${analysis.disease_lookup.query}" not found in dataset — using symptom model only`} />
            )}
            {analysis.patient_age && <Row label="Patient age" value={analysis.patient_age} />}
            {analysis.patient_gender && <Row label="Patient gender" value={analysis.patient_gender} />}
          </Card>
        )}

        {activeTab === 'risk' && (
          <Card title="Risk Assessment">
            <Row label="Risk score" value={`${risks.risk_score || 0} / 100`} />
            <Row label="Risk level" value={risks.risk_level} />
            <Row label="Urgency" value={risks.urgency} />
            <Row label="Severity (dataset)" value={risks.severity || '—'} />
            <Row label="Rationale" value={risks.rationale || '—'} />
            <Row label="Confidence" value={`${Math.round((risks.confidence || 0) * 100)}%`} />
            <div>
              <strong>Critical flags:</strong>
              <ul style={styles.list}>
                {(risks.critical_flags || []).map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          </Card>
        )}

        {activeTab === 'drugs' && (
          <Card title="Drug Safety">
            <Row label="Status" value={interactions.overall_status || '—'} />
            <Row label="Medications reviewed" value={(interactions.medications_reviewed || []).join(', ') || '—'} />
            <Row label="Total interactions" value={interactions.total_interactions ?? 0} />
            {(interactions.interactions_found || []).length > 0 ? (
              <ul style={styles.list}>
                {interactions.interactions_found.map((i, idx) => (
                  <li key={idx}>
                    <strong>{i.severity}:</strong> {i.drug_1} + {i.drug_2} → {i.effect}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No major interactions detected.</p>
            )}
          </Card>
        )}

        {activeTab === 'recommend' && (
          <Card title="Medication & Test Recommendations">
            <Row label="Primary disease" value={recs.primary_disease || '—'} />
            <Row label="Specialty" value={recs.specialty || '—'} />
            {recs.description && <Row label="Description" value={recs.description} />}
            <div>
              <strong>Suggested medications (from dataset):</strong>
              <ul style={styles.list}>
                {(recs.suggested_medications || []).map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            </div>
            <div>
              <strong>Suggested tests (from dataset):</strong>
              <ul style={styles.list}>
                {(recs.suggested_tests || []).map((t, i) => <li key={i}>{t}</li>)}
              </ul>
            </div>
            <Row label="Outcome prediction" value={recs.outcome_prediction || '—'} />
            <div>
              <strong>Monitoring plan:</strong>
              <ul style={styles.list}>
                {(recs.monitoring_plan || []).map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            </div>
            <Row label="Follow-up" value={recs.follow_up || '—'} />
          </Card>
        )}
      </div>

      <details style={styles.rawBlock}>
        <summary>Show raw JSON response</summary>
        <pre style={styles.pre}>{JSON.stringify(results, null, 2)}</pre>
      </details>
    </section>
  );
};

const Card = ({ title, children }) => (
  <div style={styles.card}>
    <h3 style={styles.cardTitle}>{title}</h3>
    {children}
  </div>
);

const Row = ({ label, value }) => (
  <div style={styles.row}>
    <span style={styles.rowLabel}>{label}</span>
    <span style={styles.rowValue}>{value}</span>
  </div>
);

const SummaryCard = ({ label, value, sub, color }) => (
  <div style={{ ...styles.summaryCard, borderTop: `3px solid ${color}` }}>
    <div style={{ ...styles.summaryValue, color }}>{value}</div>
    <div style={styles.summaryLabel}>{label}</div>
    {sub && <div style={styles.summarySub}>{sub}</div>}
  </div>
);

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%)',
    fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, sans-serif",
    color: '#1F2937',
  },
  header: {
    background: 'linear-gradient(135deg, #0F766E 0%, #10B981 100%)',
    color: 'white',
    padding: '56px 20px',
    textAlign: 'center',
  },
  headerContent: { maxWidth: '960px', margin: '0 auto' },
  badge: {
    display: 'inline-block',
    background: 'rgba(255,255,255,0.18)',
    color: 'white',
    padding: '6px 14px',
    borderRadius: '20px',
    fontSize: '12px',
    fontWeight: 600,
    marginBottom: '16px',
  },
  title: {
    margin: '0 0 12px 0',
    fontSize: '44px',
    fontWeight: 700,
    letterSpacing: '-0.5px',
  },
  subtitle: { margin: 0, fontSize: '15px', opacity: 0.9 },
  content: { maxWidth: '1100px', margin: '0 auto', padding: '40px 20px' },
  section: {
    background: 'white',
    borderRadius: '16px',
    padding: '32px',
    marginBottom: '24px',
    boxShadow: '0 4px 20px rgba(0,0,0,0.06)',
  },
  sectionTitle: { marginTop: 0, fontSize: '24px', fontWeight: 700 },
  sectionSubtitle: { marginTop: 0, color: '#6B7280', fontSize: '14px' },
  code: { background: '#F3F4F6', padding: '2px 6px', borderRadius: '4px', fontSize: '12px' },

  examplesRow: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '20px' },
  examplesLabel: { color: '#6B7280', fontSize: '13px', fontWeight: 600 },
  exampleButton: {
    background: '#ECFDF5',
    color: '#059669',
    border: '1px solid #A7F3D0',
    borderRadius: '999px',
    padding: '6px 14px',
    fontSize: '12px',
    cursor: 'pointer',
    fontWeight: 600,
  },

  inputGroup: { marginBottom: '18px', position: 'relative' },
  label: { display: 'block', marginBottom: '6px', fontWeight: 600, fontSize: '14px', color: '#374151' },
  input: {
    width: '100%',
    padding: '12px 14px',
    border: '1.5px solid #E5E7EB',
    borderRadius: '10px',
    fontSize: '14px',
    boxSizing: 'border-box',
    background: '#FAFBFC',
  },
  textarea: {
    width: '100%',
    padding: '12px 14px',
    border: '1.5px solid #E5E7EB',
    borderRadius: '10px',
    fontFamily: "'Courier New', monospace",
    fontSize: '13px',
    lineHeight: 1.5,
    minHeight: '80px',
    resize: 'vertical',
    boxSizing: 'border-box',
    background: '#FAFBFC',
  },
  suggestions: {
    position: 'absolute',
    top: '100%',
    left: 0,
    right: 0,
    background: 'white',
    border: '1px solid #E5E7EB',
    borderRadius: '10px',
    marginTop: '4px',
    zIndex: 10,
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
    maxHeight: '260px',
    overflowY: 'auto',
  },
  suggestionItem: {
    display: 'flex',
    justifyContent: 'space-between',
    width: '100%',
    padding: '10px 14px',
    background: 'none',
    border: 'none',
    borderBottom: '1px solid #F3F4F6',
    textAlign: 'left',
    cursor: 'pointer',
    fontSize: '14px',
    color: '#1F2937',
  },
  suggestionMeta: { color: '#6B7280', fontSize: '12px' },

  actionBar: { display: 'flex', gap: '12px', alignItems: 'center', marginTop: '20px' },
  button: {
    background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
    color: 'white',
    border: 'none',
    padding: '12px 24px',
    borderRadius: '10px',
    fontSize: '15px',
    fontWeight: 600,
    cursor: 'pointer',
    boxShadow: '0 4px 12px rgba(16,185,129,0.3)',
  },
  secondaryButton: {
    background: '#F3F4F6',
    color: '#6B7280',
    border: '1.5px solid #E5E7EB',
    padding: '11px 22px',
    borderRadius: '10px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  errorBox: {
    background: '#FEF2F2',
    color: '#991B1B',
    border: '1px solid #FCA5A5',
    padding: '12px 14px',
    borderRadius: '10px',
    marginTop: '16px',
    fontSize: '14px',
  },

  summaryHeader: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: '12px',
    marginBottom: '24px',
  },
  summaryCard: {
    background: '#F9FAFB',
    padding: '16px',
    borderRadius: '10px',
    textAlign: 'center',
  },
  summaryValue: { fontSize: '20px', fontWeight: 700, marginBottom: '4px' },
  summaryLabel: { fontSize: '12px', color: '#6B7280', fontWeight: 600 },
  summarySub: { fontSize: '11px', color: '#9CA3AF', marginTop: '2px' },

  tabs: {
    display: 'flex',
    borderBottom: '2px solid #E5E7EB',
    gap: '8px',
    marginBottom: '20px',
    overflowX: 'auto',
  },
  tab: {
    background: 'none',
    border: 'none',
    padding: '12px 16px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 600,
    borderBottom: '3px solid transparent',
    whiteSpace: 'nowrap',
  },
  tabContent: { animation: 'fadeInUp 0.3s ease-out' },

  card: {
    background: '#F9FAFB',
    padding: '20px',
    borderRadius: '12px',
    border: '1px solid #E5E7EB',
    borderLeft: '4px solid #10B981',
  },
  cardTitle: { margin: '0 0 12px 0', fontSize: '16px', fontWeight: 700 },
  row: { display: 'flex', gap: '12px', padding: '6px 0', borderBottom: '1px dashed #E5E7EB' },
  rowLabel: { width: '180px', color: '#6B7280', fontSize: '13px', fontWeight: 600 },
  rowValue: { flex: 1, color: '#1F2937', fontSize: '14px' },
  list: { margin: '6px 0 6px 20px', padding: 0, color: '#374151', fontSize: '14px' },

  rawBlock: { marginTop: '16px' },
  pre: {
    background: 'white',
    padding: '12px',
    borderRadius: '8px',
    overflow: 'auto',
    fontSize: '12px',
    lineHeight: 1.5,
    border: '1px solid #E5E7EB',
    color: '#374151',
  },
  footer: { textAlign: 'center', padding: '24px', color: '#6B7280', fontSize: '13px' },
};

export default Dashboard;
