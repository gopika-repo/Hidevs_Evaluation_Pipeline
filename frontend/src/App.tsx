import { useState, useEffect } from 'react';
import './App.css';

interface SubScores {
  [key: string]: any;
}

interface EvaluatorDetail {
  score: number | null;
  max_score: number;
  status: string;
  sub_scores: SubScores;
  feedback: string;
  flagged?: boolean;
  applicable?: boolean;
  percentage?: number;
  detected_intent?: string;
  expected_intent?: string;
  expected_intent_status?: string;
  misclassified?: boolean;
  critical_violation?: boolean;
}

interface ConversationEvaluation {
  conversation_id: string;
  conversation_type: string;
  raw_applicable_score: number;
  applicable_max_score: number;
  overall_health_score: number | null;
  flagged: boolean;
  flagged_for_quality: boolean;
  evaluation_failed: boolean;
  evaluations: {
    response_quality: EvaluatorDetail;
    groundedness: EvaluatorDetail;
    safety: EvaluatorDetail;
    intent_understanding: EvaluatorDetail;
    memory_and_continuity?: EvaluatorDetail;
  };
}

interface EvaluationReport {
  pipeline_phase: string;
  summary_stats: {
    total_conversations: number;
    flagged_conversations: number;
    averages: {
      [key: string]: number | null;
    };
  };
}

const renderScoreDisplay = (score: number | null, maxScore: number, status: string) => {
  if (status === 'success' || status === 'evaluated') {
    return <>{score !== null ? score : '0'} <span>/ {maxScore}</span></>;
  }
  if (status === 'not_applicable') {
    return <span className="score-text-status">NOT APPLICABLE</span>;
  }
  const statusLabels: { [key: string]: string } = {
    failed: 'FAILED',
    timeout: 'TIMED OUT',
    invalid_output: 'INVALID OUTPUT',
    unavailable: 'UNAVAILABLE',
  };
  return <span className="score-text-status error">{statusLabels[status] || status.toUpperCase()}</span>;
};

function App() {
  // Form states
  const [conversationId, setConversationId] = useState('');
  const [userQuery, setUserQuery] = useState('');
  const [daveResponse, setDaveResponse] = useState('');
  const [retrievedContext, setRetrievedContext] = useState('');
  const [chatHistory, setChatHistory] = useState('');
  const [expectedIntent, setExpectedIntent] = useState('');
  const [timestamp, setTimestamp] = useState(new Date().toISOString());

  // App API & Status States
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<{ title: string; desc: string } | null>(null);
  const [report, setReport] = useState<EvaluationReport | null>(null);

  // Accordion open states
  const [expandedCards, setExpandedCards] = useState<{ [key: string]: boolean }>({
    response_quality: false,
    groundedness: false,
    safety: false,
    intent_understanding: false,
    memory_and_continuity: false,
  });

  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  // Check API health status
  const checkHealth = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/health`, { method: 'GET' });
      if (res.ok) {
        setApiStatus('connected');
      } else {
        setApiStatus('disconnected');
      }
    } catch {
      setApiStatus('disconnected');
    }
  };

  useEffect(() => {
    checkHealth();
    // Poll health status every 15 seconds
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const toggleCard = (cardKey: string) => {
    setExpandedCards((prev) => ({ ...prev, [cardKey]: !prev[cardKey] }));
  };

  const handleEvaluate = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setReport(null);

    // Form validations
    if (!conversationId.trim()) {
      setErrorMsg({ title: 'Validation Error', desc: 'Conversation ID is required.' });
      return;
    }
    if (!userQuery.trim()) {
      setErrorMsg({ title: 'Validation Error', desc: 'User Query is required.' });
      return;
    }
    if (!daveResponse.trim()) {
      setErrorMsg({ title: 'Validation Error', desc: 'Dave Response is required.' });
      return;
    }

    setIsLoading(true);

    const payload = {
      conversation_id: conversationId.trim(),
      user_query: userQuery.trim(),
      dave_response: daveResponse.trim(),
      retrieved_context: retrievedContext.trim() || null,
      chat_history: chatHistory.trim() || null,
      expected_intent: expectedIntent || null,
      timestamp: timestamp,
    };

    try {
      const response = await fetch(`${apiBaseUrl}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (response.status === 200) {
        const data = await response.json();
        setReport(data);
      } else if (response.status === 422) {
        const errData = await response.json();
        setErrorMsg({
          title: 'Validation Error (422)',
          desc: errData.detail ? JSON.stringify(errData.detail) : 'The server rejected the inputs.',
        });
      } else if (response.status === 500) {
        setErrorMsg({
          title: 'Evaluation Failure (500)',
          desc: 'The evaluation engine crashed or failed to complete evaluations.',
        });
      } else {
        setErrorMsg({
          title: `Server Error (${response.status})`,
          desc: 'An unexpected response was returned by the server.',
        });
      }
    } catch {
      setErrorMsg({
        title: 'Connection Refused',
        desc: 'Unable to connect to the evaluation API. Verify the backend server is running.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const currentConvo = report?.conversations?.[0];

  return (
    <div className="app-container">
      <header>
        <div className="header-left">
          <h1>Dave AI Evaluation Pipeline</h1>
          <p>Phase 1 — Conversation Quality & Safety Evaluation</p>
        </div>
        <div className="api-status-badge">
          <span className={`status-dot ${apiStatus}`} />
          {apiStatus === 'checking' && 'Checking API Connection...'}
          {apiStatus === 'connected' && 'API Connected'}
          {apiStatus === 'disconnected' && 'API Disconnected'}
        </div>
      </header>

      {errorMsg && (
        <div className="error-container">
          <div>
            <h3 className="error-title">{errorMsg.title}</h3>
            <p className="error-desc">{errorMsg.desc}</p>
          </div>
        </div>
      )}

      <div className="dashboard-grid">
        {/* Input Panel */}
        <div className="panel-card">
          <h2>Conversation Input</h2>
          <form onSubmit={handleEvaluate}>
            <div className="form-group row">
              <div>
                <label>Conversation ID *</label>
                <input
                  type="text"
                  placeholder="e.g. CONV-001"
                  value={conversationId}
                  onChange={(e) => setConversationId(e.target.value)}
                  disabled={isLoading}
                />
              </div>
              <div>
                <label>Timestamp *</label>
                <input
                  type="text"
                  value={timestamp}
                  onChange={(e) => setTimestamp(e.target.value)}
                  disabled={isLoading}
                />
              </div>
            </div>

            <div className="form-group">
              <label>Expected Intent (Optional)</label>
              <select
                value={expectedIntent}
                onChange={(e) => setExpectedIntent(e.target.value)}
                disabled={isLoading}
              >
                <option value="">Not Provided</option>
                <option value="personal">Personal</option>
                <option value="technical">Technical</option>
                <option value="platform">Platform</option>
                <option value="out_of_scope">Out of Scope</option>
                <option value="ambiguous">Ambiguous</option>
              </select>
            </div>

            <div className="form-group">
              <label>User Query *</label>
              <textarea
                placeholder="Enter the user query..."
                value={userQuery}
                onChange={(e) => setUserQuery(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label>Dave Response *</label>
              <textarea
                placeholder="Enter Dave's response..."
                value={daveResponse}
                onChange={(e) => setDaveResponse(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label>Retrieved Context (Optional)</label>
              <textarea
                placeholder="Supporting knowledge base docs retrieved for RAG..."
                value={retrievedContext}
                onChange={(e) => setRetrievedContext(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label>Chat History (Optional)</label>
              <textarea
                placeholder="Prior conversation logs (e.g. User: Hi\nDave: Hello)..."
                value={chatHistory}
                onChange={(e) => setChatHistory(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <button type="submit" className="btn-evaluate" disabled={isLoading}>
              {isLoading ? (
                <>
                  <span className="spinner" /> Evaluating...
                </>
              ) : (
                'Evaluate Conversation'
              )}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="panel-card" style={{ flex: 1 }}>
          <h2>Evaluation Results</h2>

          {!currentConvo && !isLoading && (
            <div className="placeholder-box">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 0 1-2.25 2.25M16.5 7.5V18a2.25 2.25 0 0 0 2.25 2.25M16.5 7.5V4.875c0-.621-.504-1.125-1.125-1.125H4.125C3.504 3.75 3 4.254 3 4.875V18a2.25 2.25 0 0 0 2.25 2.25h13.5M6 7.5h3v3H6v-3Z" />
              </svg>
              <p style={{ margin: 0, fontWeight: 500, fontSize: '1.05rem', color: '#94a3b8' }}>No active evaluation report</p>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.9rem', color: '#64748b' }}>Enter conversation details on the left and click evaluate.</p>
            </div>
          )}

          {isLoading && (
            <div className="placeholder-box" style={{ borderStyle: 'solid', borderColor: 'rgba(255,255,255,0.02)' }}>
              <span className="spinner" style={{ width: '32px', height: '32px', borderWidth: '3px', marginBottom: '1rem', borderTopColor: '#6366f1' }} />
              <p style={{ margin: 0, fontWeight: 500, color: '#e2e8f0' }}>Executing LLM Evaluation Pipeline</p>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.9rem', color: '#64748b' }}>Running all judges concurrently. Please wait...</p>
            </div>
          )}

          {currentConvo && (
            <>
              {/* Overall Health Score Card */}
              <div className="health-overview">
                <div className="health-left">
                  <h3>Overall Health Score</h3>
                  <div className="score-ratio">
                    {currentConvo.overall_health_score !== null ? (
                      <>
                        {currentConvo.raw_applicable_score} <span>/ {currentConvo.applicable_max_score}</span>
                      </>
                    ) : (
                      'N/A'
                    )}
                  </div>
                </div>
                <div className="health-radial">
                  <div className="health-percentage">
                    {currentConvo.overall_health_score !== null ? (
                      <>
                        {currentConvo.overall_health_score}
                        <span>%</span>
                      </>
                    ) : (
                      'N/A'
                    )}
                  </div>
                </div>
              </div>

              {/* Flags Warning Box */}
              {currentConvo.flagged && (
                <div className={`warnings-panel ${currentConvo.evaluations.safety?.critical_violation ? 'critical' : ''}`}>
                  <div className="warnings-header">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                    </svg>
                    {currentConvo.evaluations.safety?.critical_violation ? 'CRITICAL SAFETY LEAK DETECTED' : 'CONVERSATION FLAGGED FOR REVIEW'}
                  </div>
                  <div className="flags-grid">
                    <div className="flag-item">
                      <span className="flag-status active" />
                      flagged
                    </div>
                    <div className="flag-item">
                      <span className={`flag-status ${currentConvo.flagged_for_quality ? 'active' : 'inactive'}`} />
                      flagged_for_quality
                    </div>
                    <div className="flag-item">
                      <span className={`flag-status ${currentConvo.evaluations.safety?.critical_violation ? 'active' : 'inactive'}`} />
                      critical_violation
                    </div>
                    <div className="flag-item">
                      <span className={`flag-status ${currentConvo.evaluation_failed ? 'active' : 'inactive'}`} />
                      evaluation_failed
                    </div>
                  </div>
                </div>
              )}

              {/* 1. Response Quality Card */}
              {(() => {
                const rq = currentConvo.evaluations.response_quality;
                return (
                  <div className="evaluator-card">
                    <div className="card-header" onClick={() => toggleCard('response_quality')}>
                      <div className="card-header-left">
                        <span className={`badge ${rq.status}`} >{rq.status}</span>
                        <h3>Response Quality</h3>
                      </div>
                      <div className="card-header-right">
                        <div className="card-score">
                          {renderScoreDisplay(rq.score, rq.max_score, rq.status)}
                        </div>
                        <svg className={`chevron ${expandedCards.response_quality ? 'open' : ''}`} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="6 9 12 15 18 9" />
                        </svg>
                      </div>
                    </div>
                    {expandedCards.response_quality && (
                      <div className="card-content">
                        <div className="subscores-grid">
                          <div className="subscore-item">
                            <div className="subscore-label">Correctness</div>
                            <div className="subscore-value">{rq.sub_scores.correctness !== undefined ? `${rq.sub_scores.correctness} / 5` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Helpfulness</div>
                            <div className="subscore-value">{rq.sub_scores.helpfulness !== undefined ? `${rq.sub_scores.helpfulness} / 5` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Clarity</div>
                            <div className="subscore-value">{rq.sub_scores.clarity !== undefined ? `${rq.sub_scores.clarity} / 5` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Completeness</div>
                            <div className="subscore-value">{rq.sub_scores.completeness !== undefined ? `${rq.sub_scores.completeness} / 5` : 'N/A'}</div>
                          </div>
                        </div>
                        <div className="feedback-box">
                          <h4>Judge Feedback</h4>
                          <p className="feedback-text">{rq.feedback}</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* 2. Groundedness Card */}
              {(() => {
                const gd = currentConvo.evaluations.groundedness;
                return (
                  <div className="evaluator-card">
                    <div className="card-header" onClick={() => toggleCard('groundedness')}>
                      <div className="card-header-left">
                        <span className={`badge ${gd.status}`}>{gd.status}</span>
                        <h3>Groundedness & Hallucination</h3>
                      </div>
                      <div className="card-header-right">
                        <div className="card-score">
                          {renderScoreDisplay(gd.score, gd.max_score, gd.status)}
                        </div>
                        <svg className={`chevron ${expandedCards.groundedness ? 'open' : ''}`} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="6 9 12 15 18 9" />
                        </svg>
                      </div>
                    </div>
                    {expandedCards.groundedness && (
                      <div className="card-content">
                        <div className="subscores-grid">
                          <div className="subscore-item">
                            <div className="subscore-label">Internal Consistency</div>
                            <div className="subscore-value">{gd.sub_scores.internal_consistency !== undefined ? `${gd.sub_scores.internal_consistency} / 6` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Overconfidence</div>
                            <div className="subscore-value">{gd.sub_scores.overconfidence !== undefined ? `${gd.sub_scores.overconfidence} / 6` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Hallucination Risk</div>
                            <div className="subscore-value">{gd.sub_scores.hallucination_risk !== undefined ? `${gd.sub_scores.hallucination_risk} / 8` : 'N/A'}</div>
                          </div>
                        </div>

                        {/* Separate TruLens & DeepEval indicators */}
                        <div className="framework-verification">
                          <div className="framework-title">External Framework Verification</div>
                          <div className="framework-flex">
                            <div className="framework-item">
                              <div className="subscore-label">TruLens — Validation</div>
                              <div className="val">
                                {gd.sub_scores.trulens_status === 'success' && gd.sub_scores.trulens_score !== undefined
                                  ? gd.sub_scores.trulens_score
                                  : gd.sub_scores.trulens_status || 'N/A'}
                              </div>
                            </div>
                            <div className="framework-item">
                              <div className="subscore-label">DeepEval — Validation</div>
                              <div className="val">
                                {gd.sub_scores.deepeval_status === 'success' && gd.sub_scores.deepeval_score !== undefined
                                  ? gd.sub_scores.deepeval_score
                                  : gd.sub_scores.deepeval_status || 'N/A'}
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="feedback-box">
                          <h4>Judge Feedback</h4>
                          <p className="feedback-text">{gd.feedback}</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* 3. Safety Card */}
              {(() => {
                const sf = currentConvo.evaluations.safety;
                return (
                  <div className="evaluator-card">
                    <div className="card-header" onClick={() => toggleCard('safety')}>
                      <div className="card-header-left">
                        <span className={`badge ${sf.status}`}>{sf.status}</span>
                        <h3>Safety & Policy Compliance</h3>
                      </div>
                      <div className="card-header-right">
                        <div className="card-score">
                          {renderScoreDisplay(sf.score, sf.max_score, sf.status)}
                        </div>
                        <svg className={`chevron ${expandedCards.safety ? 'open' : ''}`} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="6 9 12 15 18 9" />
                        </svg>
                      </div>
                    </div>
                    {expandedCards.safety && (
                      <div className="card-content">
                        <div className="subscores-grid">
                          <div className="subscore-item">
                            <div className="subscore-label">Confidentiality & Information</div>
                            <div className="subscore-value">{sf.sub_scores.confidentiality_information_protection !== undefined ? `${sf.sub_scores.confidentiality_information_protection} / 6` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Security Attack Resistance</div>
                            <div className="subscore-value">{sf.sub_scores.security_attack_resistance !== undefined ? `${sf.sub_scores.security_attack_resistance} / 6` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Boundary & Policy</div>
                            <div className="subscore-value">{sf.sub_scores.boundary_policy_compliance !== undefined ? `${sf.sub_scores.boundary_policy_compliance} / 8` : 'N/A'}</div>
                          </div>
                        </div>

                        <div className="framework-verification">
                          <div className="framework-title">System Policy Indicators</div>
                          <div className="framework-flex">
                            <div className="framework-item" style={{ flex: 'none', width: '22%' }}>
                              <div className="subscore-label">Attack Detected</div>
                              <div className="val" style={{ fontSize: '1rem', color: sf.sub_scores.attack_detected ? '#f87171' : '#a5b4fc' }}>
                                {sf.sub_scores.attack_detected ? 'YES' : 'NO'}
                              </div>
                            </div>
                            <div className="framework-item" style={{ flex: 'none', width: '22%' }}>
                              <div className="subscore-label">Attack Resisted</div>
                              <div className="val" style={{ fontSize: '1rem' }}>
                                {sf.sub_scores.attack_resisted ? 'YES' : 'NO'}
                              </div>
                            </div>
                            <div className="framework-item" style={{ flex: 'none', width: '26%' }}>
                              <div className="subscore-label">Confidential Leak</div>
                              <div className="val" style={{ fontSize: '1rem', color: sf.sub_scores.actual_confidential_leak ? '#ef4444' : '#a5b4fc' }}>
                                {sf.sub_scores.actual_confidential_leak ? 'YES' : 'NO'}
                              </div>
                            </div>
                            <div className="framework-item" style={{ flex: 'none', width: '30%' }}>
                              <div className="subscore-label">Critical Override</div>
                              <div className="val" style={{ fontSize: '1rem', color: sf.sub_scores.critical_violation ? '#ef4444' : '#a5b4fc' }}>
                                {sf.sub_scores.critical_violation ? 'APPLIED' : 'NOT APPLIED'}
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="feedback-box">
                          <h4>Judge Feedback</h4>
                          <p className="feedback-text">{sf.feedback}</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* 4. Intent Understanding Card */}
              {(() => {
                const it = currentConvo.evaluations.intent_understanding;
                return (
                  <div className="evaluator-card">
                    <div className="card-header" onClick={() => toggleCard('intent_understanding')}>
                      <div className="card-header-left">
                        <span className={`badge ${it.status}`}>{it.status}</span>
                        <h3>Intent Understanding</h3>
                      </div>
                      <div className="card-header-right">
                        <div className="card-score">
                          {renderScoreDisplay(it.score, it.max_score, it.status)}
                        </div>
                        <svg className={`chevron ${expandedCards.intent_understanding ? 'open' : ''}`} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="6 9 12 15 18 9" />
                        </svg>
                      </div>
                    </div>
                    {expandedCards.intent_understanding && (
                      <div className="card-content">
                        <div className="subscores-grid">
                          <div className="subscore-item">
                            <div className="subscore-label">Intent Accuracy</div>
                            <div className="subscore-value">{it.sub_scores.intent_accuracy !== undefined ? `${it.sub_scores.intent_accuracy} / 8` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Clarification Handling</div>
                            <div className="subscore-value">{it.sub_scores.clarification_handling !== undefined ? `${it.sub_scores.clarification_handling} / 6` : 'N/A'}</div>
                          </div>
                          <div className="subscore-item">
                            <div className="subscore-label">Misclassification Penalty</div>
                            <div className="subscore-value">{it.sub_scores.misclassification_penalty !== undefined ? `${it.sub_scores.misclassification_penalty} / 6` : 'N/A'}</div>
                          </div>
                        </div>

                        <div className="framework-verification">
                          <div className="framework-title">Intent Context Details</div>
                          <div className="framework-flex">
                            <div className="framework-item">
                              <div className="subscore-label">Detected Intent</div>
                              <div className="val" style={{ fontSize: '1rem' }}>{it.detected_intent || 'None'}</div>
                            </div>
                            <div className="framework-item">
                              <div className="subscore-label">Expected Intent</div>
                              <div className="val" style={{ fontSize: '1rem' }}>{it.expected_intent || 'Not Provided'}</div>
                            </div>
                            <div className="framework-item">
                              <div className="subscore-label">Expected Intent Status</div>
                              <div className="val" style={{ fontSize: '1rem' }}>{it.expected_intent_status || 'not_provided'}</div>
                            </div>
                            <div className="framework-item">
                              <div className="subscore-label">Misclassified</div>
                              <div className="val" style={{ fontSize: '1rem', color: it.misclassified ? '#f87171' : '#a5b4fc' }}>
                                {it.misclassified ? 'YES' : 'NO'}
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="feedback-box">
                          <h4>Judge Feedback</h4>
                          <p className="feedback-text">{it.feedback}</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* 5. Memory & Context Continuity Card */}
              {(() => {
                const me = currentConvo.evaluations.memory_and_continuity;
                if (!me) return null;

                const statusBadgeText = me.status;
                const badgeClass = me.status === 'evaluated' || me.status === 'success' ? 'success' : me.status;

                return (
                  <div className="evaluator-card">
                    <div className="card-header" onClick={() => toggleCard('memory_and_continuity')}>
                      <div className="card-header-left">
                        <span className={`badge ${badgeClass}`}>{statusBadgeText}</span>
                        <h3>Memory & Continuity</h3>
                      </div>
                      <div className="card-header-right">
                        <div className="card-score">
                          {renderScoreDisplay(me.score, me.max_score, me.status)}
                        </div>
                        <svg className={`chevron ${expandedCards.memory_and_continuity ? 'open' : ''}`} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="6 9 12 15 18 9" />
                        </svg>
                      </div>
                    </div>
                    {expandedCards.memory_and_continuity && (
                      <div className="card-content">
                        {isApp ? (
                          <>
                            <div className="subscores-grid">
                              <div className="subscore-item">
                                <div className="subscore-label">Context Continuity</div>
                                <div className="subscore-value">{me.sub_scores.context_continuity !== undefined ? `${me.sub_scores.context_continuity} / 8` : 'N/A'}</div>
                              </div>
                              <div className="subscore-item">
                                <div className="subscore-label">Information Retention</div>
                                <div className="subscore-value">{me.sub_scores.information_retention !== undefined ? `${me.sub_scores.information_retention} / 6` : 'N/A'}</div>
                              </div>
                              <div className="subscore-item">
                                <div className="subscore-label">Consistency Across Turns</div>
                                <div className="subscore-value">{me.sub_scores.consistency_across_turns !== undefined ? `${me.sub_scores.consistency_across_turns} / 6` : 'N/A'}</div>
                              </div>
                            </div>
                            <div className="feedback-box">
                              <h4>Judge Feedback</h4>
                              <p className="feedback-text">{me.feedback}</p>
                            </div>
                          </>
                        ) : (
                          <div style={{ padding: '1rem', textAlign: 'center', color: '#64748b', background: 'rgba(255,255,255,0.01)', borderRadius: '8px' }}>
                            Memory evaluation is not applicable for context-free or single-turn conversations.
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })()}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
