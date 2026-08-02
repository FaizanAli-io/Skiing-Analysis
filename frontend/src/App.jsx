import { useEffect, useMemo, useRef, useState } from "react";
import TrendsChart from "./TrendsChart";
import RunAnalysisGraph from "./RunAnalysisGraph";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const DASHBOARD_REFRESH_MS = 5000;
const SCORE_METRICS = [
  ["blue_iq", "Blue IQ"],
  ["pressure", "Pressure"],
  ["balance", "Balance"],
  ["rotation", "Rotation"],
  ["edging", "Edging"],
];

function useAutoRefresh(refresh, enabled = true) {
  const refreshRef = useRef(refresh);

  useEffect(() => {
    refreshRef.current = refresh;
  }, [refresh]);

  useEffect(() => {
    if (!enabled) return undefined;

    let refreshInProgress = false;
    const runRefresh = async () => {
      if (refreshInProgress || document.visibilityState === "hidden") return;

      refreshInProgress = true;
      try {
        await refreshRef.current();
      } catch {
        // A later poll will retry. Initial page loads still surface API errors.
      } finally {
        refreshInProgress = false;
      }
    };

    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") runRefresh();
    };
    const timer = window.setInterval(runRefresh, DASHBOARD_REFRESH_MS);

    window.addEventListener("focus", runRefresh);
    document.addEventListener("visibilitychange", refreshWhenVisible);

    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", runRefresh);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [enabled]);
}

function Logo() {
  return (
    <div className="brand">
      <svg className="brand-mark" viewBox="0 0 36 36" aria-hidden="true">
        <polygon points="16,2 25,2 15,12 6,12" />
        <polygon points="6,12 15,12 7,20 0,14" opacity="0.68" />
        <polygon points="10,20 19,20 30,31 20,31" />
        <polygon points="20,31 30,31 20,36 10,36" opacity="0.68" />
      </svg>
      <span>bluerun</span>
    </div>
  );
}

function routeTo(path) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function useRoute() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  return path;
}

function getToken(kind = "client") {
  return localStorage.getItem(kind === "admin" ? "bluerun_admin_token" : "bluerun_client_token");
}

function setToken(token, kind = "client") {
  localStorage.setItem(kind === "admin" ? "bluerun_admin_token" : "bluerun_client_token", token);
}

function clearToken(kind = "client") {
  localStorage.removeItem(kind === "admin" ? "bluerun_admin_token" : "bluerun_client_token");
}

async function api(path, { method = "GET", token, body, isForm = false } = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body && !isForm) headers["Content-Type"] = "application/json";

  const response = await fetch(`${API_BASE}/api${path}`, {
    method,
    headers,
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });

  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof data === "string" ? data : data.detail || "Request failed";
    throw new Error(Array.isArray(message) ? message.map((item) => item.msg || item).join(", ") : message);
  }
  return data;
}

function fileUrl(value) {
  if (!value) return null;
  if (String(value).startsWith("http")) return value;
  const normalized = String(value).replaceAll("\\", "/");
  const fileName = normalized.split("/").pop();
  if (!fileName) return null;
  if (normalized.startsWith("/outputs/")) return `${API_BASE}${normalized}`;
  return `${API_BASE}/outputs/${fileName}`;
}

function scoreBand(score) {
  const value = Number(score || 0);
  if (value >= 200) return ["Excellent", "excellent"];
  if (value >= 170) return ["Proficient", "proficient"];
  if (value >= 130) return ["Developing", "developing"];
  return ["Emerging", "emerging"];
}

function formatDate(value) {
  if (!value || value === "Unknown date") return "Date unavailable";
  const parsed = new Date(`${String(value).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function recordContext(record) {
  if (!record) return "No completed result yet";
  const parts = [formatDate(record.date)];
  if (record.session_number) parts.push(`Session ${record.session_number}`);
  if (record.run_number) parts.push(`Run ${record.run_number}`);
  return parts.join(" / ");
}

function comparisonLabel(comparison) {
  if (!comparison) return "";
  if (comparison.status === "baseline") return "Baseline established";
  if (comparison.status === "matches_personal_best") return "Matches personal best";
  if (comparison.status === "new_personal_best") {
    const previousScore = comparison.previous_best?.score;
    const gain = previousScore == null ? 0 : comparison.current_score - previousScore;
    return gain > 0 ? `New personal best +${gain}` : "New personal best";
  }
  return `PB ${comparison.personal_best_score} / ${comparison.points_below} points below`;
}

function AuthShell({ children, mode }) {
  return (
    <main className="auth-page">
      <section className="auth-visual">
        <Logo />
        <div className="auth-copy">
          <p className="eyebrow">{mode === "admin" ? "Instructor operations" : "Athlete progress portal"}</p>
          <h1>{mode === "admin" ? "Manage runs, reports, and client progress." : "Review your ski development in one clean place."}</h1>
          <p>
            BlueIQ combines video analysis, score history, and coach-approved feedback to make each session easier to understand.
          </p>
        </div>
      </section>
      <section className="auth-panel">{children}</section>
    </main>
  );
}

function AuthForm({ type }) {
  const isSignup = type === "signup";
  const isAdmin = type === "admin";
  const [form, setForm] = useState({ name: "", email: "", phone: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const title = isSignup ? "Create your account" : isAdmin ? "Admin sign in" : "Welcome back";

  async function submit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const endpoint = isSignup ? "/auth/signup" : "/auth/login";
      const payload = isSignup
        ? { name: form.name, email: form.email, phone: form.phone, password: form.password }
        : { email: form.email, password: form.password };
      const result = await api(endpoint, { method: "POST", body: payload });
      const expectedRole = isAdmin ? "admin" : "client";
      if (result.user.role !== expectedRole) {
        throw new Error(isAdmin ? "This account is not an admin account." : "Please use the admin sign-in page for admin accounts.");
      }
      setToken(result.access_token, isAdmin ? "admin" : "client");
      routeTo(isAdmin ? "/admin" : "/dashboard");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell mode={isAdmin ? "admin" : "client"}>
      <div className="form-card">
        <Logo />
        <p className="eyebrow">{isSignup ? "Client access" : isAdmin ? "Admin portal" : "Client portal"}</p>
        <h2>{title}</h2>
        <form onSubmit={submit}>
          {isSignup && (
            <>
              <label>Name</label>
              <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
              <label>Phone</label>
              <input value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
            </>
          )}
          <label>Email</label>
          <input type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required />
          <label>Password</label>
          <input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} minLength={8} required />
          {error && <div className="alert">{error}</div>}
          <button className="primary-button" disabled={loading}>{loading ? "Working..." : title}</button>
        </form>
        {!isAdmin && (
          <p className="switch-link">
            {isSignup ? "Already have an account?" : "New to Bluerun?"}{" "}
            <button onClick={() => routeTo(isSignup ? "/login" : "/signup")}>{isSignup ? "Sign in" : "Create account"}</button>
          </p>
        )}
        <button className="ghost-link" onClick={() => routeTo(isAdmin ? "/login" : "/admin-login")}>
          {isAdmin ? "Client sign in" : "Admin sign in"}
        </button>
      </div>
    </AuthShell>
  );
}

function AppHeader({ title, subtitle, role, onLogout }) {
  return (
    <header className="app-header">
      <Logo />
      <div>
        <p className="eyebrow">{role}</p>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <button className="secondary-button" onClick={onLogout}>Logout</button>
    </header>
  );
}

function ScorePill({ score }) {
  const [label, key] = scoreBand(score);
  return <span className={`score-pill ${key}`}>{label}</span>;
}

function PersonalBestPanel({ records, title = "Personal bests", description }) {
  return (
    <section className="section-block personal-bests-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">All-time records</p>
          <h2>{title}</h2>
          <p>{description || "Your highest completed result for Blue IQ and each pillar."}</p>
        </div>
      </div>
      <div className="personal-best-grid">
        {SCORE_METRICS.map(([key, label]) => {
          const record = records?.[key];
          return (
            <article className={`personal-best-item ${key === "blue_iq" ? "primary" : ""}`} key={key}>
              <span>{label}</span>
              <strong>{record?.score ?? "--"}<small>/240</small></strong>
              <p>{recordContext(record)}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function LeaderboardPanel({ leaderboards }) {
  const [metric, setMetric] = useState("blue_iq");
  const rows = leaderboards?.[metric] || [];
  const metricLabel = SCORE_METRICS.find(([key]) => key === metric)?.[1] || metric;

  return (
    <section className="section-block leaderboard-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Admin only</p>
          <h2>Personal-best leaderboards</h2>
          <p>Each athlete appears once, using their highest completed result.</p>
        </div>
        <div className="metric-tabs" role="tablist" aria-label="Leaderboard metric">
          {SCORE_METRICS.map(([key, label]) => (
            <button
              type="button"
              role="tab"
              aria-selected={metric === key}
              className={metric === key ? "active" : ""}
              key={key}
              onClick={() => setMetric(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="leaderboard-table" role="table" aria-label={`${metricLabel} leaderboard`}>
        <div className="leaderboard-row leaderboard-head" role="row">
          <span>Rank</span>
          <span>Athlete</span>
          <span>Best context</span>
          <span>Score</span>
        </div>
        {rows.map((row) => (
          <div className="leaderboard-row" role="row" key={`${metric}-${row.person_id}`}>
            <strong className="leaderboard-rank">{row.rank}</strong>
            <div className="leaderboard-athlete">
              <strong>{row.athlete_name}</strong>
              <small>{row.athlete_email}</small>
            </div>
            <span>{recordContext(row)}</span>
            <strong className="leaderboard-score">{row.score}<small>/240</small></strong>
          </div>
        ))}
        {!rows.length && <div className="empty-state">No completed results for {metricLabel} yet.</div>}
      </div>
    </section>
  );
}

function AttemptCard({ attempt, onViewAnalysis }) {
  const blueIq = Math.ceil(
    Number(attempt.blue_iq_score || (Number(attempt.pressure_score || 0) + Number(attempt.balance_score || 0) + Number(attempt.rotation_score || 0) + Number(attempt.edging_score || 0)) / 4)
  );
  const video = fileUrl(attempt.video_link || attempt.output_video_path);
  const report = fileUrl(attempt.report_path);
  const comparisons = attempt.personal_best_comparisons || {};
  const newPersonalBests = new Set(attempt.new_personal_bests || []);
  const runNumber = attempt.run_number || attempt.attempt_number || attempt.id;
  const sessionLabel = attempt.session_number ? `Session ${attempt.session_number}` : "Session";
  const dateLabel = formatDate(attempt.session_date || attempt.created_at || attempt.timestamp);
  return (
    <article className="attempt-card">
      <div className="attempt-top">
        <div>
          <p className="eyebrow">{sessionLabel} / Run {runNumber}</p>
          <h3>Blue IQ {blueIq}<span>/240</span></h3>
          <p className="attempt-date">{dateLabel}</p>
        </div>
        <div className="attempt-badges">
          {newPersonalBests.has("blue_iq") && <span className="new-pb-badge">New personal best</span>}
          <ScorePill score={blueIq} />
        </div>
      </div>
      {comparisons.blue_iq && <p className="blue-iq-comparison">{comparisonLabel(comparisons.blue_iq)}</p>}
      <div className="metric-grid">
        {SCORE_METRICS.slice(1).map(([key, label]) => (
          <div className="metric-row" key={key}>
            <div>
              <span>{label}</span>
              <small className={newPersonalBests.has(key) ? "new-record-copy" : ""}>
                {newPersonalBests.has(key) ? "New personal best" : comparisonLabel(comparisons[key])}
              </small>
            </div>
            <b>{Math.ceil(attempt[`${key}_score`] || 0)}<small>/240</small></b>
          </div>
        ))}
      </div>
      <div className="card-actions">
        {onViewAnalysis && (
          <button type="button" onClick={() => onViewAnalysis(attempt)}>View graph</button>
        )}
        {video && <a href={video} target="_blank" rel="noreferrer">View video</a>}
        {report && <a href={report} target="_blank" rel="noreferrer">View report</a>}
      </div>
    </article>
  );
}

function ClientDashboard() {
  const token = getToken("client");
  const [attempts, setAttempts] = useState([]);
  const [displayCount, setDisplayCount] = useState(4); // Show 4 initially
  const [user, setUser] = useState(null);
  const [personalBests, setPersonalBests] = useState({});
  const [error, setError] = useState("");

  async function loadHistory() {
    const [history, bests] = await Promise.all([
      api("/me/attempts", { token }),
      api("/me/personal-bests", { token }),
    ]);
    setAttempts(history);
    setPersonalBests(bests.personal_bests || {});
  }

  useEffect(() => {
    if (!token) {
      routeTo("/login");
      return;
    }
    Promise.all([
      api("/auth/me", { token }),
      api("/me/attempts", { token }),
      api("/me/personal-bests", { token }),
    ])
      .then(([profile, history, bests]) => {
        setUser(profile);
        setAttempts(history);
        setPersonalBests(bests.personal_bests || {});
      })
      .catch((err) => setError(err.message));
  }, [token]);

  useAutoRefresh(loadHistory, Boolean(token));

  const latest = attempts[0];
  const displayedAttempts = attempts.slice(0, displayCount);
  const hasMore = displayCount < attempts.length;

  return (
    <main className="app-shell">
      <AppHeader
        role="Client portal"
        title={`Welcome${user ? `, ${user.name}` : ""}`}
        subtitle="Review the videos and reports your Bluerun instructor has uploaded for you."
        onLogout={() => {
          clearToken("client");
          routeTo("/login");
        }}
      />
      {error && <div className="alert wide">{error}</div>}
      <section className="summary-grid">
        <div className="summary-card">
          <p className="eyebrow">Attempts</p>
          <strong>{attempts.length}</strong>
        </div>
        <div className="summary-card">
          <p className="eyebrow">Latest Blue IQ</p>
          <strong>{latest ? Math.ceil(latest.blue_iq_score || 0) : "--"}</strong>
        </div>
      </section>
      <PersonalBestPanel records={personalBests} />
      {attempts.length >= 2 && (
        <TrendsChart
          api={api}
          token={token}
          refreshKey={latest?.id || attempts.length}
        />
      )}
      <section className="section-block">
        <div className="section-heading">
          <h2>Your analysis history</h2>
          <p>Videos and PDF reports uploaded by the Bluerun team.</p>
        </div>
        <div className="cards-grid">
          {displayedAttempts.length ? displayedAttempts.map((attempt) => <AttemptCard key={attempt.id} attempt={attempt} />) : <EmptyState text="No attempts have been uploaded yet." />}
        </div>
        {hasMore && (
          <div style={{ textAlign: 'center', marginTop: '24px' }}>
            <button className="secondary-button" onClick={() => setDisplayCount(prev => prev + 4)}>
              Load 4 More
            </button>
          </div>
        )}
      </section>
    </main>
  );
}

function AdminDashboard() {
  const token = getToken("admin");
  const [users, setUsers] = useState([]);
  const [attempts, setAttempts] = useState([]);
  const [leaderboards, setLeaderboards] = useState({});
  const [displayCount, setDisplayCount] = useState(4); // Show 4 initially
  const [selectedClient, setSelectedClient] = useState(""); // Track selected client filter
  const [error, setError] = useState("");
  const [uploadState, setUploadState] = useState("");
  const [graphAttempt, setGraphAttempt] = useState(null);

  async function loadData() {
    const [userRows, attemptRows, leaderboardRows] = await Promise.all([
      api("/admin/users", { token }),
      api("/admin/attempts", { token }),
      api("/admin/leaderboards?limit=50", { token }),
    ]);
    setUsers(userRows);
    setAttempts(attemptRows);
    setLeaderboards(leaderboardRows.leaderboards || {});
  }

  async function loadActivity() {
    const [attemptRows, leaderboardRows] = await Promise.all([
      api("/admin/attempts", { token }),
      api("/admin/leaderboards?limit=50", { token }),
    ]);
    setAttempts(attemptRows);
    setLeaderboards(leaderboardRows.leaderboards || {});
  }

  useEffect(() => {
    if (!token) {
      routeTo("/admin-login");
      return;
    }
    loadData().catch((err) => setError(err.message));
  }, [token]);

  useAutoRefresh(loadActivity, Boolean(token));

  async function pollJobStatus(jobId) {
    const maxAttempts = 120; // Poll for up to 10 minutes (120 * 5 seconds)
    let attempts = 0;
    
    const poll = async () => {
      try {
        const status = await api(`/jobs/${jobId}`, { token });
        
        if (status.status === "completed") {
          setUploadState("Analysis complete!");
          await loadData();
          return true;
        } else if (status.status === "failed") {
          setError(`Analysis failed: ${status.error_message || "Unknown error"}`);
          setUploadState("");
          return true;
        } else {
          // Still processing
          const progressText = status.progress > 0 ? ` (${status.progress}%)` : "";
          setUploadState(`Processing video${progressText}...`);
          
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(poll, 5000); // Poll every 5 seconds
          } else {
            setUploadState("Analysis is taking longer than expected. Check back later.");
          }
          return false;
        }
      } catch (err) {
        setError(`Failed to check job status: ${err.message}`);
        setUploadState("");
        return true;
      }
    };
    
    await poll();
  }

  async function submitUpload(event) {
    event.preventDefault();
    const formElement = event.currentTarget;
    setError("");
    setUploadState("Uploading video...");
    const form = new FormData(formElement);
    
    try {
      // Submit video for background processing
      const response = await api("/analyze-premium-overlay/", {
        method: "POST",
        token,
        body: form,
        isForm: true,
      });
      
      // Response contains job_id for tracking
      if (response.job_id) {
        formElement.reset();
        setUploadState("Video uploaded. Processing started...");
        // Start polling for job completion
        await pollJobStatus(response.job_id);
      } else {
        // Fallback for old response format (shouldn't happen)
        formElement.reset();
        setUploadState("Analysis complete.");
        await loadData();
      }
    } catch (err) {
      setUploadState("");
      setError(err.message);
    }
  }

  // Filter attempts by selected client
  const filteredAttempts = selectedClient 
    ? attempts.filter(attempt => attempt.person_id === parseInt(selectedClient))
    : attempts;
  
  const displayedAttempts = filteredAttempts.slice(0, displayCount);
  const hasMore = displayCount < filteredAttempts.length;

  return (
    <main className="app-shell">
      <AppHeader
        role="Admin portal"
        title="Bluerun operations"
        subtitle="Upload videos for clients, review generated reports, and monitor analysis history."
        onLogout={() => {
          clearToken("admin");
          routeTo("/admin-login");
        }}
      />
      {error && <div className="alert wide">{error}</div>}
      <section className="admin-layout">
        <form className="upload-panel" onSubmit={submitUpload}>
          <p className="eyebrow">New analysis</p>
          <h2>Upload client video</h2>
          <label>Select user</label>
          <select name="user_id" required>
            <option value="">Choose a client</option>
            {users.filter((user) => user.role !== "admin").map((user) => (
              <option key={user.id} value={user.id}>{user.name} - {user.email}</option>
            ))}
          </select>
          <label>Display mode</label>
          <select name="display_mode" defaultValue="coach">
            <option value="coach">Coach</option>
            <option value="athlete">Athlete</option>
          </select>
          <label className="checkbox-row">
            <input type="checkbox" name="report" value="true" defaultChecked />
            Generate PDF report
          </label>
          <label>Video file</label>
          <input type="file" name="file" accept="video/*" required />
          <button className="primary-button">Run analysis</button>
          {uploadState && <p className="status-text">{uploadState}</p>}
        </form>
        <section className="user-panel">
          <div className="section-heading">
            <h2>Clients</h2>
            <p>{users.filter((user) => user.role !== "admin").length} client accounts</p>
          </div>
          <div className="user-list">
            {users.filter((user) => user.role !== "admin").map((user) => (
              <div className="user-row" key={user.id}>
                <span>{user.name}</span>
                <small>{user.email}</small>
              </div>
            ))}
          </div>
        </section>
      </section>
      <LeaderboardPanel leaderboards={leaderboards} />
      <section className="section-block">
        <div className="section-heading">
          <div>
            <h2>Recent attempts</h2>
            <p>Generated videos and reports across all clients.</p>
          </div>
          <div>
            <label style={{ marginRight: '8px', color: 'var(--muted)', fontSize: '13px' }}>Filter by client:</label>
            <select 
              value={selectedClient} 
              onChange={(e) => {
                setSelectedClient(e.target.value);
                setDisplayCount(4); // Reset to 4 when filter changes
              }}
              style={{ 
                padding: '8px 12px', 
                background: '#091426',
                border: '1px solid var(--line-soft)',
                borderRadius: '8px',
                color: 'var(--text)'
              }}
            >
              <option value="">All clients</option>
              {users.filter((user) => user.role !== "admin").map((user) => (
                <option key={user.id} value={user.id}>{user.name}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="cards-grid">
          {displayedAttempts.length ? displayedAttempts.map((attempt) => (
            <AttemptCard
              key={attempt.id}
              attempt={attempt}
              onViewAnalysis={setGraphAttempt}
            />
          )) : <EmptyState text="No attempts have been generated yet." />}
        </div>
        {hasMore && (
          <div style={{ textAlign: 'center', marginTop: '24px' }}>
            <button className="secondary-button" onClick={() => setDisplayCount(prev => prev + 4)}>
              Load 4 More ({filteredAttempts.length - displayCount} remaining)
            </button>
          </div>
        )}
      </section>
      {graphAttempt && (
        <RunAnalysisGraph
          api={api}
          token={token}
          attempt={graphAttempt}
          athleteName={users.find((user) => user.id === graphAttempt.person_id)?.name}
          onClose={() => setGraphAttempt(null)}
        />
      )}
    </main>
  );
}

function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>;
}

function App() {
  const path = useRoute();
  const component = useMemo(() => {
    if (path === "/signup") return <AuthForm type="signup" />;
    if (path === "/admin-login") return <AuthForm type="admin" />;
    if (path === "/admin") return <AdminDashboard />;
    if (path === "/dashboard") return <ClientDashboard />;
    return <AuthForm type="login" />;
  }, [path]);
  return component;
}

export default App;
