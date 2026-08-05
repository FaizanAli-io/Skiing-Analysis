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

function normalizeSearchText(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function fuzzyFieldScore(field, term) {
  if (!field || !term) return Number.POSITIVE_INFINITY;
  if (field === term) return 0;
  if (field.startsWith(term)) return 1 + ((field.length - term.length) / 1000);

  const substringIndex = field.indexOf(term);
  if (substringIndex >= 0) return 3 + (substringIndex / 100);

  let termIndex = 0;
  let firstMatch = -1;
  let previousMatch = -1;
  let skippedCharacters = 0;

  for (let fieldIndex = 0; fieldIndex < field.length && termIndex < term.length; fieldIndex += 1) {
    if (field[fieldIndex] !== term[termIndex]) continue;
    if (firstMatch < 0) firstMatch = fieldIndex;
    if (previousMatch >= 0) skippedCharacters += fieldIndex - previousMatch - 1;
    previousMatch = fieldIndex;
    termIndex += 1;
  }

  if (termIndex !== term.length) return Number.POSITIVE_INFINITY;
  return 10 + skippedCharacters + (firstMatch / 100);
}

function clientSearchScore(client, query) {
  const terms = normalizeSearchText(query).split(" ").filter(Boolean);
  if (!terms.length) return 0;

  const fields = [normalizeSearchText(client.name), normalizeSearchText(client.email)];
  return terms.reduce((total, term) => {
    const score = Math.min(...fields.map((field) => fuzzyFieldScore(field, term)));
    return Number.isFinite(total) && Number.isFinite(score)
      ? total + score
      : Number.POSITIVE_INFINITY;
  }, 0);
}

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
  const [topK, setTopK] = useState(10);
  const rows = (leaderboards?.[metric] || []).slice(0, topK);
  const metricLabel = SCORE_METRICS.find(([key]) => key === metric)?.[1] || metric;

  return (
    <section className="section-block leaderboard-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Admin only</p>
          <h2>Personal-best leaderboards</h2>
          <p>Each athlete appears once, using their highest completed result.</p>
        </div>
        <div className="leaderboard-controls">
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
          <label className="top-k-control">
            <span>Ranking size</span>
            <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              {[5, 10, 25, 50].map((value) => (
                <option value={value} key={value}>Top {value}</option>
              ))}
            </select>
          </label>
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

function UploadAnalysisPanel({ token, clients = [], fixedUser = null, onCompleted }) {
  const [uploadState, setUploadState] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function waitForJob(jobId) {
    for (let pollCount = 0; pollCount < 120; pollCount += 1) {
      const status = await api(`/jobs/${jobId}`, { token });

      if (status.status === "completed") {
        setUploadState("Analysis complete!");
        if (onCompleted) await onCompleted();
        return;
      }
      if (status.status === "failed") {
        throw new Error(`Analysis failed: ${status.error_message || "Unknown error"}`);
      }

      const progressText = status.progress > 0 ? ` (${status.progress}%)` : "";
      setUploadState(`Processing video${progressText}...`);
      await new Promise((resolve) => window.setTimeout(resolve, 5000));
    }

    setUploadState("Analysis is taking longer than expected. Check back later.");
  }

  async function submitUpload(event) {
    event.preventDefault();
    const formElement = event.currentTarget;
    setUploadError("");
    setUploadState("Uploading video...");
    setIsSubmitting(true);

    try {
      const response = await api("/analyze-premium-overlay/", {
        method: "POST",
        token,
        body: new FormData(formElement),
        isForm: true,
      });

      formElement.reset();
      if (response.job_id) {
        setUploadState("Video uploaded. Processing started...");
        await waitForJob(response.job_id);
      } else {
        setUploadState("Analysis complete.");
        if (onCompleted) await onCompleted();
      }
    } catch (err) {
      setUploadState("");
      setUploadError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form className="upload-panel" onSubmit={submitUpload}>
      <p className="eyebrow">New analysis</p>
      <h2>{fixedUser ? `Upload a run for ${fixedUser.name}` : "Upload client video"}</h2>
      {fixedUser ? (
        <>
          <input type="hidden" name="user_id" value={fixedUser.id} />
          <div className="fixed-upload-athlete">
            <span>{fixedUser.name}</span>
            <small>{fixedUser.email}</small>
          </div>
        </>
      ) : (
        <>
          <label>Select user</label>
          <select name="user_id" required>
            <option value="">Choose a client</option>
            {clients.map((user) => (
              <option key={user.id} value={user.id}>{user.name} - {user.email}</option>
            ))}
          </select>
        </>
      )}
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
      <button className="primary-button" disabled={isSubmitting}>
        {isSubmitting ? "Processing..." : "Run analysis"}
      </button>
      {uploadError && <div className="alert">{uploadError}</div>}
      {uploadState && <p className="status-text">{uploadState}</p>}
    </form>
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

function AdminAthleteProfile({ userId }) {
  const token = getToken("admin");
  const [user, setUser] = useState(null);
  const [attempts, setAttempts] = useState([]);
  const [personalBests, setPersonalBests] = useState({});
  const [displayCount, setDisplayCount] = useState(4);
  const [graphAttempt, setGraphAttempt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadProfileActivity() {
    const [attemptRows, bestRows] = await Promise.all([
      api(`/admin/users/${userId}/attempts?limit=100`, { token }),
      api(`/admin/users/${userId}/personal-bests`, { token }),
    ]);
    setAttempts(attemptRows);
    setPersonalBests(bestRows.personal_bests || {});
  }

  useEffect(() => {
    if (!token) {
      routeTo("/admin-login");
      return;
    }

    let cancelled = false;
    setLoading(true);
    Promise.all([
      api(`/admin/users/${userId}`, { token }),
      api(`/admin/users/${userId}/attempts?limit=100`, { token }),
      api(`/admin/users/${userId}/personal-bests`, { token }),
    ])
      .then(([profile, attemptRows, bestRows]) => {
        if (cancelled) return;
        setUser(profile);
        setAttempts(attemptRows);
        setPersonalBests(bestRows.personal_bests || {});
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [token, userId]);

  useAutoRefresh(loadProfileActivity, Boolean(token && userId));

  const completedAttempts = attempts.filter((attempt) => !attempt.status || attempt.status === "completed");
  const latest = completedAttempts[0];
  const latestBlueIq = latest
    ? Math.ceil(Number(latest.blue_iq_score || (
      Number(latest.pressure_score || 0)
      + Number(latest.balance_score || 0)
      + Number(latest.rotation_score || 0)
      + Number(latest.edging_score || 0)
    ) / 4))
    : null;
  const displayedAttempts = completedAttempts.slice(0, displayCount);
  const hasMore = displayCount < completedAttempts.length;
  const latestVideo = latest ? fileUrl(latest.video_link || latest.output_video_path) : null;
  const latestReport = latest ? fileUrl(latest.report_path) : null;

  return (
    <main className="app-shell">
      <div className="profile-back-row">
        <button type="button" className="secondary-button" onClick={() => routeTo("/admin")}>Back to athletes</button>
      </div>
      <AppHeader
        role="Admin / Athlete profile"
        title={user?.name || "Athlete profile"}
        subtitle="Upload new runs and review this athlete's generated videos, reports, and score history."
        onLogout={() => {
          clearToken("admin");
          routeTo("/admin-login");
        }}
      />
      {error && <div className="alert wide">{error}</div>}
      {loading ? (
        <section className="section-block"><div className="empty-state">Loading athlete profile...</div></section>
      ) : user ? (
        <>
          <section className="section-block selected-athlete-panel">
            <div className="selected-athlete-identity">
              <div className="athlete-avatar" aria-hidden="true">{user.name?.charAt(0)?.toUpperCase() || "A"}</div>
              <div>
                <p className="eyebrow">Athlete profile</p>
                <h2>{user.name}</h2>
                <p>{user.email}{user.phone ? ` / ${user.phone}` : ""}</p>
              </div>
            </div>
            <div className="selected-athlete-stats">
              <div>
                <span>Completed runs</span>
                <strong>{completedAttempts.length}</strong>
              </div>
              <div>
                <span>Latest Blue IQ</span>
                <strong>{latestBlueIq ?? "--"}{latestBlueIq != null && <small>/240</small>}</strong>
              </div>
              <div>
                <span>Latest activity</span>
                <strong className="date-stat">{latest ? formatDate(latest.session_date || latest.created_at) : "No runs yet"}</strong>
              </div>
            </div>
          </section>

          <section className="admin-layout profile-workspace">
            <UploadAnalysisPanel token={token} fixedUser={user} onCompleted={loadProfileActivity} />
            <section className="user-panel latest-run-panel">
              <p className="eyebrow">Latest analysis</p>
              {latest ? (
                <>
                  <div className="latest-run-heading">
                    <div>
                      <h2>Run {latest.run_number || latest.attempt_number || latest.id}</h2>
                      <p>{formatDate(latest.session_date || latest.created_at)}</p>
                    </div>
                    <ScorePill score={latestBlueIq} />
                  </div>
                  <div className="latest-score-row">
                    <span>Blue IQ</span>
                    <strong>{latestBlueIq}<small>/240</small></strong>
                  </div>
                  <div className="card-actions">
                    <button type="button" onClick={() => setGraphAttempt(latest)}>View graph</button>
                    {latestVideo && <a href={latestVideo} target="_blank" rel="noreferrer">View video</a>}
                    {latestReport && <a href={latestReport} target="_blank" rel="noreferrer">View report</a>}
                  </div>
                </>
              ) : (
                <p className="empty-copy">No completed analysis yet. Upload the athlete's first run to establish a baseline.</p>
              )}
            </section>
          </section>

          <PersonalBestPanel
            records={personalBests}
            title={`${user.name}'s personal bests`}
            description="Highest completed Blue IQ and pillar results, with the session and run where each record was set."
          />

          <section className="section-block">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Analysis history</p>
                <h2>Videos and reports</h2>
                <p>Open any completed run, PDF report, or detailed score graph for this athlete.</p>
              </div>
            </div>
            <div className="cards-grid">
              {displayedAttempts.length ? displayedAttempts.map((attempt) => (
                <AttemptCard
                  key={attempt.id}
                  attempt={attempt}
                  onViewAnalysis={setGraphAttempt}
                />
              )) : <EmptyState text="No completed runs for this athlete yet." />}
            </div>
            {hasMore && (
              <div className="load-more-row">
                <button className="secondary-button" onClick={() => setDisplayCount((count) => count + 4)}>
                  Load 4 More ({completedAttempts.length - displayCount} remaining)
                </button>
              </div>
            )}
          </section>
        </>
      ) : null}
      {graphAttempt && (
        <RunAnalysisGraph
          api={api}
          token={token}
          attempt={graphAttempt}
          athleteName={user?.name}
          onClose={() => setGraphAttempt(null)}
        />
      )}
    </main>
  );
}

function AdminDashboard() {
  const token = getToken("admin");
  const [users, setUsers] = useState([]);
  const [attempts, setAttempts] = useState([]);
  const [leaderboards, setLeaderboards] = useState({});
  const [adminView, setAdminView] = useState("upload");
  const [displayCount, setDisplayCount] = useState(4); // Show 4 initially
  const [clientSearch, setClientSearch] = useState("");
  const [error, setError] = useState("");
  const [graphAttempt, setGraphAttempt] = useState(null);

  async function loadData() {
    const [userRows, attemptRows, leaderboardRows] = await Promise.all([
      api("/admin/users?limit=1000", { token }),
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

  const clients = useMemo(() => users.filter((user) => user.role !== "admin"), [users]);
  const searchedClients = useMemo(() => {
    const ranked = clients
      .map((client) => ({ client, score: clientSearchScore(client, clientSearch) }))
      .filter(({ score }) => Number.isFinite(score));

    if (clientSearch.trim()) {
      ranked.sort((left, right) => (
        left.score - right.score
        || left.client.name.localeCompare(right.client.name)
      ));
    }
    return ranked.map(({ client }) => client);
  }, [clients, clientSearch]);
  const displayedAttempts = attempts.slice(0, displayCount);
  const hasMore = displayCount < attempts.length;

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
      <nav className="admin-view-tabs" aria-label="Admin workspace">
        <button
          type="button"
          className={adminView === "upload" ? "active" : ""}
          aria-current={adminView === "upload" ? "page" : undefined}
          onClick={() => setAdminView("upload")}
        >
          Upload
        </button>
        <button
          type="button"
          className={adminView === "leaderboards" ? "active" : ""}
          aria-current={adminView === "leaderboards" ? "page" : undefined}
          onClick={() => setAdminView("leaderboards")}
        >
          Leaderboards
        </button>
        <button
          type="button"
          className={adminView === "profiles" ? "active" : ""}
          aria-current={adminView === "profiles" ? "page" : undefined}
          onClick={() => setAdminView("profiles")}
        >
          Athlete Profiles
        </button>
      </nav>

      {adminView === "upload" && (
        <>
          <section className="upload-workspace">
            <UploadAnalysisPanel token={token} clients={clients} onCompleted={loadActivity} />
          </section>

          <section className="section-block">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Analysis history</p>
                <h2>Recent attempts</h2>
                <p>Open recently generated videos, reports, and run graphs across all clients.</p>
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
              <div className="load-more-row">
                <button className="secondary-button" onClick={() => setDisplayCount(prev => prev + 4)}>
                  Load 4 More ({attempts.length - displayCount} remaining)
                </button>
              </div>
            )}
          </section>
        </>
      )}

      {adminView === "leaderboards" && <LeaderboardPanel leaderboards={leaderboards} />}

      {adminView === "profiles" && (
        <section className="user-panel athlete-directory-panel">
          <div className="section-heading athlete-directory-heading">
            <div>
              <p className="eyebrow">Client directory</p>
              <h2>Athlete profiles</h2>
              <p>Search an athlete to review personal bests, run history, videos, and reports.</p>
            </div>
            <span className="directory-count">
              {clients.length} {clients.length === 1 ? "athlete" : "athletes"}
            </span>
          </div>
          <label className="client-search">
            <span>Search athletes</span>
            <input
              type="search"
              value={clientSearch}
              onChange={(event) => setClientSearch(event.target.value)}
              placeholder="Search by name or email"
              autoComplete="off"
            />
          </label>
          <div className="user-list athlete-directory-list">
            {searchedClients.map((user) => {
              const runCount = attempts.filter((attempt) => attempt.person_id === user.id).length;
              return (
                <button
                  type="button"
                  className="user-row"
                  key={user.id}
                  onClick={() => routeTo(`/admin/athletes/${user.id}`)}
                  aria-label={`Open ${user.name}'s athlete profile`}
                >
                  <span>{user.name}</span>
                  <small>{user.email}</small>
                  <b>{runCount} {runCount === 1 ? "run" : "runs"}</b>
                </button>
              );
            })}
            {!clients.length && <EmptyState text="No client accounts yet." />}
            {clients.length > 0 && !searchedClients.length && (
              <EmptyState text={`No athletes match "${clientSearch.trim()}".`} />
            )}
          </div>
        </section>
      )}
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
    const athleteProfileMatch = path.match(/^\/admin\/athletes\/(\d+)$/);
    if (path === "/signup") return <AuthForm type="signup" />;
    if (path === "/admin-login") return <AuthForm type="admin" />;
    if (athleteProfileMatch) return <AdminAthleteProfile userId={Number(athleteProfileMatch[1])} />;
    if (path === "/admin") return <AdminDashboard />;
    if (path === "/dashboard") return <ClientDashboard />;
    return <AuthForm type="login" />;
  }, [path]);
  return component;
}

export default App;
