import { useEffect, useMemo, useState } from "react";
import TrendsChart from "./TrendsChart";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

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

  const response = await fetch(`${API_BASE}${path}`, {
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

function AttemptCard({ attempt }) {
  const blueIq = Math.ceil(
    Number(attempt.blue_iq_score || (Number(attempt.pressure_score || 0) + Number(attempt.balance_score || 0) + Number(attempt.rotation_score || 0) + Number(attempt.edging_score || 0)) / 4)
  );
  const video = fileUrl(attempt.video_link || attempt.output_video_path);
  const report = fileUrl(attempt.report_path);
  return (
    <article className="attempt-card">
      <div className="attempt-top">
        <div>
          <p className="eyebrow">Run {attempt.attempt_number || attempt.id}</p>
          <h3>Blue IQ {blueIq}<span>/240</span></h3>
        </div>
        <ScorePill score={blueIq} />
      </div>
      <div className="metric-grid">
        <span>Pressure <b>{Math.ceil(attempt.pressure_score || 0)}</b></span>
        <span>Balance <b>{Math.ceil(attempt.balance_score || 0)}</b></span>
        <span>Rotation <b>{Math.ceil(attempt.rotation_score || 0)}</b></span>
        <span>Edging <b>{Math.ceil(attempt.edging_score || 0)}</b></span>
      </div>
      <div className="card-actions">
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
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) {
      routeTo("/login");
      return;
    }
    Promise.all([
      api("/auth/me", { token }),
      api("/me/attempts", { token }),
    ])
      .then(([profile, history]) => {
        setUser(profile);
        setAttempts(history);
      })
      .catch((err) => setError(err.message));
  }, [token]);

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
      {attempts.length >= 2 && <TrendsChart api={api} token={token} />}
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
  const [displayCount, setDisplayCount] = useState(4); // Show 4 initially
  const [selectedClient, setSelectedClient] = useState(""); // Track selected client filter
  const [error, setError] = useState("");
  const [uploadState, setUploadState] = useState("");

  async function loadData() {
    const [userRows, attemptRows] = await Promise.all([
      api("/admin/users", { token }),
      api("/admin/attempts", { token }),
    ]);
    setUsers(userRows);
    setAttempts(attemptRows);
  }

  useEffect(() => {
    if (!token) {
      routeTo("/admin-login");
      return;
    }
    loadData().catch((err) => setError(err.message));
  }, [token]);

  async function submitUpload(event) {
    event.preventDefault();
    const formElement = event.currentTarget; // Save form reference before async operations
    setError("");
    setUploadState("Processing video...");
    const form = new FormData(formElement);
    try {
      await api("/analyze-premium-overlay/", {
        method: "POST",
        token,
        body: form,
        isForm: true,
      });
      formElement.reset(); // Use saved reference instead of event.currentTarget
      setUploadState("Analysis complete.");
      await loadData();
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
          {displayedAttempts.length ? displayedAttempts.map((attempt) => <AttemptCard key={attempt.id} attempt={attempt} />) : <EmptyState text="No attempts have been generated yet." />}
        </div>
        {hasMore && (
          <div style={{ textAlign: 'center', marginTop: '24px' }}>
            <button className="secondary-button" onClick={() => setDisplayCount(prev => prev + 4)}>
              Load 4 More ({filteredAttempts.length - displayCount} remaining)
            </button>
          </div>
        )}
      </section>
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
