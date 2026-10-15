import { useEffect, useMemo, useRef, useState } from "react";
import TrendsChart from "./TrendsChart";
import { scoreBand } from "./scoreBands";
import RunAnalysisGraph from "./RunAnalysisGraph";
import logoImg from "../../services/bluerun.png";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const DASHBOARD_REFRESH_MS = 30000;
// The admin dashboard follows the light jobs list closely and reloads the
// heavier runs/leaderboards only when a job finishes (or as a slow fallback).
const JOBS_POLL_ACTIVE_MS = 3000;
const JOBS_POLL_IDLE_MS = 15000;
const ADMIN_ACTIVITY_FALLBACK_MS = 60000;
const SCORE_METRICS = [
  ["blue_iq", "Blue IQ"],
  ["pressure", "Pressure"],
  ["balance", "Balance"],
  ["rotation", "Rotation"],
  ["edging", "Parallelism"],
];

function normalizeSearchText(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function levenshteinDistance(s1, s2) {
  if (s1 === s2) return 0;
  if (!s1.length) return s2.length;
  if (!s2.length) return s1.length;
  const prev = [];
  for (let i = 0; i <= s2.length; i++) prev[i] = i;
  for (let i = 0; i < s1.length; i++) {
    const current = [i + 1];
    for (let j = 0; j < s2.length; j++) {
      const cost = s1[i] === s2[j] ? 0 : 1;
      current[j + 1] = Math.min(
        current[j] + 1,
        prev[j + 1] + 1,
        prev[j] + cost
      );
    }
    prev.splice(0, prev.length, ...current);
  }
  return prev[s2.length];
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

  if (termIndex === term.length) {
    return 10 + skippedCharacters + (firstMatch / 100);
  }

  if (term.length >= 3) {
    const tokens = field.split(/[\s@._-]+/).filter(Boolean);
    let minDistance = Number.POSITIVE_INFINITY;
    const maxAllowed = term.length <= 4 ? 1 : 2;
    for (const token of tokens) {
      if (Math.abs(token.length - term.length) <= maxAllowed) {
        const dist = levenshteinDistance(term, token);
        if (dist <= maxAllowed && dist < minDistance) {
          minDistance = dist;
        }
      }
    }
    if (Number.isFinite(minDistance)) {
      return 20 + minDistance * 5;
    }
  }

  return Number.POSITIVE_INFINITY;
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

// Clients ranked best-first by fuzzy name/email match; non-matches dropped.
function rankClients(clients, query, limit = Infinity) {
  if (!normalizeSearchText(query)) return [];
  return clients
    .map((client) => ({ client, score: clientSearchScore(client, query) }))
    .filter(({ score }) => Number.isFinite(score))
    .sort((a, b) => a.score - b.score || a.client.name.localeCompare(b.client.name))
    .slice(0, limit)
    .map(({ client }) => client);
}

// Re-runs `refresh` when the tab regains focus and every `intervalMs` while visible.
function useAutoRefresh(refresh, enabled = true, intervalMs = DASHBOARD_REFRESH_MS) {
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
    const timer = window.setInterval(runRefresh, intervalMs);

    window.addEventListener("focus", runRefresh);
    document.addEventListener("visibilitychange", refreshWhenVisible);

    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", runRefresh);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [enabled, intervalMs]);
}

function Logo() {
  return (
    <div className="brand">
      <img src={logoImg} alt="bluerun" className="brand-logo-img" />
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

function handleAuthExpired(isAdmin = false) {
  clearToken("admin");
  clearToken("client");
  const isCurrentlyAdmin = isAdmin || window.location.pathname.startsWith("/admin");
  routeTo(isCurrentlyAdmin ? "/admin-login?expired=1" : "/login?expired=1");
}

// Views capture the token when they render. After a password change the
// stored token is replaced (and the old one revoked), so always send the
// newest stored token for this portal; this also covers other open tabs.
function resolveToken(token) {
  if (!token) return token;
  if (token === getToken("admin") || token === getToken("client")) return token;
  const kind = window.location.pathname.startsWith("/admin") ? "admin" : "client";
  return getToken(kind) || token;
}

async function api(path, { method = "GET", token: requestedToken, body, isForm = false } = {}) {
  const token = resolveToken(requestedToken);
  const isFormData = isForm || (typeof FormData !== "undefined" && body instanceof FormData);
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body && !isFormData) headers["Content-Type"] = "application/json";

  const response = await fetch(`${API_BASE}/api${path}`, {
    method,
    headers,
    body: isFormData ? body : body ? JSON.stringify(body) : undefined,
  });

  // A 401 only means "session expired" when a session was sent; sign-in and
  // password-reset calls report their own errors.
  if (response.status === 401 && token) {
    handleAuthExpired();
    throw new Error("Your session has expired. Redirecting to login...");
  }

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

function getInitials(name) {
  if (!name) return "";
  const parts = String(name).trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
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
  return `PB ${comparison.personal_best_score} · ${comparison.points_below} points below`;
}

function formatComparisonStatus(comparison, isNewPb = false) {
  if (isNewPb || comparison?.status === "new_personal_best") {
    const gain = comparison?.previous_best?.score != null ? comparison.current_score - comparison.previous_best.score : 0;
    return {
      text: gain > 0 ? `New personal best +${gain}` : "New personal best",
      status: "pb",
    };
  }
  if (!comparison || comparison.status === "baseline") {
    return { text: "Baseline established", status: "baseline" };
  }
  if (comparison.status === "matches_personal_best") {
    return { text: "Matches personal best", status: "matches" };
  }
  const pbScore = comparison.personal_best_score;
  const pts = comparison.points_below;
  return {
    text: `PB ${pbScore} · ${pts} point${pts === 1 ? "" : "s"} below`,
    status: "below",
  };
}

function useAuthConfig() {
  const [config, setConfig] = useState({ email_verification_required: false, password_reset_available: false });
  useEffect(() => {
    let cancelled = false;
    api("/auth/config")
      .then((data) => !cancelled && setConfig(data))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);
  return config;
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
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const [codeStep, setCodeStep] = useState(false);
  const [code, setCode] = useState("");
  const authConfig = useAuthConfig();
  const isExpired = window.location.search.includes("expired=1");
  const isReset = window.location.search.includes("reset=1");

  const title = codeStep ? "Check your email" : isSignup ? "Create your account" : isAdmin ? "Admin sign in" : "Welcome back";

  async function requestSignupCode() {
    const result = await api("/auth/signup/request-code", {
      method: "POST",
      body: { name: form.name, email: form.email, phone: form.phone, password: form.password },
    });
    setNotice(result.message);
    setCodeStep(true);
  }

  async function resendCode() {
    setError("");
    setNotice("");
    setLoading(true);
    try {
      await requestSignupCode();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (isSignup && authConfig.email_verification_required && !codeStep) {
        await requestSignupCode();
        return;
      }
      let result;
      if (isSignup && codeStep) {
        result = await api("/auth/signup/verify", { method: "POST", body: { email: form.email, code } });
      } else {
        const endpoint = isSignup ? "/auth/signup" : "/auth/login";
        const payload = isSignup
          ? { name: form.name, email: form.email, phone: form.phone, password: form.password }
          : { email: form.email, password: form.password };
        result = await api(endpoint, { method: "POST", body: payload });
      }
      const expectedRole = isAdmin ? "admin" : "client";
      if (result.user.role !== expectedRole) {
        throw new Error(isAdmin ? "This account is not an admin account." : "This account can't sign in here.");
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
        {codeStep ? (
          <form onSubmit={submit}>
            {notice && <div className="notice">{notice}</div>}
            <label>Verification code</label>
            <input
              className="code-input"
              inputMode="numeric"
              autoComplete="one-time-code"
              maxLength={6}
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
              required
            />
            {error && <div className="alert">{error}</div>}
            <button className="primary-button" disabled={loading || code.length !== 6}>{loading ? "Working..." : "Verify and create account"}</button>
            <div className="auth-inline-actions">
              <button type="button" className="ghost-link" onClick={resendCode} disabled={loading}>Send a new code</button>
              <button type="button" className="ghost-link" onClick={() => { setCodeStep(false); setCode(""); setNotice(""); setError(""); }}>Change details</button>
            </div>
          </form>
        ) : (
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
          <div className="label-row">
            <label>Password</label>
            {!isSignup && authConfig.password_reset_available && (
              <button type="button" className="ghost-link small" onClick={() => routeTo(isAdmin ? "/forgot-password?admin=1" : "/forgot-password")}>
                Forgot password?
              </button>
            )}
          </div>
          <input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} minLength={8} required />
          {isReset && !error && (
            <div className="notice success">Your password has been reset. Please sign in.</div>
          )}
          {isExpired && !error && (
            <div className="alert" style={{ background: "rgba(245, 158, 11, 0.15)", borderColor: "#f59e0b", color: "#fbbf24" }}>
              Your session has expired. Please sign in again.
            </div>
          )}
          {error && <div className="alert">{error}</div>}
          <button className="primary-button" disabled={loading}>{loading ? "Working..." : title}</button>
        </form>
        )}
        {!isAdmin && !codeStep && (
          <p className="switch-link">
            {isSignup ? "Already have an account?" : "New to Bluerun?"}{" "}
            <button onClick={() => routeTo(isSignup ? "/login" : "/signup")}>{isSignup ? "Sign in" : "Create account"}</button>
          </p>
        )}
      </div>
    </AuthShell>
  );
}

function ForgotPasswordForm() {
  const isAdmin = window.location.search.includes("admin=1");
  const loginPath = isAdmin ? "/admin-login" : "/login";
  const [step, setStep] = useState("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);

  async function requestCode(event) {
    if (event) event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const result = await api("/auth/password/forgot", { method: "POST", body: { email } });
      setNotice(result.message);
      setStep("reset");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function resetPassword(event) {
    event.preventDefault();
    setError("");
    if (password !== confirm) {
      setError("The passwords do not match.");
      return;
    }
    setLoading(true);
    try {
      await api("/auth/password/reset", { method: "POST", body: { email, code, new_password: password } });
      routeTo(`${loginPath}?reset=1`);
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
        <p className="eyebrow">Account recovery</p>
        <h2>{step === "email" ? "Reset your password" : "Enter your code"}</h2>
        {step === "email" ? (
          <form onSubmit={requestCode}>
            <p className="form-intro">Enter the email you sign in with and we will send you a 6-digit code.</p>
            <label>Email</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" required />
            {error && <div className="alert">{error}</div>}
            <button className="primary-button" disabled={loading}>{loading ? "Sending..." : "Send reset code"}</button>
          </form>
        ) : (
          <form onSubmit={resetPassword}>
            {notice && <div className="notice">{notice}</div>}
            <label>Reset code</label>
            <input
              className="code-input"
              inputMode="numeric"
              autoComplete="one-time-code"
              maxLength={6}
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
              required
            />
            <PasswordField label="New password" value={password} autoComplete="new-password" onChange={setPassword} />
            <PasswordField label="Confirm new password" value={confirm} autoComplete="new-password" onChange={setConfirm} />
            {error && <div className="alert">{error}</div>}
            <button className="primary-button" disabled={loading || code.length !== 6}>{loading ? "Saving..." : "Set new password"}</button>
            <div className="auth-inline-actions">
              <button type="button" className="ghost-link" onClick={() => requestCode()} disabled={loading}>Send a new code</button>
            </div>
          </form>
        )}
        <button className="ghost-link" onClick={() => routeTo(loginPath)}>Back to sign in</button>
      </div>
    </AuthShell>
  );
}

function PasswordField({ label, value, onChange, autoComplete, minLength = 8 }) {
  const [visible, setVisible] = useState(false);
  return (
    <>
      <label>{label}</label>
      <div className="password-field">
        <input
          type={visible ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          minLength={minLength}
          autoComplete={autoComplete}
          required
        />
        <button type="button" className="password-toggle" onClick={() => setVisible((v) => !v)}>
          {visible ? "Hide" : "Show"}
        </button>
      </div>
    </>
  );
}

function ChangePasswordModal({ tokenKind, onClose }) {
  const [form, setForm] = useState({ current: "", next: "", confirm: "" });
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const onKey = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function submit(event) {
    event.preventDefault();
    setError("");
    if (form.next !== form.confirm) {
      setError("The new passwords do not match.");
      return;
    }
    setSaving(true);
    try {
      const result = await api("/auth/password/change", {
        method: "POST",
        token: getToken(tokenKind),
        body: { current_password: form.current, new_password: form.next },
      });
      // Changing the password signs out other sessions; keep this one.
      setToken(result.access_token, tokenKind);
      setDone(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-card" role="dialog" aria-modal="true" aria-labelledby="change-password-title">
        <div className="modal-header">
          <h2 id="change-password-title">Change password</h2>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">×</button>
        </div>
        {done ? (
          <>
            <div className="notice success">Your password has been updated. Other devices have been signed out.</div>
            <button type="button" className="primary-button" onClick={onClose}>Done</button>
          </>
        ) : (
          <form onSubmit={submit}>
            <PasswordField label="Current password" value={form.current} minLength={1} autoComplete="current-password" onChange={(v) => setForm({ ...form, current: v })} />
            <PasswordField label="New password" value={form.next} autoComplete="new-password" onChange={(v) => setForm({ ...form, next: v })} />
            <PasswordField label="Confirm new password" value={form.confirm} autoComplete="new-password" onChange={(v) => setForm({ ...form, confirm: v })} />
            <p className="field-hint">At least 8 characters.</p>
            {error && <div className="alert">{error}</div>}
            <button className="primary-button" disabled={saving}>{saving ? "Saving..." : "Update password"}</button>
          </form>
        )}
      </div>
    </div>
  );
}

function ProgressRing({ value, size = 38, stroke = 3.5, children }) {
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const known = Number.isFinite(value);
  const offset = known ? circumference * (1 - Math.min(100, Math.max(0, value)) / 100) : circumference * 0.7;
  return (
    <span className={`progress-ring ${known ? "" : "is-indeterminate"}`} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={radius} className="progress-ring-track" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          className="progress-ring-value"
          strokeWidth={stroke}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <span className="progress-ring-label">{children ?? (known ? `${Math.round(value)}%` : "")}</span>
    </span>
  );
}

function timeAgo(value) {
  if (!value) return "";
  const text = String(value);
  const date = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(text) ? text : `${text}Z`);
  const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}

function NotificationBell({ tokenKind }) {
  const isAdmin = tokenKind === "admin";
  const [open, setOpen] = useState(false);
  const [data, setData] = useState({ unread: 0, active_jobs: [], preparing_videos: [], items: [] });
  const [downloads, setDownloads] = useState({}); // id -> { percent, done, error }
  const ref = useRef(null);
  const openRef = useRef(false);

  async function load() {
    try {
      const result = await api("/notifications", { token: getToken(tokenKind) });
      setData({ preparing_videos: [], ...result });
    } catch {
      // keep the last state; the next poll retries
    }
  }

  const busy = data.active_jobs.length > 0
    || data.preparing_videos.length > 0
    || Object.values(downloads).some((d) => !d.done && !d.error);
  useEffect(() => {
    load();
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "hidden") load();
    }, busy ? 3000 : 20000);
    const refreshNow = () => load();
    window.addEventListener("bluerun:notifications-refresh", refreshNow);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("bluerun:notifications-refresh", refreshNow);
    };
  }, [busy, tokenKind]);

  // Closing the panel clears items that were already downloaded and seen.
  async function closePanel() {
    if (!openRef.current) return;
    openRef.current = false;
    setOpen(false);
    const finished = data.items.filter((item) => item.downloaded || downloads[item.id]?.done);
    if (!finished.length) return;
    setData((current) => ({ ...current, items: current.items.filter((item) => !finished.some((f) => f.id === item.id)) }));
    await Promise.all(finished.map((item) =>
      api(`/notifications/${item.id}/dismiss`, { method: "POST", token: getToken(tokenKind) }).catch(() => {})
    ));
  }

  function openPanel() {
    openRef.current = true;
    setOpen(true);
    if (data.unread) {
      setData((current) => ({ ...current, unread: 0, items: current.items.map((item) => ({ ...item, read: true })) }));
      api("/notifications/read-all", { method: "POST", token: getToken(tokenKind) }).catch(() => {});
    }
  }

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (event) => {
      if (ref.current && !ref.current.contains(event.target)) closePanel();
    };
    const onKey = (event) => event.key === "Escape" && closePanel();
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open, data, downloads]);

  async function download(item) {
    setDownloads((current) => ({ ...current, [item.id]: { percent: 0 } }));
    try {
      const isVersion = item.kind === "video_ready" && item.video_format;
      await downloadAttemptVideo(item.analysis_id, isVersion ? item.video_mode : null, isVersion ? item.video_format : "original", tokenKind, {
        onProgress: ({ received, total }) => setDownloads((current) => ({
          ...current,
          [item.id]: { percent: total ? Math.round((received / total) * 100) : null },
        })),
      });
      setDownloads((current) => ({ ...current, [item.id]: { percent: 100, done: true } }));
      api(`/notifications/${item.id}/downloaded`, { method: "POST", token: getToken(tokenKind) }).catch(() => {});
    } catch (err) {
      setDownloads((current) => ({ ...current, [item.id]: { error: err.message || "Download failed" } }));
    }
  }

  async function dismiss(item) {
    setData((current) => ({ ...current, items: current.items.filter((other) => other.id !== item.id) }));
    api(`/notifications/${item.id}/dismiss`, { method: "POST", token: getToken(tokenKind) }).catch(() => {});
  }

  async function clearAll() {
    setData((current) => ({ ...current, items: [], unread: 0 }));
    api("/notifications/clear", { method: "POST", token: getToken(tokenKind) }).catch(() => {});
  }

  const activeDownload = Object.values(downloads).find((d) => !d.done && !d.error);
  const badge = data.unread;

  return (
    <div className="notification-bell" ref={ref}>
      <button
        type="button"
        className={`bell-trigger ${busy ? "is-busy" : ""}`}
        aria-label={badge ? `Notifications, ${badge} new` : "Notifications"}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => (open ? closePanel() : openPanel())}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.73 21a2 2 0 0 1-3.46 0" />
        </svg>
        {badge > 0 && <span className="bell-badge">{badge > 9 ? "9+" : badge}</span>}
        {!badge && busy && <span className="bell-busy-dot" aria-hidden="true" />}
      </button>

      {open && (
        <div className="notification-panel" role="dialog" aria-label="Notifications">
          <div className="notification-panel-head">
            <strong>Notifications</strong>
            {data.items.length > 0 && <button type="button" onClick={clearAll}>Clear all</button>}
          </div>

          <div className="notification-list">
            {data.active_jobs.map((job) => (
              <div className="notification-item is-processing" key={job.job_id}>
                <ProgressRing value={job.status === "pending" ? NaN : job.progress} />
                <div className="notification-text">
                  <strong>{job.status === "pending" ? "Waiting to start…" : "Analysing a run…"}</strong>
                  <span>
                    {isAdmin
                      ? `${job.athlete_name || "Athlete"}${job.file_name ? ` · ${job.file_name}` : ""}`
                      : "Your coach uploaded a new run. We'll let you know when it's ready."}
                  </span>
                </div>
              </div>
            ))}

            {data.preparing_videos.map((video) => (
              <div className="notification-item is-processing" key={`${video.analysis_id}-${video.mode}-${video.format}`}>
                <ProgressRing value={video.progress || NaN} />
                <div className="notification-text">
                  <strong>Preparing your {video.label} video…</strong>
                  <span>{video.progress ? `${video.progress}% done. ` : ""}We'll let you know here when it's ready to download.</span>
                </div>
              </div>
            ))}

            {data.items.map((item) => {
              const state = downloads[item.id] || {};
              const downloaded = item.downloaded || state.done;
              const downloading = state.percent !== undefined && !state.done && !state.error;
              return (
                <div className={`notification-item ${item.read ? "" : "is-unread"}`} key={item.id}>
                  {downloading ? (
                    <ProgressRing value={state.percent ?? NaN} />
                  ) : (
                    <span className={`notification-emoji ${downloaded ? "is-done" : ""}`} aria-hidden="true">
                      {downloaded ? "✓" : item.emoji || "🔔"}
                    </span>
                  )}
                  <div className="notification-text">
                    {isAdmin && item.athlete_id ? (
                      <button type="button" className="notification-title-link" onClick={() => { closePanel(); routeTo(`/admin/athletes/${item.athlete_id}`); }}>
                        {item.title}
                      </button>
                    ) : (
                      <strong>{item.title}</strong>
                    )}
                    <span>
                      {downloaded
                        ? "Downloaded. This will clear when you close notifications."
                        : downloading
                          ? `Downloading… ${state.percent != null ? `${state.percent}%` : ""}`
                          : item.body}
                    </span>
                    {state.error && <span className="notification-error">{state.error}</span>}
                    <small>{timeAgo(item.created_at)}</small>
                  </div>
                  <div className="notification-actions">
                    {item.analysis_id && !downloaded && !downloading && (item.kind === "run_ready" || item.kind === "video_ready") && (
                      <button type="button" className="notification-download" onClick={() => download(item)}>
                        {state.error ? "Retry" : "Download"}
                      </button>
                    )}
                    {!downloading && (
                      <button type="button" className="notification-dismiss" aria-label="Dismiss" onClick={() => dismiss(item)}>×</button>
                    )}
                  </div>
                </div>
              );
            })}

            {!data.active_jobs.length && !data.preparing_videos.length && !data.items.length && (
              <div className="notification-empty">
                <span aria-hidden="true">✨</span>
                <strong>You're all caught up</strong>
                <small>{isAdmin ? "Finished runs and uploads will show up here." : "New runs from your coach will show up here."}</small>
              </div>
            )}
          </div>
          {activeDownload && <p className="notification-foot">Keep this tab open until the download finishes.</p>}
        </div>
      )}
    </div>
  );
}

function SettingsMenu({ tokenKind, onLogout }) {
  const [open, setOpen] = useState(false);
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) setOpen(false);
    };
    const onKey = (event) => event.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="settings-menu" ref={menuRef}>
      <button
        type="button"
        className="secondary-button app-header-logout-btn settings-trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
        <span>Settings</span>
      </button>
      {open && (
        <div className="settings-dropdown" role="menu">
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              setShowPasswordModal(true);
            }}
          >
            Change password
          </button>
          <button type="button" role="menuitem" className="danger" onClick={onLogout}>
            Log out
          </button>
        </div>
      )}
      {showPasswordModal && (
        <ChangePasswordModal tokenKind={tokenKind} onClose={() => setShowPasswordModal(false)} />
      )}
    </div>
  );
}

function AppHeader({ title, subtitle, role, onLogout, tokenKind = "admin" }) {
  return (
    <header className="app-header">
      <div className="app-header-left">
        <Logo />
        <div className="app-header-text">
          <p className="app-header-eyebrow eyebrow">{role ? role.toUpperCase() : "ADMIN DASHBOARD"}</p>
          <h1 className="app-header-title">{title || "BLUERUN PORTAL"}</h1>
          <p className="app-header-subtitle">{subtitle || "Manage clients, process runs, and track performance across all athletes."}</p>
        </div>
      </div>
      <div className="header-actions">
        <NotificationBell tokenKind={tokenKind} />
        <SettingsMenu tokenKind={tokenKind} onLogout={onLogout} />
      </div>
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

const DOWNLOAD_FORMATS = [
  { key: "original", label: "Original", detail: "16:9 with score panel" },
  { key: "vertical", label: "Vertical", detail: "9:16 for Reels, TikTok, Stories" },
  { key: "square", label: "Square", detail: "1:1 for Instagram & Facebook feed" },
];

const DOWNLOAD_MODES = [
  { key: "athlete", label: "Athlete", hint: "Clean footage" },
  { key: "coach", label: "Coach", hint: "With technique lines" },
];

// Keep in step with MAX_NOTE_CHARS in services/coach_notes.py.
const NOTE_MAX_CHARS = 280;

const NOTE_SOURCE_LABELS = {
  ai: "AI draft",
  template: "Auto draft",
  coach: "Edited by coach",
};

function saveWithLink(href, filename) {
  const link = document.createElement("a");
  link.href = href;
  link.download = filename || "";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function formatMegabytes(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(bytes >= 100 * 1024 * 1024 ? 0 : 1)} MB`;
}

// Downloads the file in the page so progress can be shown, then saves it.
// If the file host does not allow this (e.g. S3 without a CORS GET rule),
// it falls back to a normal browser download. Returns { filename, inApp }.
async function downloadAttemptVideo(attemptId, mode, format, tokenKind, { onProgress, signal } = {}) {
  const query = new URLSearchParams({ format });
  if (mode) query.set("mode", mode);
  const data = await api(`/videos/${attemptId}/download-url?${query.toString()}`, { token: getToken(tokenKind) });
  const url = String(data.download_url).startsWith("http") ? data.download_url : `${API_BASE}${data.download_url}`;
  const filename = data.filename || "bluerun_video.mp4";

  let response;
  try {
    response = await fetch(url, { signal });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    saveWithLink(url, filename);
    return { filename, inApp: false };
  }
  if (!response.ok || !response.body) {
    saveWithLink(url, filename);
    return { filename, inApp: false };
  }

  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  onProgress?.({ received, total });
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.({ received, total });
  }
  const blobUrl = URL.createObjectURL(new Blob(chunks, { type: "video/mp4" }));
  saveWithLink(blobUrl, filename);
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
  return { filename, inApp: true };
}

function DownloadMenu({ attempt, tokenKind }) {
  const [status, setStatus] = useState({
    available: attempt.available_downloads || {},
    pending: attempt.pending_downloads || [],
    preparable: attempt.preparable_downloads || {},
    progress: attempt.download_progress || {},
  });
  const defaultMode = (status.available.coach || []).includes("original") ? "coach" : "athlete";
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState(defaultMode);
  const [busyFormat, setBusyFormat] = useState("");
  const [error, setError] = useState("");
  const [started, setStarted] = useState("");
  const [progress, setProgress] = useState(null); // { label, received, total }
  const abortRef = useRef(null);
  const menuRef = useRef(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Keep in sync with the card when the list refreshes.
  useEffect(() => {
    setStatus({
      available: attempt.available_downloads || {},
      pending: attempt.pending_downloads || [],
      preparable: attempt.preparable_downloads || {},
      progress: attempt.download_progress || {},
    });
  }, [attempt.available_downloads, attempt.pending_downloads, attempt.preparable_downloads, attempt.download_progress]);

  const pendingCount = status.pending.length;
  // While a version is being made, check this run every few seconds.
  useEffect(() => {
    if (!pendingCount) return undefined;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const data = await api(`/videos/${attempt.id}/download-status`, { token: getToken(tokenKind) });
        if (!cancelled) {
          setStatus({
            available: data.available_downloads || {},
            pending: data.pending_downloads || [],
            preparable: data.preparable_downloads || {},
            progress: data.download_progress || {},
          });
        }
      } catch {
        // try again on the next tick
      }
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [pendingCount, attempt.id, tokenKind]);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) setOpen(false);
    };
    const onKey = (event) => event.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => {
    if (!started) return undefined;
    const timer = window.setTimeout(() => {
      setStarted("");
      setOpen(false);
    }, 3500);
    return () => window.clearTimeout(timer);
  }, [started]);

  async function download(format) {
    setError("");
    setStarted("");
    setBusyFormat(format);
    const controller = new AbortController();
    abortRef.current = controller;
    const label = `${DOWNLOAD_MODES.find((m) => m.key === mode)?.label} · ${DOWNLOAD_FORMATS.find((f) => f.key === format)?.label}`;
    try {
      const result = await downloadAttemptVideo(attempt.id, mode, format, tokenKind, {
        signal: controller.signal,
        onProgress: ({ received, total }) => setProgress({ label, received, total }),
      });
      setStarted(result.inApp ? `Saved ${result.filename}.` : `Download started. Check your browser's downloads for ${result.filename}.`);
    } catch (err) {
      if (err.name === "AbortError") {
        setError("Download cancelled.");
      } else {
        setError(err.message || "Download failed");
      }
    } finally {
      abortRef.current = null;
      setProgress(null);
      setBusyFormat("");
    }
  }

  const percent = progress?.total ? Math.min(100, Math.round((progress.received / progress.total) * 100)) : null;

  async function prepare(format) {
    setError("");
    setBusyFormat(format);
    try {
      await api(`/videos/${attempt.id}/prepare?mode=${mode}&format=${format}`, { method: "POST", token: getToken(tokenKind) });
      // Let the notification bell show it right away.
      window.dispatchEvent(new Event("bluerun:notifications-refresh"));
      setStatus((current) => ({
        ...current,
        pending: [...new Set([...current.pending, `${mode}:${format}`])],
        preparable: {
          ...current.preparable,
          [mode]: (current.preparable[mode] || []).filter((item) => item !== format),
        },
      }));
    } catch (err) {
      setError(err.message || "Could not start this version");
    } finally {
      setBusyFormat("");
    }
  }

  function statusFor(format) {
    if ((status.available[mode] || []).includes(format.key)) return { action: "download", text: format.detail };
    if (status.pending.includes(`${mode}:${format.key}`)) {
      const percent = status.progress?.[`${mode}:${format.key}`];
      return { action: null, text: percent ? `Preparing… ${percent}%` : "Preparing… usually about a minute", pending: true };
    }
    if ((status.preparable[mode] || []).includes(format.key)) return { action: "prepare", text: "Not made yet · click to create it" };
    return { action: null, text: "Not available for this run" };
  }

  return (
    <div className="download-menu" ref={menuRef}>
      <button
        type="button"
        className="btn-action-2x2 action-download-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        title="Download this run's video"
        style={progress ? { "--download-progress": `${percent ?? 0}%` } : undefined}
        data-downloading={progress ? "true" : undefined}
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
        <span>{progress ? (percent != null ? `${percent}%` : "Downloading…") : "Download"}</span>
      </button>
      {open && (
        <div className="download-dropdown" role="menu">
          <p className="download-dropdown-title">Download video</p>
          <div className="download-mode-toggle" role="radiogroup" aria-label="Video style">
            {DOWNLOAD_MODES.map((option) => (
              <button
                key={option.key}
                type="button"
                role="radio"
                aria-checked={mode === option.key}
                className={mode === option.key ? "is-active" : ""}
                onClick={() => {
                  setMode(option.key);
                  setError("");
                }}
              >
                <strong>{option.label}</strong>
                <small>{option.hint}</small>
              </button>
            ))}
          </div>
          {DOWNLOAD_FORMATS.map((format) => {
            const state = statusFor(format);
            return (
              <button
                key={format.key}
                type="button"
                role="menuitem"
                className={`download-option ${state.action === "prepare" ? "is-preparable" : ""}`}
                disabled={!state.action || Boolean(busyFormat)}
                onClick={() => (state.action === "download" ? download(format.key) : prepare(format.key))}
              >
                <span className={`format-glyph format-${format.key}`} aria-hidden="true" />
                <span className="download-option-text">
                  <strong>
                    {busyFormat === format.key
                      ? (state.action === "prepare" ? "Starting..." : "Getting link...")
                      : format.label}
                  </strong>
                  <small className={state.pending ? "is-pending" : ""}>
                    {state.pending && <span className="mini-spinner" aria-hidden="true" />}
                    {state.text}
                  </small>
                </span>
                {state.action === "prepare" && <span className="download-option-badge">Create</span>}
              </button>
            );
          })}
          {progress && (
            <div className="download-progress" role="status" aria-live="polite">
              <div className="download-progress-head">
                <span>Downloading {progress.label}</span>
                <button type="button" onClick={() => abortRef.current?.abort()}>Cancel</button>
              </div>
              <div className={`download-progress-track ${percent == null ? "is-indeterminate" : ""}`}>
                <div style={{ width: percent == null ? "35%" : `${percent}%` }} />
              </div>
              <small>
                {formatMegabytes(progress.received)}
                {progress.total ? ` of ${formatMegabytes(progress.total)} · ${percent}%` : ""}
              </small>
            </div>
          )}
          {started && <p className="download-started">{started}</p>}
          {error && <p className="download-error">{error}</p>}
        </div>
      )}
    </div>
  );
}

function AttemptCard({
  attempt,
  clientName: clientNameProp,
  onViewAnalysis,
  onArchive,
  onRestore,
  onDelete,
  onUpdateNotes,
  isAdmin = false,
}) {
  const [notesText, setNotesText] = useState(attempt.instructor_notes || "");
  const [savingNotes, setSavingNotes] = useState(false);
  const [showNotesBox, setShowNotesBox] = useState(false);

  useEffect(() => {
    setNotesText(attempt.instructor_notes || "");
  }, [attempt.instructor_notes]);

  const handleSaveNotes = async () => {
    setSavingNotes(true);
    try {
      if (onUpdateNotes) {
        await onUpdateNotes(attempt.id, notesText);
      } else {
        const token = getToken("admin");
        const updated = await api(`/admin/attempts/${attempt.id}/notes`, {
          method: "PATCH",
          token,
          body: { notes: notesText },
        });
        attempt.instructor_notes = notesText;
        attempt.notes_source = updated?.notes_source ?? null;
        if (updated && updated.report_path) {
          attempt.report_path = updated.report_path;
        }
      }
      setShowNotesBox(false);
    } catch (err) {
      alert(err.message || "Failed to save note");
    } finally {
      setSavingNotes(false);
    }
  };

  const handleClearNotes = async () => {
    setNotesText("");
    setSavingNotes(true);
    try {
      if (onUpdateNotes) {
        await onUpdateNotes(attempt.id, "");
      } else {
        const token = getToken("admin");
        const updated = await api(`/admin/attempts/${attempt.id}/notes`, {
          method: "PATCH",
          token,
          body: { notes: "" },
        });
        attempt.instructor_notes = "";
        attempt.notes_source = null;
        if (updated && updated.report_path) {
          attempt.report_path = updated.report_path;
        }
      }
      setShowNotesBox(false);
    } catch (err) {
      alert(err.message || "Failed to clear note");
    } finally {
      setSavingNotes(false);
    }
  };

  const blueIq = Math.ceil(
    Number(attempt.blue_iq_score || (Number(attempt.pressure_score || 0) + Number(attempt.balance_score || 0) + Number(attempt.rotation_score || 0) + Number(attempt.edging_score || 0)) / 4)
  );
  const [, tierKey] = scoreBand(blueIq);
  const video = fileUrl(attempt.video_link || attempt.output_video_path);
  const rawReportUrl = fileUrl(attempt.report_path);
  const report = rawReportUrl
    ? `${rawReportUrl}${rawReportUrl.includes("?") ? "&" : "?"}t=${new Date(attempt.updated_at || Date.now()).getTime()}`
    : null;
  const comparisons = attempt.personal_best_comparisons || {};
  const newPersonalBests = new Set(attempt.new_personal_bests || []);
  const runNumber = attempt.run_number || attempt.attempt_number || attempt.id;
  const sessionLabel = attempt.session_number ? `Session ${attempt.session_number}` : "Session";
  const dateLabel = formatDate(attempt.session_date || attempt.created_at || attempt.timestamp);
  const clientName = clientNameProp || attempt.client_name || attempt.person_name || attempt.user_name || attempt.person?.name;
  const initials = getInitials(clientName);
  const blueIqComparison = formatComparisonStatus(comparisons.blue_iq, newPersonalBests.has("blue_iq"));
  const blueIqPercent = Math.min(100, Math.max(0, Math.round((blueIq / 240) * 100)));

  return (
    <article className={`attempt-card ${attempt.is_archived ? "is-archived" : ""}`}>
      {/* (a) Card Header */}
      <div className="attempt-card-header">
        <div className="attempt-card-identity">
          <div className="athlete-avatar-circle" title={clientName || "Athlete"}>
            {initials || (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            )}
          </div>
          <div className="attempt-athlete-meta">
            <strong className="attempt-athlete-name">{clientName || "Athlete"}</strong>
            <span className="attempt-session-tag">
              {sessionLabel} · Run {runNumber}
              {dateLabel && dateLabel !== "Date unavailable" ? ` · ${dateLabel}` : ""}
            </span>
          </div>
        </div>
        <div className="attempt-header-badges">
          {attempt.is_archived && <span className="attempt-archived-pill">Archived</span>}
          <ScorePill score={blueIq} />
        </div>
      </div>

      {/* (b) Inset Score Panel (Blue IQ) */}
      <div className="score-inset-panel">
        <div className="score-inset-top">
          <span className="score-inset-label">BLUE IQ SCORE</span>
          <div className="score-inset-value">
            <strong>{blueIq}</strong>
            <small>/240</small>
          </div>
        </div>
        <div className="score-bar-track">
          <div
            className={`score-bar-fill ${tierKey}`}
            style={{ width: `${blueIqPercent}%` }}
          />
        </div>
        <div className="score-inset-sub">
          <span className={`status-dot status-${blueIqComparison.status}`} />
          <span className="score-status-text">{blueIqComparison.text}</span>
        </div>
      </div>

      {/* (c) Four Metric Rows */}
      <div className="metric-rows-list">
        {SCORE_METRICS.slice(1).map(([key, label]) => {
          const scoreVal = Math.ceil(attempt[`${key}_score`] || 0);
          const comp = formatComparisonStatus(comparisons[key], newPersonalBests.has(key));
          const pct = Math.min(100, Math.max(0, Math.round((scoreVal / 240) * 100)));
          return (
            <div className="metric-item-row" key={key}>
              <div className="metric-row-header">
                <div className="metric-row-info">
                  <span className="metric-row-name">{label}</span>
                  <small className={`metric-row-sub ${comp.status === "pb" ? "is-new-pb" : ""}`}>
                    {comp.text}
                  </small>
                </div>
                <div className="metric-row-val">
                  <strong>{scoreVal}</strong>
                  <small>/240</small>
                </div>
              </div>
              <div className="metric-bar-track">
                <div className="metric-bar-fill" style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* (e) Coach's note: read view for everyone, inline editor for admins */}
      {showNotesBox && isAdmin ? (
        <div className="coach-note-editor">
          <div className="coach-note-head">
            <span className="coach-note-icon" aria-hidden="true">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20h9" />
                <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
              </svg>
            </span>
            <div className="coach-note-heading">
              <strong>Coach's note</strong>
              <small>Shown on the client's card and PDF report</small>
            </div>
            {attempt.notes_source && NOTE_SOURCE_LABELS[attempt.notes_source] && (
              <span className={`notes-source-tag source-${attempt.notes_source}`}>{NOTE_SOURCE_LABELS[attempt.notes_source]}</span>
            )}
            <button
              type="button"
              className="coach-note-close"
              aria-label="Close note"
              title="Close"
              disabled={savingNotes}
              onClick={() => {
                setNotesText(attempt.instructor_notes || "");
                setShowNotesBox(false);
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <textarea
            className="coach-note-textarea"
            placeholder="Write a short, encouraging note for this run..."
            value={notesText}
            onChange={(e) => setNotesText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setNotesText(attempt.instructor_notes || "");
                setShowNotesBox(false);
              }
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && notesText !== (attempt.instructor_notes || "")) {
                handleSaveNotes();
              }
            }}
            rows={4}
            maxLength={NOTE_MAX_CHARS}
            autoFocus
            disabled={savingNotes}
          />
          <div className="coach-note-meter">
            <span>Saving updates the PDF report. Your edit replaces the automatic note.</span>
            <span className={`coach-note-count ${notesText.length >= NOTE_MAX_CHARS ? "at-limit" : ""}`}>
              {notesText.length}/{NOTE_MAX_CHARS}
            </span>
          </div>
          <div className="coach-note-progress" aria-hidden="true">
            <div style={{ width: `${Math.min(100, (notesText.length / NOTE_MAX_CHARS) * 100)}%` }} />
          </div>
          <div className="coach-note-actions">
            {attempt.instructor_notes && (
              <button type="button" className="coach-note-remove" disabled={savingNotes} onClick={handleClearNotes}>
                Remove note
              </button>
            )}
            <span className="coach-note-actions-spacer" />
            <button
              type="button"
              className="coach-note-cancel"
              disabled={savingNotes}
              onClick={() => {
                setNotesText(attempt.instructor_notes || "");
                setShowNotesBox(false);
              }}
            >
              Cancel
            </button>
            <button
              type="button"
              className="coach-note-save"
              disabled={savingNotes || !notesText.trim() || notesText === (attempt.instructor_notes || "")}
              onClick={handleSaveNotes}
            >
              {savingNotes ? (
                <>
                  <span className="coach-note-spinner" aria-hidden="true" />
                  Updating PDF...
                </>
              ) : (
                "Save"
              )}
            </button>
          </div>
        </div>
      ) : !isAdmin && attempt.instructor_notes ? (
        <div className="coach-note-readonly">
          <div className="coach-note-readonly-head">
            <span className="coach-note-label">Coach's note</span>
          </div>
          <p>{attempt.instructor_notes}</p>
        </div>
      ) : null}

      {/* (d) Card Actions (Pinned 2x2 grid + footer row) */}
      <div className="card-actions-wrapper">
        <div className="card-actions-2x2">
          {report ? (
            <a href={report} target="_blank" rel="noreferrer" className="btn-action-2x2" title="View PDF report">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
              <span>Report</span>
            </a>
          ) : (
            <span className="btn-action-2x2 is-disabled" title="No report available">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <span>Report</span>
            </span>
          )}

          {video ? (
            <a href={video} target="_blank" rel="noreferrer" className="btn-action-2x2" title="View video with overlay">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              <span>Video</span>
            </a>
          ) : (
            <span className="btn-action-2x2 is-disabled" title="No video available">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              <span>Video</span>
            </span>
          )}

          {onViewAnalysis ? (
            <button
              type="button"
              className="btn-action-2x2"
              onClick={() => onViewAnalysis(attempt)}
              title="View run analysis graph"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="20" x2="18" y2="10" />
                <line x1="12" y1="20" x2="12" y2="4" />
                <line x1="6" y1="20" x2="6" y2="14" />
              </svg>
              <span>Graph</span>
            </button>
          ) : (
            <span className="btn-action-2x2 is-disabled">
              <span>Graph</span>
            </span>
          )}

          {video ? (
            <DownloadMenu attempt={attempt} tokenKind={isAdmin ? "admin" : "client"} />
          ) : (
            <span className="btn-action-2x2 is-disabled" title="No download available">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              <span>Download</span>
            </span>
          )}
        </div>

        <div className="card-actions-footer">
          {isAdmin ? (
            <button
              type="button"
              className="btn-footer-ghost action-notes-toggle"
              onClick={() => {
                if (showNotesBox) setNotesText(attempt.instructor_notes || "");
                setShowNotesBox((prev) => !prev);
              }}
            >
              {showNotesBox ? "Hide note" : attempt.instructor_notes ? "Edit note" : "Add note"}
            </button>
          ) : (
            <span />
          )}

          <div className="card-footer-right">
            {onRestore && attempt.is_archived && (
              <button
                type="button"
                className="btn-footer-ghost action-restore"
                onClick={() => onRestore(attempt)}
              >
                Restore
              </button>
            )}
            {onArchive && !attempt.is_archived && (
              <button
                type="button"
                className="btn-footer-ghost action-archive"
                onClick={() => onArchive(attempt)}
              >
                Archive
              </button>
            )}
            {onDelete && (
              <button
                type="button"
                className="btn-footer-ghost btn-danger-ghost action-delete"
                onClick={() => onDelete(attempt)}
              >
                Delete
              </button>
            )}
          </div>
        </div>
      </div>
    </article>
  );
}

function extractDriveId(linkOrId) {
  if (!linkOrId) return null;
  const str = linkOrId.trim();
  const matchFile = str.match(/\/file\/d\/([a-zA-Z0-9_-]+)/);
  if (matchFile) return matchFile[1];
  const matchId = str.match(/[?&]id=([a-zA-Z0-9_-]+)/);
  if (matchId) return matchId[1];
  const matchDirect = str.match(/\/d\/([a-zA-Z0-9_-]+)/);
  if (matchDirect) return matchDirect[1];
  if (/^[a-zA-Z0-9_-]{20,60}$/.test(str)) return str;
  return null;
}

function UploadAnalysisPanel({ token, clients = [], fixedUser = null, onCompleted, onJobQueued, onGoToQueue }) {
  const [uploadSource, setUploadSource] = useState("local"); // "local" | "drive"
  const [directUpload, setDirectUpload] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api("/config/uploads")
      .then((config) => !cancelled && setDirectUpload(Boolean(config.direct_s3)))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);
  const [uploadState, setUploadState] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0); // 0-100
  const [progressPhase, setProgressPhase] = useState(""); // 'uploading' | 'queued' | ''
  const [uploadError, setUploadError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [queuedJobInfo, setQueuedJobInfo] = useState(null);

  // File drop & selection state
  const [selectedFileName, setSelectedFileName] = useState("");
  const [selectedFileSize, setSelectedFileSize] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Video preview & orientation state
  const [previewUrl, setPreviewUrl] = useState("");
  const [videoRotation, setVideoRotation] = useState(""); // "" (auto), "-90", "90", "180", "0"
  const [isPortraitDetected, setIsPortraitDetected] = useState(false);

  function handleLocalFileSelect(e) {
    const file = e.target.files?.[0];
    if (!file) {
      setSelectedFileName("");
      setSelectedFileSize("");
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl("");
      setVideoRotation("");
      setIsPortraitDetected(false);
      return;
    }
    setSelectedFileName(file.name);
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    setSelectedFileSize(`${mb} MB`);
    const url = URL.createObjectURL(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(url);

    const v = document.createElement("video");
    v.preload = "metadata";
    v.src = url;
    v.onloadedmetadata = () => {
      const isPortrait = v.videoHeight > v.videoWidth;
      setIsPortraitDetected(isPortrait);
      if (isPortrait) {
        setVideoRotation("-90");
      } else {
        setVideoRotation("");
      }
    };
  }

  function cycleRotation() {
    const cycle = {
      "": "-90",
      "-90": "180",
      "180": "90",
      "90": "0",
      "0": "",
    };
    setVideoRotation(cycle[videoRotation] ?? "");
  }

  // Google Drive state
  const [driveFile, setDriveFile] = useState(null); // { id, name, size, oauthToken }
  const [driveLinkInput, setDriveLinkInput] = useState("");
  const [googleConfig, setGoogleConfig] = useState({ client_id: "", api_key: "", app_id: "" });
  const [isPickerLoading, setIsPickerLoading] = useState(false);
  const [pickerError, setPickerError] = useState("");

  // Client fuzzy search state
  const [selectedClient, setSelectedClient] = useState(null);
  const [clientSearchText, setClientSearchText] = useState("");
  const [isClientDropdownOpen, setIsClientDropdownOpen] = useState(false);
  const [highlightedClientIndex, setHighlightedClientIndex] = useState(0);
  const clientDropdownRef = useRef(null);
  const clientInputRef = useRef(null);

  const matchedClients = useMemo(() => {
    if (!clients || !clients.length) return [];
    if (!clientSearchText.trim()) {
      return [...clients]
        .sort((a, b) => (a.name || "").localeCompare(b.name || ""))
        .slice(0, 30);
    }
    const ranked = clients
      .map((client) => ({ client, score: clientSearchScore(client, clientSearchText) }))
      .filter(({ score }) => Number.isFinite(score));

    ranked.sort((a, b) => a.score - b.score || (a.client.name || "").localeCompare(b.client.name || ""));
    return ranked.map(({ client }) => client);
  }, [clients, clientSearchText]);

  function handleClientKeyDown(e) {
    if (!isClientDropdownOpen) {
      if (e.key === "ArrowDown" || e.key === "Enter") {
        setIsClientDropdownOpen(true);
      }
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedClientIndex((prev) => (prev < matchedClients.length - 1 ? prev + 1 : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedClientIndex((prev) => (prev > 0 ? prev - 1 : matchedClients.length - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (matchedClients[highlightedClientIndex]) {
        setSelectedClient(matchedClients[highlightedClientIndex]);
        setIsClientDropdownOpen(false);
        setClientSearchText("");
      }
    } else if (e.key === "Escape") {
      setIsClientDropdownOpen(false);
    }
  }

  useEffect(() => {
    function handleClickOutside(event) {
      if (clientDropdownRef.current && !clientDropdownRef.current.contains(event.target)) {
        setIsClientDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    api("/config/google-drive")
      .then((cfg) => {
        if (cfg) setGoogleConfig(cfg);
      })
      .catch(() => {});
  }, []);

  function loadGoogleScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }
      const script = document.createElement("script");
      script.src = src;
      script.async = true;
      script.defer = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.body.appendChild(script);
    });
  }

  async function openGooglePicker() {
    setPickerError("");
    if (!googleConfig.client_id || !googleConfig.api_key) {
      setPickerError(
        "Google API credentials are not yet configured in server environment (GOOGLE_CLIENT_ID, GOOGLE_API_KEY). You can paste any shared Google Drive video link directly below instead!"
      );
      return;
    }

    setIsPickerLoading(true);
    try {
      await Promise.all([
        loadGoogleScript("https://apis.google.com/js/api.js"),
        loadGoogleScript("https://accounts.google.com/gsi/client"),
      ]);

      await new Promise((resolve) => window.gapi.load("picker", resolve));

      const tokenClient = window.google.accounts.oauth2.initTokenClient({
        client_id: googleConfig.client_id,
        scope: "https://www.googleapis.com/auth/drive.readonly",
        callback: (tokenResponse) => {
          setIsPickerLoading(false);
          if (tokenResponse.error !== undefined) {
            setPickerError(`Google authorization failed: ${tokenResponse.error}`);
            return;
          }

          const accessToken = tokenResponse.access_token;
          const view = new window.google.picker.View(window.google.picker.ViewId.DOCS);
          view.setMimeTypes("video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm");

          const picker = new window.google.picker.PickerBuilder()
            .setAppId(googleConfig.app_id || "")
            .setOAuthToken(accessToken)
            .addView(view)
            .addView(new window.google.picker.DocsUploadView())
            .setDeveloperKey(googleConfig.api_key)
            .setCallback((data) => {
              if (data.action === window.google.picker.Action.PICKED) {
                const doc = data.docs[0];
                setDriveFile({
                  id: doc.id,
                  name: doc.name,
                  size: doc.sizeBytes,
                  oauthToken: accessToken,
                });
                setDriveLinkInput("");
              }
            })
            .build();
          picker.setVisible(true);
        },
      });

      tokenClient.requestAccessToken({ prompt: "" });
    } catch (err) {
      setIsPickerLoading(false);
      setPickerError(`Could not open Google Picker: ${err.message}. You can paste a direct video link below.`);
    }
  }

  function handleAttachLink(e) {
    if (e) e.preventDefault();
    const id = extractDriveId(driveLinkInput);
    if (!id) {
      setPickerError("Could not find a valid Google Drive file ID in this link. Please check the URL.");
      return;
    }
    setPickerError("");
    setDriveFile({
      id: id,
      name: `Google Drive Video (${id.slice(0, 8)}...)`,
      oauthToken: null,
    });
  }

  function uploadWithProgress(url, formData) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      const currentToken = resolveToken(token);
      if (currentToken) {
        xhr.setRequestHeader("Authorization", `Bearer ${currentToken}`);
      }

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(percent);
          setUploadState(`Uploading video file (${percent}%)...`);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            resolve(data);
          } catch {
            resolve(xhr.responseText);
          }
        } else if (xhr.status === 401) {
          handleAuthExpired(true);
          reject(new Error("Your session has expired. Redirecting to login..."));
        } else {
          try {
            const errData = JSON.parse(xhr.responseText);
            const msg = errData.detail || "Upload failed";
            reject(new Error(Array.isArray(msg) ? msg.map(m => m.msg || m).join(", ") : msg));
          } catch {
            reject(new Error(xhr.responseText || `Upload failed with status ${xhr.status}`));
          }
        }
      };

      xhr.onerror = () => reject(new Error("Network error during upload. Please check your connection."));
      xhr.ontimeout = () => reject(new Error("Upload timed out. Please retry."));
      xhr.send(formData);
    });
  }

  // Direct upload: the browser sends the file straight to S3 with a signed form.
  function uploadToStorage(signed, file) {
    return new Promise((resolve, reject) => {
      const body = new FormData();
      Object.entries(signed.fields || {}).forEach(([name, value]) => body.append(name, value));
      body.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", signed.url);
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(percent);
          setUploadState(`Uploading video file (${percent}%)...`);
        }
      };
      xhr.onload = () => (xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new Error("The upload to storage failed. Please try again.")));
      xhr.onerror = () => reject(new Error("Network error during upload. Please check your connection."));
      xhr.send(body);
    });
  }

  async function submitDirectUpload(formElement, file, targetUserId) {
    const contentType = file.type || "video/mp4";
    const signed = await api("/uploads/presign", {
      method: "POST",
      token,
      body: { file_name: file.name, content_type: contentType, size: file.size },
    });
    await uploadToStorage(signed, file);
    setUploadState("Upload complete. Queueing analysis...");
    return api("/analyze-s3/", {
      method: "POST",
      token,
      body: {
        s3_key: signed.key,
        file_name: file.name,
        user_id: targetUserId,
        display_mode: formElement.querySelector('[name="display_mode"]')?.value || "athlete",
        report: Boolean(formElement.querySelector('[name="report"]')?.checked),
        rotation: videoRotation || null,
      },
    });
  }

  async function submitUpload(event) {
    event.preventDefault();
    const formElement = event.currentTarget;
    setUploadError("");
    setQueuedJobInfo(null);
    setUploadState("Initiating...");
    setUploadProgress(0);
    setProgressPhase("uploading");
    setIsSubmitting(true);

    try {
      const targetUserId = fixedUser ? fixedUser.id : (selectedClient ? selectedClient.id : formElement.querySelector('[name="user_id"]')?.value);
      if (!targetUserId) {
        throw new Error("Please search and select a client before submitting.");
      }

      let response;
      if (uploadSource === "local") {
        const file = formElement.querySelector('[name="file"]')?.files?.[0];
        if (!file) {
          throw new Error("Please choose or drop a video file before uploading.");
        }
        if (directUpload) {
          response = await submitDirectUpload(formElement, file, targetUserId);
        } else {
          const formData = new FormData(formElement);
          formData.set("user_id", targetUserId);
          if (!videoRotation) {
            formData.delete("rotation");
          }
          response = await uploadWithProgress(
            `${API_BASE}/api/analyze-premium-overlay/`,
            formData
          );
        }
      } else {
        const targetFileId = driveFile?.id || extractDriveId(driveLinkInput);
        if (!targetFileId) {
          throw new Error("Please select a video from Google Drive or paste a valid Google Drive video link.");
        }
        const formData = new FormData();
        const displayMode = formElement.querySelector('[name="display_mode"]').value;
        const report = formElement.querySelector('[name="report"]')?.checked ? "true" : "false";

        formData.append("user_id", targetUserId);
        formData.append("display_mode", displayMode);
        formData.append("report", report);
        formData.append("drive_file_id", targetFileId);
        if (driveFile?.name) formData.append("file_name", driveFile.name);
        if (driveFile?.oauthToken) formData.append("oauth_token", driveFile.oauthToken);
        if (videoRotation) formData.append("rotation", videoRotation);

        setUploadState("Queueing Google Drive video for download and analysis...");
        setUploadProgress(100);
        response = await api("/analyze-google-drive/", {
          method: "POST",
          token,
          body: formData,
          isForm: true,
        });
      }

      formElement.reset();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl("");
      setVideoRotation("");
      setIsPortraitDetected(false);
      setSelectedFileName("");
      setSelectedFileSize("");
      if (fileInputRef.current) fileInputRef.current.value = "";
      setDriveFile(null);
      setDriveLinkInput("");
      setSelectedClient(null);
      setClientSearchText("");

      if (response.job_id) {
        setUploadProgress(100);
        setProgressPhase("queued");
        setUploadState(response.message || "Video queued for analysis!");
        setQueuedJobInfo({ jobId: response.job_id });
        if (onJobQueued) onJobQueued(response.job_id);
      } else {
        setUploadState("Analysis complete.");
        setUploadProgress(100);
        setProgressPhase("");
        if (onCompleted) await onCompleted();
      }
    } catch (err) {
      setUploadState("");
      setUploadProgress(0);
      setProgressPhase("");
      setUploadError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form className="upload-panel" onSubmit={submitUpload}>
      <div className="upload-form-header">
        <p className="eyebrow upload-form-eyebrow">NEW ANALYSIS</p>
        <h2 className="upload-form-title">{fixedUser ? `Upload a run for ${fixedUser.name}` : "Upload client video"}</h2>
        <p className="upload-form-subtitle">Assign a run to an athlete, configure overlays, and queue for processing.</p>
      </div>

      <div className="upload-source-toggle">
        <button
          type="button"
          className={`upload-source-btn ${uploadSource === "local" ? "active" : ""}`}
          onClick={() => setUploadSource("local")}
          disabled={isSubmitting}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
          </svg>
          <span>Local Computer</span>
        </button>
        <button
          type="button"
          className={`upload-source-btn ${uploadSource === "drive" ? "active" : ""}`}
          onClick={() => setUploadSource("drive")}
          disabled={isSubmitting}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" />
          </svg>
          <span>Google Drive</span>
        </button>
      </div>

      {fixedUser ? (
        <>
          <input type="hidden" name="user_id" value={fixedUser.id} />
          <div className="fixed-upload-athlete">
            <span>{fixedUser.name}</span>
            <small>{fixedUser.email}</small>
          </div>
        </>
      ) : (
        <div className="client-select-field">
          <label className="field-label">ASSIGN TO CLIENT</label>
          {selectedClient ? (
            <div className="selected-client-card">
              <div className="selected-client-info">
                <div className="selected-client-avatar">
                  {getInitials(selectedClient.name) || "👤"}
                </div>
                <div className="selected-client-details">
                  <span className="selected-client-name">{selectedClient.name}</span>
                  <small className="selected-client-email">{selectedClient.email}</small>
                </div>
              </div>
              <button
                type="button"
                className="change-client-btn"
                onClick={() => {
                  setSelectedClient(null);
                  setClientSearchText("");
                  setIsClientDropdownOpen(true);
                  setTimeout(() => clientInputRef.current?.focus(), 50);
                }}
                disabled={isSubmitting}
              >
                Change
              </button>
              <input type="hidden" name="user_id" value={selectedClient.id} required />
            </div>
          ) : (
            <div className="client-search-wrapper" ref={clientDropdownRef}>
              <div className="client-search-input-box">
                <span className="client-search-icon">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                </span>
                <input
                  ref={clientInputRef}
                  type="text"
                  className="client-search-input"
                  placeholder="Search client by name or email..."
                  value={clientSearchText}
                  onChange={(e) => {
                    setClientSearchText(e.target.value);
                    setIsClientDropdownOpen(true);
                    setHighlightedClientIndex(0);
                  }}
                  onFocus={() => setIsClientDropdownOpen(true)}
                  onKeyDown={handleClientKeyDown}
                  disabled={isSubmitting}
                  autoComplete="off"
                />
                {clientSearchText && (
                  <button
                    type="button"
                    className="client-search-clear-btn"
                    onClick={() => {
                      setClientSearchText("");
                      clientInputRef.current?.focus();
                    }}
                    tabIndex={-1}
                  >
                    ✕
                  </button>
                )}
              </div>

              {isClientDropdownOpen && (
                <div className="client-search-dropdown">
                  {matchedClients.length > 0 ? (
                    matchedClients.map((client, idx) => (
                      <div
                        key={client.id}
                        className={`client-search-dropdown-item ${idx === highlightedClientIndex ? "active" : ""}`}
                        onMouseDown={(e) => {
                          e.preventDefault();
                          setSelectedClient(client);
                          setIsClientDropdownOpen(false);
                          setClientSearchText("");
                        }}
                        onMouseEnter={() => setHighlightedClientIndex(idx)}
                      >
                        <div className="client-dropdown-name">{client.name}</div>
                        <div className="client-dropdown-email">{client.email}</div>
                      </div>
                    ))
                  ) : (
                    <div className="client-search-empty">
                      No clients found matching "{clientSearchText.trim()}"
                    </div>
                  )}
                </div>
              )}
              <input type="hidden" name="user_id" value="" required />
            </div>
          )}
        </div>
      )}

      <div className="form-options-row">
        <div className="form-option-col">
          <label className="field-label" htmlFor="display-mode-select">DISPLAY MODE</label>
          <div className="custom-select-wrapper">
            <select
              id="display-mode-select"
              name="display_mode"
              defaultValue="coach"
              disabled={isSubmitting}
              className="form-control-select"
            >
              <option value="coach">Coach</option>
              <option value="athlete">Athlete</option>
            </select>
            <span className="select-chevron">▾</span>
          </div>
        </div>
        <div className="form-option-col">
          <label className="field-label">REPORT</label>
          <label className="boxed-checkbox-label">
            <input
              type="checkbox"
              name="report"
              value="true"
              defaultChecked
              disabled={isSubmitting}
              className="boxed-checkbox-native"
            />
            <span className="boxed-checkbox-box">
              <svg width="12" height="10" viewBox="0 0 12 10" fill="none">
                <path d="M1 5L4.5 8.5L11 1.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <span className="boxed-checkbox-text">Generate PDF report</span>
          </label>
        </div>
      </div>

      {uploadSource === "local" ? (
        <div className="upload-file-section">
          <label className="field-label">VIDEO FILE</label>
          <div
            className={`upload-dropzone ${isDragging ? "is-dragover" : ""} ${selectedFileName ? "has-file" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              if (e.dataTransfer.files?.length && fileInputRef.current) {
                fileInputRef.current.files = e.dataTransfer.files;
                handleLocalFileSelect({ target: fileInputRef.current });
              }
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              name="file"
              accept="video/*"
              disabled={isSubmitting}
              onChange={handleLocalFileSelect}
              className="visually-hidden-file-input"
            />
            <div className="dropzone-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            {selectedFileName ? (
              <div className="dropzone-selected-info">
                <p className="dropzone-filename">{selectedFileName}</p>
                <p className="dropzone-filesize">{selectedFileSize} · Click or drop another to replace</p>
              </div>
            ) : (
              <div className="dropzone-prompt">
                <p className="dropzone-prompt-text">Choose a video file or drag and drop here</p>
                <p className="dropzone-prompt-sub">MP4, MOV up to 500MB</p>
              </div>
            )}
          </div>

          {previewUrl && (
            <div className="video-upload-preview-card">
              <div className="video-preview-header">
                <span>
                  🎬 Video Preview
                  {isPortraitDetected && <span className="portrait-badge">📱 Portrait (Auto Horizontal)</span>}
                </span>
                <button
                  type="button"
                  className="ghost-button rotate-cycle-btn"
                  onClick={cycleRotation}
                  title="Cycle 90° rotation"
                  disabled={isSubmitting}
                >
                  ⟲ Rotate 90° {videoRotation ? `(${videoRotation}°)` : "(Auto)"}
                </button>
              </div>
              <div className="video-preview-wrapper">
                <video
                  src={previewUrl}
                  controls
                  style={{
                    transform: videoRotation ? `rotate(${videoRotation}deg)` : "none",
                    transition: "transform 0.25s ease",
                  }}
                />
              </div>
              <div className="rotation-control-row">
                <label>Orientation:</label>
                <select
                  name="rotation"
                  value={videoRotation}
                  onChange={(e) => setVideoRotation(e.target.value)}
                  disabled={isSubmitting}
                >
                  <option value="">Auto-Detect (Always Horizontal)</option>
                  <option value="-90">Rotate 90° CCW (Counter-Clockwise)</option>
                  <option value="90">Rotate 90° CW (Clockwise)</option>
                  <option value="180">Rotate 180° (Upside Down)</option>
                  <option value="0">Original (No Rotation)</option>
                </select>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="drive-picker-container">
          <label className="field-label">GOOGLE DRIVE VIDEO</label>
          {driveFile ? (
            <div className="selected-drive-card">
              <div className="selected-drive-icon">☁️</div>
              <div className="selected-drive-info">
                <strong>{driveFile.name}</strong>
                <small>
                  File ID: {driveFile.id}
                  {driveFile.size ? ` • ${(driveFile.size / (1024 * 1024)).toFixed(1)} MB` : ""}
                </small>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={() => { setDriveFile(null); setDriveLinkInput(""); }}
              >
                ✕ Change
              </button>
            </div>
          ) : (
            <div className="drive-selection-methods">
              <button
                type="button"
                className="secondary-button gdrive-browse-btn"
                onClick={openGooglePicker}
                disabled={isSubmitting || isPickerLoading}
              >
                {isPickerLoading ? "Opening Google..." : "🔍 Browse Google Drive"}
              </button>

              <div className="drive-or-divider">
                <span>OR PASTE DRIVE SHARE LINK</span>
              </div>

              <div className="drive-link-input-row">
                <input
                  type="text"
                  placeholder="https://drive.google.com/file/d/..."
                  value={driveLinkInput}
                  onChange={(e) => setDriveLinkInput(e.target.value)}
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  className="secondary-button inline-attach-btn"
                  onClick={handleAttachLink}
                  disabled={!driveLinkInput.trim() || isSubmitting}
                >
                  Attach
                </button>
              </div>
              {pickerError && <p className="picker-error-note">{pickerError}</p>}
            </div>
          )}
        </div>
      )}

      <button type="submit" className="primary-button upload-submit-btn" disabled={isSubmitting}>
        {isSubmitting
          ? uploadSource === "local"
            ? `Uploading (${uploadProgress}%)...`
            : "Queueing Drive Video..."
          : uploadSource === "local"
            ? "Upload to Queue"
            : "Queue Google Drive Video"}
      </button>

      {isSubmitting && uploadProgress > 0 && uploadSource === "local" && (
        <div className="upload-progress-container">
          <div className="upload-progress-track">
            <div className="upload-progress-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
          <div className="upload-progress-meta">
            <span>Transferring video to server</span>
            <span>{uploadProgress}%</span>
          </div>
        </div>
      )}

      {queuedJobInfo && (
        <div className="upload-queued-banner">
          <div className="queued-banner-content">
            <strong>Video queued for processing!</strong>
            <p>
              The video was registered and placed into the sequential queue.
              You can immediately upload another video, or monitor live progress in the <strong>Upload Progress</strong> tab.
            </p>
          </div>
          <div className="queued-banner-actions">
            {onGoToQueue && (
              <button type="button" className="secondary-button" onClick={onGoToQueue}>
                View Upload Progress ➔
              </button>
            )}
            <button type="button" className="ghost-button" onClick={() => setQueuedJobInfo(null)}>
              Dismiss
            </button>
          </div>
        </div>
      )}

      {uploadError && <div className="alert">{uploadError}</div>}
      {uploadState && !queuedJobInfo && <p className="status-text">{uploadState}</p>}
    </form>
  );
}

function UploadQueuePanel({ jobs = [], token, onRefresh, onViewAnalysis, onNavigateUpload }) {
  const [clearing, setClearing] = useState(false);

  const activeJobs = jobs
    .filter((j) => j.status === "processing")
    .sort((a, b) => new Date(a.started_at || a.created_at) - new Date(b.started_at || b.created_at));
  const queuedJobs = jobs
    .filter((j) => j.status === "pending")
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
  const completedJobs = jobs
    .filter((j) => j.status === "completed")
    .sort((a, b) => new Date(b.completed_at || b.created_at) - new Date(a.completed_at || a.created_at));
  const failedJobs = jobs
    .filter((j) => j.status === "failed")
    .sort((a, b) => new Date(b.completed_at || b.created_at) - new Date(a.completed_at || a.created_at));

  async function handleClearQueue() {
    const confirmed = window.confirm(
      "Are you sure you want to clear all active and queued videos from the processing queue?"
    );
    if (!confirmed) return;
    setClearing(true);
    try {
      await api("/jobs/clear-queue", { method: "POST", token });
      await onRefresh();
    } catch (err) {
      alert(`Failed to clear queue: ${err.message}`);
    } finally {
      setClearing(false);
    }
  }

  async function handleDeleteJob(jobId) {
    try {
      await api(`/jobs/${jobId}`, { method: "DELETE", token });
      await onRefresh();
    } catch (err) {
      alert(`Failed to remove job: ${err.message}`);
    }
  }

  return (
    <section className="queue-workspace">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Processing Engine</p>
          <h2>Upload & Processing Progress</h2>
          <p>
            Videos are analyzed sequentially to prevent server overload. When multiple videos are submitted, they queue in line.
          </p>
        </div>
        <div className="queue-heading-actions">
          {(activeJobs.length > 0 || queuedJobs.length > 0) && (
            <button
              type="button"
              className="danger-button inline-btn"
              onClick={handleClearQueue}
              disabled={clearing}
            >
              {clearing ? "Clearing..." : "Clear Queue"}
            </button>
          )}
          {onNavigateUpload && (
            <button type="button" className="primary-button inline-btn" onClick={onNavigateUpload}>
              + Upload Another Video
            </button>
          )}
          <button type="button" className="secondary-button inline-btn" onClick={onRefresh}>
            Refresh Queue
          </button>
        </div>
      </div>

      <div className="queue-overview-stats">
        <div className="queue-stat-card">
          <span>Active</span>
          <strong className={activeJobs.length > 0 ? "highlight-active" : ""}>{activeJobs.length}</strong>
        </div>
        <div className="queue-stat-card">
          <span>In Queue</span>
          <strong className={queuedJobs.length > 0 ? "highlight-queue" : ""}>{queuedJobs.length}</strong>
        </div>
        <div className="queue-stat-card">
          <span>Completed</span>
          <strong>{completedJobs.length}</strong>
        </div>
      </div>

      <div className="queue-cards-section">
        <h3 className="queue-section-title">Active & Queued Tasks</h3>
        {activeJobs.length === 0 && queuedJobs.length === 0 ? (
          <div className="empty-state queue-empty-card">
            <p>No videos currently processing. All uploaded runs have finished analysis.</p>
            {onNavigateUpload && (
              <button type="button" className="secondary-button" style={{ marginTop: '14px' }} onClick={onNavigateUpload}>
                Upload a client video
              </button>
            )}
          </div>
        ) : (
          <div className="queue-active-grid">
            {activeJobs.map((job) => (
              <div className="queue-card active-job-card" key={job.job_id}>
                <div className="queue-card-header">
                  <span className="queue-status-badge processing">
                    <span className="pulse-dot" /> Analyzing Video
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="queue-display-mode">{job.display_mode === "athlete" ? "Athlete Mode" : "Coach Mode"}</span>
                    <button
                      type="button"
                      className="ghost-button"
                      title="Cancel and remove this job"
                      onClick={() => handleDeleteJob(job.job_id)}
                    >
                      ✕
                    </button>
                  </div>
                </div>
                <div className="queue-card-body">
                  <h4>{job.file_name}</h4>
                  <p className="queue-client-name">
                    Client: <strong>{job.person_name || `Client #${job.person_id}`}</strong>
                  </p>
                  
                  <div className="queue-progress-bar-wrap">
                    <div className="queue-progress-bar-track">
                      <div
                        className="queue-progress-bar-fill animated"
                        style={{ width: `${Math.max(10, job.progress || 10)}%` }}
                      />
                    </div>
                    <div className="queue-progress-bar-labels">
                      <span>{job.progress <= 65 ? "Processing video frames..." : job.progress <= 80 ? "Uploading assets..." : "Finalizing analysis..."}</span>
                      <strong>{job.progress}%</strong>
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {queuedJobs.map((job, idx) => (
              <div className="queue-card queued-job-card" key={job.job_id}>
                <div className="queue-card-header">
                  <span className="queue-status-badge queued">
                    ⏳ #{idx + 1} in queue
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="queue-time-ago">{formatDate(job.created_at)}</span>
                    <button
                      type="button"
                      className="ghost-button"
                      title="Cancel this queued job"
                      onClick={() => handleDeleteJob(job.job_id)}
                    >
                      ✕ Cancel
                    </button>
                  </div>
                </div>
                <div className="queue-card-body">
                  <h4>{job.file_name}</h4>
                  <p className="queue-client-name">
                    Client: <strong>{job.person_name || `Client #${job.person_id}`}</strong>
                  </p>
                  <p className="queue-wait-note">
                    Waiting for current video to finish. Will begin automatically once the slot is free.
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {(completedJobs.length > 0 || failedJobs.length > 0) && (
        <div className="queue-history-section">
          <h3 className="queue-section-title">Recent Queue Activity</h3>
          <div className="queue-history-list">
            {failedJobs.map((job) => (
              <div className="queue-history-item failed" key={job.job_id}>
                <div className="queue-history-info">
                  <span className="queue-status-badge failed">Failed</span>
                  <strong>{job.file_name}</strong>
                  <span>Client: {job.person_name || `Client #${job.person_id}`}</span>
                  <small className="error-text">{job.error_message || "Analysis failed"}</small>
                </div>
                <div className="queue-history-meta">
                  <small>{formatDate(job.completed_at || job.created_at)}</small>
                  <button
                    type="button"
                    className="ghost-button"
                    title="Remove from history"
                    onClick={() => handleDeleteJob(job.job_id)}
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}

            {completedJobs.slice(0, 8).map((job) => (
              <div className="queue-history-item completed" key={job.job_id}>
                <div className="queue-history-info">
                  <span className="queue-status-badge completed">Completed</span>
                  <strong>{job.file_name}</strong>
                  <span>Client: {job.person_name || `Client #${job.person_id}`}</span>
                  <small>{job.display_mode === "athlete" ? "Athlete Mode" : "Coach Mode"}</small>
                </div>
                <div className="queue-history-meta">
                  <small>{formatDate(job.completed_at || job.created_at)}</small>
                  {job.video_analysis_id && onViewAnalysis && (
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => onViewAnalysis({ id: job.video_analysis_id })}
                    >
                      View Graph ➔
                    </button>
                  )}
                  <button
                    type="button"
                    className="ghost-button"
                    title="Remove from history"
                    onClick={() => handleDeleteJob(job.job_id)}
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function ClientDashboard() {
  const token = getToken("client");
  const [attempts, setAttempts] = useState([]);
  const [displayCount, setDisplayCount] = useState(12); // Show 12 initially to fill widescreen area
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
        tokenKind="client"
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
          {displayedAttempts.length ? displayedAttempts.map((attempt) => (
            <AttemptCard
              key={attempt.id}
              attempt={attempt}
              clientName={attempt.person_name || attempt.client_name || user?.name}
            />
          )) : <EmptyState text="No attempts have been uploaded yet." />}
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

function ClientSearchBox({ clients, query, onQueryChange, selected, onSelect, onClear }) {
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const boxRef = useRef(null);
  const inputRef = useRef(null);
  const suggestions = useMemo(() => rankClients(clients, query, 6), [clients, query]);

  useEffect(() => {
    setHighlight(0);
  }, [query]);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (event) => {
      if (boxRef.current && !boxRef.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  function choose(client) {
    onSelect(client);
    setOpen(false);
  }

  function onKeyDown(event) {
    if (event.key === "ArrowDown" && suggestions.length) {
      event.preventDefault();
      setOpen(true);
      setHighlight((h) => (h + 1) % suggestions.length);
    } else if (event.key === "ArrowUp" && suggestions.length) {
      event.preventDefault();
      setHighlight((h) => (h - 1 + suggestions.length) % suggestions.length);
    } else if (event.key === "Enter" && open && suggestions[highlight]) {
      event.preventDefault();
      choose(suggestions[highlight]);
    } else if (event.key === "Escape") {
      setOpen(false);
    } else if (event.key === "Backspace" && !query && selected) {
      onClear();
    }
  }

  const showList = open && !selected && query.trim() !== "";

  return (
    <div className="runs-filter-search client-combobox" ref={boxRef}>
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
      {selected ? (
        <div className="client-search-chip-row" onClick={() => inputRef.current?.focus()}>
          <span className="client-search-chip">
            <span className="client-search-chip-avatar">{getInitials(selected.name)}</span>
            <span className="client-search-chip-name">{selected.name}</span>
            <button type="button" onClick={onClear} aria-label={`Remove ${selected.name}`} title="Show all athletes">✕</button>
          </span>
          <input ref={inputRef} className="client-search-ghost-input" onKeyDown={onKeyDown} aria-label="Search athletes" readOnly />
        </div>
      ) : (
        <input
          ref={inputRef}
          type="text"
          placeholder="Search athlete name or email..."
          value={query}
          onChange={(e) => {
            onQueryChange(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          className="runs-search-input"
          role="combobox"
          aria-expanded={showList}
          aria-autocomplete="list"
          aria-controls="client-search-list"
        />
      )}
      {!selected && query && (
        <button type="button" className="runs-search-clear-btn" onClick={() => onQueryChange("")} title="Clear search">
          ✕
        </button>
      )}
      {showList && (
        <div className="client-search-list" id="client-search-list" role="listbox">
          {suggestions.length ? (
            <>
              <p className="client-search-list-title">Athletes</p>
              {suggestions.map((client, index) => (
                <button
                  key={client.id}
                  type="button"
                  role="option"
                  aria-selected={index === highlight}
                  className={`client-search-option ${index === highlight ? "is-active" : ""}`}
                  onMouseEnter={() => setHighlight(index)}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => choose(client)}
                >
                  <span className="client-search-avatar">{getInitials(client.name)}</span>
                  <span className="client-search-option-text">
                    <strong>{client.name}</strong>
                    {client.email && <small>{client.email}</small>}
                  </span>
                  {index === highlight && <kbd>Enter</kbd>}
                </button>
              ))}
            </>
          ) : (
            <p className="client-search-none">No athlete matches “{query.trim()}”</p>
          )}
        </div>
      )}
    </div>
  );
}

// Calendar days are handled as local "YYYY-MM-DD" keys; "" means all dates.
function dayKey(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function parseDayKey(key) {
  const [y, m, d] = key.split("-").map(Number);
  return new Date(y, m - 1, d);
}

function todayKey() {
  return dayKey(new Date());
}

function shiftDayKey(key, days) {
  const date = parseDayKey(key);
  date.setDate(date.getDate() + days);
  return dayKey(date);
}

function dayRangeIso(key) {
  const start = parseDayKey(key);
  const end = new Date(start);
  end.setDate(end.getDate() + 1);
  return [start.toISOString(), end.toISOString()];
}

function attemptDayKey(attempt) {
  const raw = attempt.created_at || attempt.timestamp || attempt.session_date;
  if (!raw) return "";
  // created_at is UTC without a zone marker; read it as UTC so the local day is right.
  const text = String(raw);
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return text;
  const parsed = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(text) ? text : `${text}Z`);
  return Number.isNaN(parsed.getTime()) ? "" : dayKey(parsed);
}

// "today", "yesterday" or "on Mon, Oct 6" for use inside a sentence.
function dayPhrase(key) {
  const label = dayLabel(key);
  return label === "Today" || label === "Yesterday" ? label.toLowerCase() : `on ${label}`;
}

function dayLabel(key) {
  if (!key) return "All dates";
  const today = todayKey();
  if (key === today) return "Today";
  if (key === shiftDayKey(today, -1)) return "Yesterday";
  const date = parseDayKey(key);
  const sameYear = date.getFullYear() === new Date().getFullYear();
  return date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    ...(sameYear ? {} : { year: "numeric" }),
  });
}

const WEEKDAY_LABELS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];

function DateNavigator({ value, onChange, allowAll = true }) {
  const today = todayKey();
  const [open, setOpen] = useState(false);
  const [viewMonth, setViewMonth] = useState(() => {
    const base = parseDayKey(value || today);
    return new Date(base.getFullYear(), base.getMonth(), 1);
  });
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const base = parseDayKey(value || today);
    setViewMonth(new Date(base.getFullYear(), base.getMonth(), 1));
    const onDown = (event) => {
      if (ref.current && !ref.current.contains(event.target)) setOpen(false);
    };
    const onKey = (event) => event.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const atToday = value === today;
  const goPrev = () => onChange(shiftDayKey(value || today, value ? -1 : 0));
  const goNext = () => {
    if (value && !atToday) onChange(shiftDayKey(value, 1));
  };

  const thisMonth = new Date(parseDayKey(today).getFullYear(), parseDayKey(today).getMonth(), 1);
  const canNextMonth = viewMonth < thisMonth;
  const leading = (viewMonth.getDay() + 6) % 7; // Monday-first grid
  const daysInMonth = new Date(viewMonth.getFullYear(), viewMonth.getMonth() + 1, 0).getDate();
  const cells = [
    ...Array.from({ length: leading }, () => null),
    ...Array.from({ length: daysInMonth }, (_, i) => dayKey(new Date(viewMonth.getFullYear(), viewMonth.getMonth(), i + 1))),
  ];

  function pick(key) {
    onChange(key);
    setOpen(false);
  }

  return (
    <div className="date-nav" ref={ref}>
      <button type="button" className="date-nav-step" onClick={goPrev} aria-label="Previous day" title="Previous day">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
      <button
        type="button"
        className={`date-nav-trigger ${open ? "is-open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="3" y="4" width="18" height="18" rx="2" />
          <line x1="16" y1="2" x2="16" y2="6" />
          <line x1="8" y1="2" x2="8" y2="6" />
          <line x1="3" y1="10" x2="21" y2="10" />
        </svg>
        <span>{dayLabel(value)}</span>
        <svg className="date-nav-caret" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      <button
        type="button"
        className="date-nav-step"
        onClick={goNext}
        disabled={!value || atToday}
        aria-label="Next day"
        title={atToday ? "You are on today" : "Next day"}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>

      {open && (
        <div className="date-nav-popover" role="dialog" aria-label="Choose a date">
          <div className="calendar-head">
            <button
              type="button"
              className="calendar-month-step"
              onClick={() => setViewMonth(new Date(viewMonth.getFullYear(), viewMonth.getMonth() - 1, 1))}
              aria-label="Previous month"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <polyline points="15 18 9 12 15 6" />
              </svg>
            </button>
            <strong>{viewMonth.toLocaleDateString(undefined, { month: "long", year: "numeric" })}</strong>
            <button
              type="button"
              className="calendar-month-step"
              onClick={() => setViewMonth(new Date(viewMonth.getFullYear(), viewMonth.getMonth() + 1, 1))}
              disabled={!canNextMonth}
              aria-label="Next month"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </button>
          </div>
          <div className="calendar-grid">
            {WEEKDAY_LABELS.map((label) => (
              <span key={label} className="calendar-weekday">{label}</span>
            ))}
            {cells.map((key, index) => key ? (
              <button
                key={key}
                type="button"
                className={`calendar-day ${key === value ? "is-selected" : ""} ${key === today ? "is-today" : ""}`}
                disabled={key > today}
                onClick={() => pick(key)}
                aria-pressed={key === value}
              >
                {Number(key.slice(8))}
              </button>
            ) : (
              <span key={`blank-${index}`} />
            ))}
          </div>
          <div className="calendar-foot">
            <button type="button" onClick={() => pick(today)}>Today</button>
            {allowAll && (
              <button type="button" className={!value ? "is-active" : ""} onClick={() => pick("")}>All dates</button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function UploadVideoModal({ token, user, onCompleted, onClose }) {
  useEffect(() => {
    const onKey = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="upload-modal" role="dialog" aria-modal="true" aria-label={`Upload a video for ${user.name}`}>
        <button type="button" className="modal-close upload-modal-close" onClick={onClose} aria-label="Close">×</button>
        <UploadAnalysisPanel token={token} fixedUser={user} onCompleted={onCompleted} />
        <p className="upload-modal-hint">You can close this window. Processing continues in the background and the run appears here when it is ready.</p>
      </div>
    </div>
  );
}

function AdminAthleteProfile({ userId }) {
  const token = getToken("admin");
  const [user, setUser] = useState(null);
  const [attempts, setAttempts] = useState([]);
  const [personalBests, setPersonalBests] = useState({});
  const [displayCount, setDisplayCount] = useState(12);
  const [selectedDay, setSelectedDay] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [graphAttempt, setGraphAttempt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadProfileActivity() {
    const [attemptRows, bestRows] = await Promise.all([
      api(`/admin/users/${userId}/attempts?limit=100&include_archived=true`, { token }),
      api(`/admin/users/${userId}/personal-bests`, { token }),
    ]);
    setAttempts(attemptRows);
    setPersonalBests(bestRows.personal_bests || {});
  }

  async function handleArchive(attempt) {
    try {
      await api(`/admin/attempts/${attempt.id}/archive`, { method: "POST", token });
      await loadProfileActivity();
    } catch (err) {
      alert(`Failed to archive attempt: ${err.message}`);
    }
  }

  async function handleRestore(attempt) {
    try {
      await api(`/admin/attempts/${attempt.id}/restore`, { method: "POST", token });
      await loadProfileActivity();
    } catch (err) {
      alert(`Failed to restore attempt: ${err.message}`);
    }
  }

  async function handleDelete(attempt) {
    const confirmed = window.confirm(
      `Are you sure you want to permanently delete Run ${attempt.run_number || attempt.attempt_number || attempt.id}?\n\nThis will permanently delete the video, report, and metrics. This action cannot be undone.`
    );
    if (!confirmed) return;

    try {
      await api(`/admin/attempts/${attempt.id}`, { method: "DELETE", token });
      await loadProfileActivity();
    } catch (err) {
      alert(`Failed to delete attempt: ${err.message}`);
    }
  }

  async function handleUpdateNotes(attemptId, notes) {
    await api(`/admin/attempts/${attemptId}/notes`, {
      method: "PATCH",
      token,
      body: { notes },
    });
    await loadProfileActivity();
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
      api(`/admin/users/${userId}/attempts?limit=100&include_archived=true`, { token }),
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
  const filteredCompletedAttempts = useMemo(() => {
    return completedAttempts.filter((attempt) => {
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const text = `run ${attempt.run_number || attempt.id} ${attempt.video_name || ""} ${attempt.instructor_notes || ""}`.toLowerCase();
        if (!text.includes(q)) return false;
      }
      if (selectedDay && attemptDayKey(attempt) !== selectedDay) return false;
      return true;
    });
  }, [completedAttempts, searchQuery, selectedDay]);

  const isProfileFilterActive = searchQuery.trim() !== "" || selectedDay !== "";
  const clearProfileFilters = () => {
    setSearchQuery("");
    setSelectedDay("");
    setDisplayCount(12);
  };

  const displayedAttempts = filteredCompletedAttempts.slice(0, displayCount);
  const hasMore = displayCount < filteredCompletedAttempts.length;

  return (
    <main className="app-shell">
      <div className="profile-back-row">
        <button type="button" className="secondary-button" onClick={() => routeTo("/admin")}>Back to athletes</button>
      </div>
      <AppHeader
        role="Admin / Athlete profile"
        title={user?.name || "Athlete profile"}
        subtitle="Review this athlete's personal bests, score history, videos and reports."
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
            <button type="button" className="primary-button profile-upload-btn" onClick={() => setShowUpload(true)}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload a video
            </button>
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

          <PersonalBestPanel
            records={personalBests}
            title={`${user.name}'s personal bests`}
            description="Highest completed Blue IQ and pillar results, with the session and run where each record was set."
          />

          {completedAttempts.length >= 2 && (
            <TrendsChart
              api={api}
              token={token}
              endpoint={`/admin/users/${userId}/trends`}
              audience="admin"
              athleteName={user.name}
              refreshKey={`${completedAttempts.length}-${latest?.id || ""}`}
            />
          )}

          <section className="section-block">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Analysis history</p>
                <h2>Videos and reports</h2>
                <p>Open any completed run, PDF report, or detailed score graph for this athlete.</p>
              </div>
            </div>

            {completedAttempts.length > 0 && (
              <div className="runs-filter-bar">
                <div className="runs-filter-search">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                  <input
                    type="text"
                    placeholder="Search run #, notes, or video..."
                    value={searchQuery}
                    onChange={(e) => { setSearchQuery(e.target.value); setDisplayCount(12); }}
                    className="runs-search-input"
                  />
                  {searchQuery && (
                    <button
                      type="button"
                      className="runs-search-clear-btn"
                      onClick={() => setSearchQuery("")}
                      title="Clear search"
                    >
                      ✕
                    </button>
                  )}
                </div>

                <div className="runs-filter-controls">
                  <DateNavigator value={selectedDay} onChange={(key) => { setSelectedDay(key); setDisplayCount(12); }} />

                  {isProfileFilterActive && (
                    <button type="button" className="btn-clear-filters" onClick={clearProfileFilters} title="Reset filters">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                      Reset
                    </button>
                  )}
                </div>
              </div>
            )}

            {completedAttempts.length > 0 && (
              <div className="runs-filter-status-row">
                <span className="runs-count-badge">
                  Showing <strong>{displayedAttempts.length}</strong> of <strong>{filteredCompletedAttempts.length}</strong> runs
                  {filteredCompletedAttempts.length !== completedAttempts.length && ` (filtered from ${completedAttempts.length} total)`}
                </span>
              </div>
            )}
            <div className="cards-grid">
              {displayedAttempts.length ? displayedAttempts.map((attempt) => (
                <AttemptCard
                  key={attempt.id}
                  attempt={attempt}
                  clientName={attempt.person_name || attempt.client_name || user?.name}
                  onViewAnalysis={setGraphAttempt}
                  onArchive={handleArchive}
                  onRestore={handleRestore}
                  onDelete={handleDelete}
                  onUpdateNotes={handleUpdateNotes}
                  isAdmin={true}
                />
              )) : (
                <FunEmptyState
                  kind={
                    !completedAttempts.length ? "athlete"
                      : searchQuery.trim() ? "search"
                        : selectedDay === todayKey() ? "today" : "day"
                  }
                  q={searchQuery.trim()}
                  day={selectedDay}
                  actionLabel={completedAttempts.length && isProfileFilterActive ? "Clear filters" : undefined}
                  onAction={clearProfileFilters}
                />
              )}
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
      {showUpload && user && (
        <UploadVideoModal
          token={token}
          user={user}
          onCompleted={loadProfileActivity}
          onClose={() => setShowUpload(false)}
        />
      )}
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
  const [attemptsLoading, setAttemptsLoading] = useState(false);
  const [leaderboards, setLeaderboards] = useState({});
  const [jobs, setJobs] = useState([]);
  const [adminView, setAdminView] = useState("upload");
  const [displayCount, setDisplayCount] = useState(12); // Show 12 initially to fill widescreen area
  const [showArchived, setShowArchived] = useState(false);
  const [clientSearch, setClientSearch] = useState("");
  const [runSearchQuery, setRunSearchQuery] = useState("");
  const [debouncedRunSearch, setDebouncedRunSearch] = useState("");
  const [selectedRunClient, setSelectedRunClient] = useState(null);
  const [selectedDay, setSelectedDay] = useState(todayKey);
  // Set once the admin picks a date themselves; until then a search spans all dates.
  const [dayChosen, setDayChosen] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedRunSearch(runSearchQuery.trim()), 300);
    return () => window.clearTimeout(timer);
  }, [runSearchQuery]);

  // The server filters by day and athlete name/email, so any past date works
  // (not just the most recent runs already in the browser).
  const searchActive = Boolean(selectedRunClient) || debouncedRunSearch !== "";
  // Looking up an athlete should show their whole history, not just today.
  const effectiveDay = searchActive && !dayChosen ? "" : selectedDay;

  const attemptsUrl = useMemo(() => {
    const params = new URLSearchParams({ limit: "100" });
    if (showArchived) params.set("only_archived", "true");
    if (effectiveDay) {
      const [start, end] = dayRangeIso(effectiveDay);
      params.set("start", start);
      params.set("end", end);
    }
    if (selectedRunClient) {
      params.set("person_ids", String(selectedRunClient.id));
    } else if (debouncedRunSearch) {
      // Fuzzy-match athletes in the browser (typos welcome), then fetch their runs.
      const matches = rankClients(users.filter((user) => user.role !== "admin"), debouncedRunSearch, 200);
      if (matches.length) params.set("person_ids", matches.map((client) => client.id).join(","));
      else params.set("q", debouncedRunSearch);
    }
    return `/admin/attempts?${params.toString()}`;
  }, [showArchived, effectiveDay, debouncedRunSearch, selectedRunClient, users]);
  const attemptsUrlRef = useRef(attemptsUrl);
  attemptsUrlRef.current = attemptsUrl;
  const attemptsRequestRef = useRef(0);
  const [graphAttempt, setGraphAttempt] = useState(null);

  async function loadJobs() {
    try {
      const jobRows = await api("/jobs/", { token });
      setJobs(Array.isArray(jobRows) ? jobRows : []);
    } catch {
      // ignore network blips
    }
  }

  async function loadAttempts() {
    const url = attemptsUrlRef.current;
    const requestId = ++attemptsRequestRef.current;
    const rows = await api(url, { token });
    // Ignore answers for filters the admin has already moved away from.
    if (requestId === attemptsRequestRef.current) setAttempts(rows);
  }

  async function loadData() {
    const [userRows, leaderboardRows, jobRows] = await Promise.all([
      api("/admin/users?limit=1000", { token }),
      api("/admin/leaderboards?limit=50", { token }),
      api("/jobs/", { token }).catch(() => []),
    ]);
    setUsers(userRows);
    setLeaderboards(leaderboardRows.leaderboards || {});
    setJobs(Array.isArray(jobRows) ? jobRows : []);
  }

  async function loadActivity() {
    const [, leaderboardRows, jobRows] = await Promise.all([
      loadAttempts(),
      api("/admin/leaderboards?limit=50", { token }),
      api("/jobs/", { token }).catch(() => []),
    ]);
    setLeaderboards(leaderboardRows.leaderboards || {});
    setJobs(Array.isArray(jobRows) ? jobRows : []);
  }

  async function handleArchive(attempt) {
    try {
      await api(`/admin/attempts/${attempt.id}/archive`, { method: "POST", token });
      await Promise.all([loadActivity(), loadData()]);
    } catch (err) {
      alert(`Failed to archive attempt: ${err.message}`);
    }
  }

  async function handleRestore(attempt) {
    try {
      await api(`/admin/attempts/${attempt.id}/restore`, { method: "POST", token });
      await Promise.all([loadActivity(), loadData()]);
    } catch (err) {
      alert(`Failed to restore attempt: ${err.message}`);
    }
  }

  async function handleDelete(attempt) {
    const confirmed = window.confirm(
      `Are you sure you want to permanently delete Run ${attempt.run_number || attempt.attempt_number || attempt.id}?\n\nThis will permanently delete the video, report, and metrics. This action cannot be undone.`
    );
    if (!confirmed) return;

    try {
      await api(`/admin/attempts/${attempt.id}`, { method: "DELETE", token });
      await Promise.all([loadActivity(), loadData()]);
    } catch (err) {
      alert(`Failed to delete attempt: ${err.message}`);
    }
  }

  async function handleUpdateNotes(attemptId, notes) {
    await api(`/admin/attempts/${attemptId}/notes`, {
      method: "PATCH",
      token,
      body: { notes },
    });
    await loadActivity();
  }

  useEffect(() => {
    if (!token) {
      routeTo("/admin-login");
      return;
    }
    loadData().catch((err) => setError(err.message));
  }, [token]);

  useEffect(() => {
    if (!token) return;
    setAttemptsLoading(true);
    loadAttempts()
      .catch((err) => setError(err.message))
      .finally(() => setAttemptsLoading(false));
  }, [token, attemptsUrl]);

  const activeJobsCount = jobs.filter((j) => j.status === "processing" || j.status === "pending").length;

  // Athlete run counts come with the user list; refresh it when uploads finish.
  const previousActiveJobs = useRef(activeJobsCount);
  useEffect(() => {
    if (token && previousActiveJobs.current > 0 && activeJobsCount < previousActiveJobs.current) {
      // A job just finished: refresh runs, leaderboards and athlete run counts.
      loadActivity().catch(() => {});
      loadData().catch(() => {});
    }
    previousActiveJobs.current = activeJobsCount;
  }, [activeJobsCount, token]);

  useEffect(() => {
    if (!token) return;
    const intervalTime = activeJobsCount > 0 ? JOBS_POLL_ACTIVE_MS : JOBS_POLL_IDLE_MS;
    const interval = setInterval(() => {
      if (document.visibilityState !== "hidden") loadJobs();
    }, intervalTime);
    return () => clearInterval(interval);
  }, [token, activeJobsCount]);

  useAutoRefresh(loadActivity, Boolean(token), ADMIN_ACTIVITY_FALLBACK_MS);

  const clients = useMemo(() => users.filter((user) => user.role !== "admin"), [users]);
  const clientsById = useMemo(() => Object.fromEntries(users.map((u) => [u.id, u.name])), [users]);
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
  const filteredAttempts = attempts;

  const runSearchLabel = selectedRunClient ? selectedRunClient.name : debouncedRunSearch;
  const isAnyFilterActive = runSearchQuery.trim() !== "" || Boolean(selectedRunClient) || selectedDay !== todayKey();
  const clearFilters = () => {
    setRunSearchQuery("");
    setSelectedRunClient(null);
    setDayChosen(false);
    setSelectedDay(todayKey());
    setDisplayCount(12);
  };

  const displayedAttempts = filteredAttempts.slice(0, displayCount);
  const hasMore = displayCount < filteredAttempts.length;

  return (
    <main className="app-shell">
      <AppHeader
        role="Admin dashboard"
        title="BLUERUN PORTAL"
        subtitle="Manage clients, process runs, and track performance across all athletes."
        onLogout={() => {
          clearToken("admin");
          routeTo("/admin-login");
        }}
      />
      {error && <div className="alert wide">{error}</div>}
      <nav className="admin-view-tabs" aria-label="Admin workspace">
        <button
          type="button"
          className={`tab-pill-btn ${adminView === "upload" ? "active" : ""}`}
          aria-current={adminView === "upload" ? "page" : undefined}
          onClick={() => setAdminView("upload")}
        >
          Upload
        </button>
        <button
          type="button"
          className={`tab-pill-btn ${adminView === "queue" ? "active" : ""}`}
          aria-current={adminView === "queue" ? "page" : undefined}
          onClick={() => setAdminView("queue")}
        >
          Upload Progress
          {activeJobsCount > 0 && (
            <span className="tab-pill-badge pulsing">{activeJobsCount}</span>
          )}
        </button>
        <button
          type="button"
          className={`tab-pill-btn ${adminView === "leaderboards" ? "active" : ""}`}
          aria-current={adminView === "leaderboards" ? "page" : undefined}
          onClick={() => setAdminView("leaderboards")}
        >
          Leaderboards
        </button>
        <button
          type="button"
          className={`tab-pill-btn ${adminView === "profiles" ? "active" : ""}`}
          aria-current={adminView === "profiles" ? "page" : undefined}
          onClick={() => setAdminView("profiles")}
        >
          Athlete Profiles
        </button>
      </nav>

      {adminView === "upload" && (
        <>
          <section className="upload-workspace">
            <UploadAnalysisPanel
              token={token}
              clients={clients}
              onCompleted={loadActivity}
              onJobQueued={() => {
                loadJobs();
                loadActivity();
              }}
              onGoToQueue={() => setAdminView("queue")}
            />
          </section>

          <section className="section-block recent-attempts-section">
            <div className="section-heading recent-attempts-heading">
              <div className="recent-attempts-title-group">
                <p className="eyebrow section-eyebrow">ANALYSIS HISTORY</p>
                <h2 className="section-title">{showArchived ? "Archived attempts" : "Recent attempts"}</h2>
                <p className="section-subtitle">Open recently generated videos, reports, and run graphs across all clients.</p>
              </div>
              <div className="filter-pill-group">
                <button
                  type="button"
                  className={`filter-pill-btn ${!showArchived ? "active" : ""}`}
                  onClick={() => { setShowArchived(false); setDisplayCount(12); }}
                >
                  Active runs
                </button>
                <button
                  type="button"
                  className={`filter-pill-btn ${showArchived ? "active" : ""}`}
                  onClick={() => { setShowArchived(true); setDisplayCount(12); }}
                >
                  Archived runs
                </button>
              </div>
            </div>

            {/* Advanced Search & Day-wise Filter Controls */}
            <div className="runs-filter-bar">
              <ClientSearchBox
                clients={clients}
                query={runSearchQuery}
                onQueryChange={(value) => { setRunSearchQuery(value); setDisplayCount(12); }}
                selected={selectedRunClient}
                onSelect={(client) => { setSelectedRunClient(client); setRunSearchQuery(""); setDisplayCount(12); }}
                onClear={() => { setSelectedRunClient(null); setDisplayCount(12); }}
              />

              <div className="runs-filter-controls">
                <DateNavigator value={effectiveDay} onChange={(key) => { setSelectedDay(key); setDayChosen(true); setDisplayCount(12); }} />

                {isAnyFilterActive && (
                  <button type="button" className="btn-clear-filters" onClick={clearFilters} title="Reset all filters">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                    Reset
                  </button>
                )}
              </div>
            </div>

            <div className="runs-filter-status-row">
              <span className="runs-count-badge">
                Showing <strong>{displayedAttempts.length}</strong> of <strong>{filteredAttempts.length}</strong> runs
                {effectiveDay ? ` · ${dayLabel(effectiveDay)}` : " · All dates"}
                {selectedRunClient
                  ? ` · ${selectedRunClient.name}`
                  : debouncedRunSearch && ` · matching “${debouncedRunSearch}”`}
              </span>
            </div>

            {attemptsLoading ? (
              <div className="empty-state" style={{ padding: '48px 0', color: 'var(--blue)' }}>
                Loading {showArchived ? "archived" : "active"} attempts...
              </div>
            ) : (
              <>
                <div className="cards-grid">
                  {displayedAttempts.length ? displayedAttempts.map((attempt) => (
                    <AttemptCard
                      key={attempt.id}
                      attempt={attempt}
                      clientName={attempt.person_name || attempt.client_name || clientsById[attempt.person_id]}
                      onViewAnalysis={setGraphAttempt}
                      onArchive={handleArchive}
                      onRestore={handleRestore}
                      onDelete={handleDelete}
                      onUpdateNotes={handleUpdateNotes}
                      isAdmin={true}
                    />
                  )) : (
                    <FunEmptyState
                      kind={
                        runSearchLabel ? "search"
                          : showArchived ? "archived"
                            : !effectiveDay ? "none"
                              : effectiveDay === todayKey() ? "today" : "day"
                      }
                      q={runSearchLabel}
                      day={effectiveDay}
                      actionLabel={effectiveDay ? "Show all dates" : undefined}
                      onAction={() => { setSelectedDay(""); setDayChosen(true); }}
                    />
                  )}
                </div>
                {hasMore && (
                  <div className="load-more-row">
                    <button className="secondary-button" onClick={() => setDisplayCount(prev => prev + 8)}>
                      Load 8 More ({filteredAttempts.length - displayCount} remaining)
                    </button>
                  </div>
                )}
              </>
            )}
          </section>
        </>
      )}

      {adminView === "queue" && (
        <UploadQueuePanel
          jobs={jobs}
          token={token}
          onRefresh={loadJobs}
          onViewAnalysis={setGraphAttempt}
          onNavigateUpload={() => setAdminView("upload")}
        />
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
              const runCount = user.run_count ?? 0;
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
            {!clients.length && <FunEmptyState kind="noClients" />}
            {clients.length > 0 && !searchedClients.length && (
              <FunEmptyState
                kind="clients"
                q={clientSearch.trim()}
                actionLabel="Clear search"
                onAction={() => setClientSearch("")}
              />
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

// Light-hearted copy for empty results. {q} is the search text, {day} the date phrase.
const EMPTY_MESSAGES = {
  today: [
    ["🏔️", "The slopes are quiet today", "No runs yet. The skis are lowkey waiting for their main character moment."],
    ["🫥", "It's giving… empty", "Zero runs today. Upload one and let the analysis cook."],
    ["🧢", "No runs, no cap", "Today's feed is out touching grass. Upload a run to change that."],
  ],
  day: [
    ["🎿", "Nobody shredded {day}", "That day was a vibe check nobody showed up for."],
    ["😴", "{Day} said “not today”", "No runs recorded. The mountain was in its rest era."],
    ["👻", "Ghost town {day}", "Not a single run. Try another day, or see every date."],
  ],
  search: [
    ["🔍", "No runs for “{q}”", "We looked everywhere. They might be in their off-season era."],
    ["🤷", "“{q}”? Never heard of them", "No runs match. Double-check the name or email, bestie."],
    ["🙅", "It's a no from us", "Nothing matches “{q}”. Try another spelling or another day."],
  ],
  archived: [
    ["📦", "Nothing in the archive", "Clean slate. Very organised of you, honestly."],
    ["✨", "Archive is spotless", "No archived runs here. Marie Kondo would be proud."],
  ],
  none: [
    ["🚀", "No runs yet", "The first upload is where the glow-up begins."],
    ["🌨️", "Fresh powder, zero tracks", "Upload the first run and let's see what this crew can do."],
  ],
  athlete: [
    ["🎿", "No runs here yet", "This athlete's highlight reel is loading. First upload, let's go."],
    ["⏳", "Highlight reel pending", "No runs to show yet. The comeback starts with one upload."],
  ],
  clients: [
    ["🕵️", "No athletes found", "“{q}” isn't on the roster. Typo, or a new recruit?"],
    ["🫠", "Zero matches for “{q}”", "Not in the squad (yet). Check the spelling, bestie."],
  ],
  noClients: [
    ["👋", "No athletes yet", "The roster is giving empty. Add the first athlete to get things rolling."],
  ],
};

function pickEmptyMessage(kind, { q = "", day = "" } = {}) {
  const options = EMPTY_MESSAGES[kind] || EMPTY_MESSAGES.none;
  // Stable choice per situation so auto-refresh does not swap the joke.
  const seed = `${kind}|${q}|${day}`.split("").reduce((sum, ch) => (sum * 31 + ch.charCodeAt(0)) >>> 0, 7);
  const [emoji, title, body] = options[seed % options.length];
  const phrase = day ? dayPhrase(day) : "";
  const fill = (text) => text
    .replaceAll("{q}", q)
    .replaceAll("{Day}", phrase ? phrase.charAt(0).toUpperCase() + phrase.slice(1) : "")
    .replaceAll("{day}", phrase);
  return { emoji, title: fill(title), body: fill(body) };
}

function FunEmptyState({ kind, q, day, actionLabel, onAction }) {
  const { emoji, title, body } = pickEmptyMessage(kind, { q, day });
  return (
    <div className="fun-empty-state" role="status">
      <span className="fun-empty-emoji" aria-hidden="true">{emoji}</span>
      <h3>{title}</h3>
      <p>{body}</p>
      {actionLabel && onAction && (
        <button type="button" className="secondary-button fun-empty-action" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  );
}

function App() {
  const path = useRoute();
  const component = useMemo(() => {
    const athleteProfileMatch = path.match(/^\/admin\/athletes\/(\d+)$/);
    if (path === "/signup") return <AuthForm type="signup" />;
    if (path === "/admin-login") return <AuthForm type="admin" />;
    if (path === "/forgot-password") return <ForgotPasswordForm />;
    if (athleteProfileMatch) return <AdminAthleteProfile userId={Number(athleteProfileMatch[1])} />;
    if (path === "/admin") return <AdminDashboard />;
    if (path === "/dashboard") return <ClientDashboard />;
    return <AuthForm type="login" />;
  }, [path]);
  return component;
}

export default App;
