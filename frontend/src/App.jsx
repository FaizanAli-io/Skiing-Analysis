import { useEffect, useMemo, useRef, useState } from "react";
import TrendsChart from "./TrendsChart";
import RunAnalysisGraph from "./RunAnalysisGraph";
import logoImg from "../../services/bluerun.png";

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
  if (value >= 225) return ["Mastering 3", "mastering"];
  if (value >= 210) return ["Mastering 2", "mastering"];
  if (value >= 195) return ["Mastering 1", "mastering"];
  if (value >= 180) return ["Advancing 3", "advancing"];
  if (value >= 165) return ["Advancing 2", "advancing"];
  if (value >= 150) return ["Advancing 1", "advancing"];
  if (value >= 135) return ["Progressing 3", "progressing"];
  if (value >= 120) return ["Progressing 2", "progressing"];
  if (value >= 105) return ["Progressing 1", "progressing"];
  if (value >= 90) return ["Building 3", "building"];
  if (value >= 75) return ["Building 2", "building"];
  return ["Building 1", "building"];
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

function AttemptCard({ attempt, onViewAnalysis, onArchive, onRestore, onDelete }) {
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
    <article className={`attempt-card ${attempt.is_archived ? "is-archived" : ""}`}>
      <div className="attempt-top">
        <div>
          <p className="eyebrow">{sessionLabel} / Run {runNumber}</p>
          <h3>Blue IQ {blueIq}<span>/240</span></h3>
          <p className="attempt-date">{dateLabel}</p>
        </div>
        <div className="attempt-badges">
          {attempt.is_archived && <span className="filter-chip" style={{ color: '#f59e0b', borderColor: '#f59e0b', background: 'rgba(245,158,11,0.1)' }}>Archived</span>}
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
        {onRestore && attempt.is_archived && (
          <button type="button" className="action-restore" onClick={() => onRestore(attempt)}>
            Restore
          </button>
        )}
        {onArchive && !attempt.is_archived && (
          <button type="button" className="action-archive" onClick={() => onArchive(attempt)}>
            Archive
          </button>
        )}
        {onDelete && (
          <button type="button" className="action-delete" onClick={() => onDelete(attempt)}>
            Delete
          </button>
        )}
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
  const [uploadState, setUploadState] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0); // 0-100
  const [progressPhase, setProgressPhase] = useState(""); // 'uploading' | 'queued' | ''
  const [uploadError, setUploadError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [queuedJobInfo, setQueuedJobInfo] = useState(null);

  // Google Drive state
  const [driveFile, setDriveFile] = useState(null); // { id, name, size, oauthToken }
  const [driveLinkInput, setDriveLinkInput] = useState("");
  const [googleConfig, setGoogleConfig] = useState({ client_id: "", api_key: "", app_id: "" });
  const [isPickerLoading, setIsPickerLoading] = useState(false);
  const [pickerError, setPickerError] = useState("");

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
      if (token) {
        xhr.setRequestHeader("Authorization", `Bearer ${token}`);
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
      let response;
      if (uploadSource === "local") {
        response = await uploadWithProgress(
          `${API_BASE}/api/analyze-premium-overlay/`,
          new FormData(formElement)
        );
      } else {
        const targetFileId = driveFile?.id || extractDriveId(driveLinkInput);
        if (!targetFileId) {
          throw new Error("Please select a video from Google Drive or paste a valid Google Drive video link.");
        }
        const formData = new FormData();
        const userId = fixedUser ? fixedUser.id : formElement.querySelector('[name="user_id"]').value;
        const displayMode = formElement.querySelector('[name="display_mode"]').value;
        const report = formElement.querySelector('[name="report"]')?.checked ? "true" : "false";

        formData.append("user_id", userId);
        formData.append("display_mode", displayMode);
        formData.append("report", report);
        formData.append("drive_file_id", targetFileId);
        if (driveFile?.name) formData.append("file_name", driveFile.name);
        if (driveFile?.oauthToken) formData.append("oauth_token", driveFile.oauthToken);

        setUploadState("Queueing Google Drive video for download and analysis...");
        setUploadProgress(100);
        response = await api("/analyze-google-drive/", {
          method: "POST",
          token,
          body: formData,
        });
      }

      formElement.reset();
      setDriveFile(null);
      setDriveLinkInput("");

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
      <p className="eyebrow">New analysis</p>
      <h2>{fixedUser ? `Upload a run for ${fixedUser.name}` : "Upload client video"}</h2>

      <div className="upload-source-toggle">
        <button
          type="button"
          className={`upload-source-btn ${uploadSource === "local" ? "active" : ""}`}
          onClick={() => setUploadSource("local")}
          disabled={isSubmitting}
        >
          📁 Local Computer
        </button>
        <button
          type="button"
          className={`upload-source-btn ${uploadSource === "drive" ? "active" : ""}`}
          onClick={() => setUploadSource("drive")}
          disabled={isSubmitting}
        >
          ☁️ Google Drive
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
        <>
          <label>Select user</label>
          <select name="user_id" required disabled={isSubmitting}>
            <option value="">Choose a client</option>
            {clients.map((user) => (
              <option key={user.id} value={user.id}>{user.name} - {user.email}</option>
            ))}
          </select>
        </>
      )}
      <label>Display mode</label>
      <select name="display_mode" defaultValue="coach" disabled={isSubmitting}>
        <option value="coach">Coach</option>
        <option value="athlete">Athlete</option>
      </select>
      <label className="checkbox-row">
        <input type="checkbox" name="report" value="true" defaultChecked disabled={isSubmitting} />
        Generate PDF report
      </label>

      {uploadSource === "local" ? (
        <>
          <label>Video file</label>
          <input type="file" name="file" accept="video/*" required disabled={isSubmitting} />
        </>
      ) : (
        <div className="drive-picker-container">
          <label>Google Drive Video</label>
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

      <button className="primary-button" disabled={isSubmitting}>
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
  const [displayCount, setDisplayCount] = useState(12);
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
                  onArchive={handleArchive}
                  onRestore={handleRestore}
                  onDelete={handleDelete}
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
  const [attemptsLoading, setAttemptsLoading] = useState(false);
  const [leaderboards, setLeaderboards] = useState({});
  const [jobs, setJobs] = useState([]);
  const [adminView, setAdminView] = useState("upload");
  const [displayCount, setDisplayCount] = useState(12); // Show 12 initially to fill widescreen area
  const [showArchived, setShowArchived] = useState(false);
  const [clientSearch, setClientSearch] = useState("");
  const [error, setError] = useState("");
  const [graphAttempt, setGraphAttempt] = useState(null);

  async function loadJobs() {
    try {
      const jobRows = await api("/jobs/", { token });
      setJobs(Array.isArray(jobRows) ? jobRows : []);
    } catch {
      // ignore network blips
    }
  }

  async function loadData() {
    setAttemptsLoading(true);
    try {
      const attemptsUrl = showArchived
        ? "/admin/attempts?limit=100&only_archived=true"
        : "/admin/attempts?limit=100";
      const [userRows, attemptRows, leaderboardRows, jobRows] = await Promise.all([
        api("/admin/users?limit=1000", { token }),
        api(attemptsUrl, { token }),
        api("/admin/leaderboards?limit=50", { token }),
        api("/jobs/", { token }).catch(() => []),
      ]);
      setUsers(userRows);
      setAttempts(attemptRows);
      setLeaderboards(leaderboardRows.leaderboards || {});
      setJobs(Array.isArray(jobRows) ? jobRows : []);
    } finally {
      setAttemptsLoading(false);
    }
  }

  async function loadActivity() {
    const attemptsUrl = showArchived
      ? "/admin/attempts?limit=100&only_archived=true"
      : "/admin/attempts?limit=100";
    const [attemptRows, leaderboardRows, jobRows] = await Promise.all([
      api(attemptsUrl, { token }),
      api("/admin/leaderboards?limit=50", { token }),
      api("/jobs/", { token }).catch(() => []),
    ]);
    setAttempts(attemptRows);
    setLeaderboards(leaderboardRows.leaderboards || {});
    setJobs(Array.isArray(jobRows) ? jobRows : []);
  }

  async function handleArchive(attempt) {
    try {
      await api(`/admin/attempts/${attempt.id}/archive`, { method: "POST", token });
      await loadActivity();
    } catch (err) {
      alert(`Failed to archive attempt: ${err.message}`);
    }
  }

  async function handleRestore(attempt) {
    try {
      await api(`/admin/attempts/${attempt.id}/restore`, { method: "POST", token });
      await loadActivity();
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
      await loadActivity();
    } catch (err) {
      alert(`Failed to delete attempt: ${err.message}`);
    }
  }

  useEffect(() => {
    if (!token) {
      routeTo("/admin-login");
      return;
    }
    loadData().catch((err) => setError(err.message));
  }, [token, showArchived]);

  const activeJobsCount = jobs.filter((j) => j.status === "processing" || j.status === "pending").length;

  useEffect(() => {
    if (!token) return;
    const intervalTime = activeJobsCount > 0 ? 2500 : 8000;
    const interval = setInterval(() => {
      loadJobs();
      if (activeJobsCount > 0) {
        loadActivity();
      }
    }, intervalTime);
    return () => clearInterval(interval);
  }, [token, activeJobsCount]);

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
          className={adminView === "queue" ? "active" : ""}
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

          <section className="section-block">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Analysis history</p>
                <h2>{showArchived ? "Archived attempts" : "Recent attempts"}</h2>
                <p>Open recently generated videos, reports, and run graphs across all clients.</p>
                <div className="filter-toggle-row">
                  <button
                    type="button"
                    className={`filter-chip ${!showArchived ? "active" : ""}`}
                    onClick={() => { setShowArchived(false); setDisplayCount(12); }}
                  >
                    Active runs
                  </button>
                  <button
                    type="button"
                    className={`filter-chip ${showArchived ? "active" : ""}`}
                    onClick={() => { setShowArchived(true); setDisplayCount(12); }}
                  >
                    Archived runs
                  </button>
                </div>
              </div>
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
                      onViewAnalysis={setGraphAttempt}
                      onArchive={handleArchive}
                      onRestore={handleRestore}
                      onDelete={handleDelete}
                    />
                  )) : <EmptyState text={showArchived ? "No archived attempts." : "No attempts have been generated yet."} />}
                </div>
                {hasMore && (
                  <div className="load-more-row">
                    <button className="secondary-button" onClick={() => setDisplayCount(prev => prev + 8)}>
                      Load 8 More ({attempts.length - displayCount} remaining)
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
