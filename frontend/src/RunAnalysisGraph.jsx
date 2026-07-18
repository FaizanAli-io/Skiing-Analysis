import { useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";


ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
);


const PILLAR_ORDER = ["pressure", "balance", "rotation", "edging"];


function valueAtPath(source, path) {
  return path.split(".").reduce((value, key) => value?.[key], source);
}


function initialVisibleKeys(series) {
  return new Set(
    series.filter((item) => item.default_visible !== false).map((item) => item.key),
  );
}


function formatNumber(value) {
  if (!Number.isFinite(Number(value))) return "--";
  const number = Number(value);
  return Math.abs(number) >= 100 ? Math.round(number) : number.toFixed(1);
}


function buildValueScales(activeSeries) {
  const axisKeys = [...new Set(activeSeries.map((series) => series.axis || series.unit))];
  return axisKeys.reduce((scales, axisKey, index) => {
    const series = activeSeries.find((item) => (item.axis || item.unit) === axisKey);
    const scaleId = `value-${axisKey}`;
    scales[scaleId] = {
      type: "linear",
      position: index === 0 ? "left" : "right",
      beginAtZero: false,
      grid: {
        color: index === 0 ? "rgba(255, 255, 255, 0.07)" : "transparent",
      },
      border: { color: "rgba(255, 255, 255, 0.12)" },
      ticks: {
        color: series?.color || "#8fa3bd",
        padding: 8,
        callback: (value) => `${value}`,
      },
      title: {
        display: true,
        text: axisKey === "percent" ? "%" : series?.unit || "Value",
        color: "#8fa3bd",
        font: { size: 11, weight: "500" },
      },
    };
    return scales;
  }, {});
}


export default function RunAnalysisGraph({ api, token, attempt, athleteName, onClose }) {
  const [timeline, setTimeline] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("score");
  const [pillar, setPillar] = useState("pressure");
  const [visibleKeys, setVisibleKeys] = useState(new Set());

  useEffect(() => {
    let active = true;
    setTimeline(null);
    setError("");
    api(`/admin/attempts/${attempt.id}/timeline?resolution=second`, { token })
      .then((payload) => {
        if (active) setTimeline(payload);
      })
      .catch((requestError) => {
        if (active) setError(requestError.message);
      });
    return () => {
      active = false;
    };
  }, [api, attempt.id, token]);

  useEffect(() => {
    const closeOnEscape = (event) => {
      if (event.key === "Escape") onClose();
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [onClose]);

  const pillarConfig = timeline?.parameter_config?.[pillar];
  const availableSeries = pillarConfig?.[mode === "score" ? "score_series" : "value_series"] || [];

  useEffect(() => {
    setVisibleKeys(initialVisibleKeys(availableSeries));
  }, [mode, pillar, timeline]);

  const activeSeries = availableSeries.filter((series) => visibleKeys.has(series.key));

  const chartData = useMemo(() => ({
    labels: (timeline?.samples || []).map((sample) => `${sample.time_seconds}s`),
    datasets: activeSeries.map((series) => ({
      label: series.label,
      data: timeline.samples.map((sample) => {
        const value = valueAtPath(sample, series.path);
        return Number.isFinite(Number(value)) ? Number(value) : null;
      }),
      borderColor: series.color,
      backgroundColor: `${series.color}18`,
      borderWidth: series.key === "overall" ? 3 : 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointBackgroundColor: series.color,
      pointBorderColor: "#07111f",
      pointBorderWidth: 2,
      tension: 0.28,
      spanGaps: true,
      fill: false,
      yAxisID: mode === "score" ? "score" : `value-${series.axis || series.unit}`,
      unit: series.unit,
    })),
  }), [activeSeries, mode, timeline]);

  const chartOptions = useMemo(() => {
    const scoreScales = {
      score: {
        min: 60,
        max: 240,
        grid: { color: "rgba(255, 255, 255, 0.07)" },
        border: { color: "rgba(255, 255, 255, 0.12)" },
        ticks: { color: "#8fa3bd", stepSize: 30, padding: 8 },
        title: {
          display: true,
          text: "Score /240",
          color: "#8fa3bd",
          font: { size: 11, weight: "500" },
        },
      },
    };
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#071426",
          borderColor: "rgba(106, 175, 255, 0.28)",
          borderWidth: 1,
          titleColor: "#eef6ff",
          bodyColor: "#c8d7ea",
          padding: 12,
          callbacks: {
            label: (context) => {
              const unit = context.dataset.unit || "";
              return `${context.dataset.label}: ${formatNumber(context.parsed.y)}${unit === "/240" ? "/240" : ` ${unit}`}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          border: { color: "rgba(255, 255, 255, 0.12)" },
          ticks: { color: "#8fa3bd", maxTicksLimit: 12, maxRotation: 0 },
          title: {
            display: true,
            text: "Run time (seconds)",
            color: "#8fa3bd",
            font: { size: 11, weight: "500" },
          },
        },
        ...(mode === "score" ? scoreScales : buildValueScales(activeSeries)),
      },
    };
  }, [activeSeries, mode]);

  function toggleSeries(key) {
    setVisibleKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <div
      className="run-graph-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section className="run-graph-dialog" role="dialog" aria-modal="true" aria-labelledby="run-graph-title">
        <header className="run-graph-header">
          <div>
            <p className="eyebrow">Run {attempt.attempt_number || attempt.id} analysis</p>
            <h2 id="run-graph-title">{athleteName || "Athlete"} metric timeline</h2>
            <p>Compare each scored component through the run.</p>
          </div>
          <button className="run-graph-close" type="button" onClick={onClose} aria-label="Close metric timeline">x</button>
        </header>

        {error ? (
          <div className="run-graph-message error-state">
            <strong>Timeline unavailable</strong>
            <span>{error}</span>
          </div>
        ) : !timeline ? (
          <div className="run-graph-message">
            <span className="loading-line" />
            <strong>Loading run metrics...</strong>
          </div>
        ) : (
          <>
            <div className="run-graph-toolbar">
              <div className="pillar-tabs" role="tablist" aria-label="Metric pillar">
                {PILLAR_ORDER.map((key) => {
                  const config = timeline.parameter_config[key];
                  return (
                    <button
                      key={key}
                      type="button"
                      role="tab"
                      aria-selected={pillar === key}
                      className={pillar === key ? "active" : ""}
                      style={{ "--pillar-color": config.color }}
                      onClick={() => setPillar(key)}
                    >
                      {config.label}
                    </button>
                  );
                })}
              </div>
              <div className="graph-mode-toggle" aria-label="Chart value mode">
                <button type="button" className={mode === "score" ? "active" : ""} onClick={() => setMode("score")}>Score</button>
                <button type="button" className={mode === "value" ? "active" : ""} onClick={() => setMode("value")}>Actual value</button>
              </div>
            </div>

            <div className="series-controls" aria-label="Visible chart lines">
              {availableSeries.map((series) => {
                const selected = visibleKeys.has(series.key);
                return (
                  <button
                    type="button"
                    key={series.key}
                    className={selected ? "selected" : ""}
                    aria-pressed={selected}
                    onClick={() => toggleSeries(series.key)}
                  >
                    <span className="series-swatch" style={{ backgroundColor: series.color }} />
                    <span>{series.label}</span>
                    {series.weight ? <small>{series.weight}%</small> : null}
                  </button>
                );
              })}
            </div>

            <div className="run-chart-wrap">
              {activeSeries.length ? (
                <Line data={chartData} options={chartOptions} />
              ) : (
                <div className="chart-empty">Select at least one line to display.</div>
              )}
            </div>

            <div className="run-graph-meta">
              <span>One-second averages</span>
              <span>{timeline.duration_seconds.toFixed(1)} sec run</span>
              <span>{timeline.sample_rate_hz.toFixed(0)} Hz source data</span>
            </div>

            <section className="score-reference" aria-labelledby="score-reference-title">
              <div>
                <p className="eyebrow">Coach reference</p>
                <h3 id="score-reference-title">How {pillarConfig.label.toLowerCase()} is scored</h3>
              </div>
              <div className="reference-table">
                {pillarConfig.references.map((reference) => (
                  <div className="reference-row" key={reference.label}>
                    <strong>{reference.label}</strong>
                    <div className="reference-copy">
                      <span>{reference.explanation || reference.mapping}</span>
                      {reference.explanation && <small>{reference.mapping}</small>}
                    </div>
                    <small className="reference-weight">{reference.weight}</small>
                  </div>
                ))}
              </div>
            </section>
          </>
        )}
      </section>
    </div>
  );
}
