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
  Title,
  Tooltip,
} from "chart.js";
import { scoreBand } from "./scoreBands";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
);

const METRICS = [
  ["blue_iq", "Blue IQ"],
  ["pressure", "Pressure"],
  ["balance", "Balance"],
  ["rotation", "Rotation"],
  ["edging", "Parallelism"],
];

// Never leads with a drop: below the first run, talk about where the score sits.
const KEEP_GOING_LINES = [
  "Every run adds to the foundation. The next one could be the breakthrough.",
  "Progress isn't a straight line. Keep stacking the reps.",
  "Steady work at this level is exactly how the next level arrives.",
];

function firstRunComparison(comparison, { you, metricLabel, metricIndex }) {
  if (!comparison) return { headline: "No comparison yet", detail: "Complete another run to see progress." };
  const current = comparison.current_score;
  const level = scoreBand(current)[0];
  const difference = comparison.difference;
  if (comparison.is_first_run) {
    return {
      headline: "This is the starting line",
      detail: `Every future ${metricLabel} score is measured from here.`,
    };
  }
  if (difference > 0) {
    return {
      headline: `Up ${difference} point${difference === 1 ? "" : "s"}`,
      detail: `${metricLabel} since ${you.possessive} first run.`,
    };
  }
  if (difference === 0) {
    return {
      headline: `Holding at ${level}`,
      detail: `${current} matches ${you.possessive} first run. Consistency is a skill too.`,
    };
  }
  return {
    headline: `${current} puts ${you.object} at ${level}`,
    detail: KEEP_GOING_LINES[metricIndex % KEEP_GOING_LINES.length],
  };
}

function bestContext(record) {
  if (!record) return "No completed result yet";
  const parts = [];
  if (record.date && record.date !== "Unknown date") parts.push(record.date);
  if (record.session_number) parts.push(`Session ${record.session_number}`);
  if (record.run_number) parts.push(`Run ${record.run_number}`);
  return parts.join(" / ") || "Recorded result";
}

export default function TrendsChart({
  api,
  token,
  refreshKey,
  endpoint = "/me/trends",
  audience = "client",
  athleteName = "",
}) {
  const isClient = audience === "client";
  const you = isClient
    ? { possessive: "your", object: "you" }
    : { possessive: "their", object: athleteName ? athleteName.split(" ")[0] : "them" };
  const [trends, setTrends] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState("blue_iq");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api(endpoint, { token })
      .then((data) => {
        if (!active) return;
        if (data.error) {
          setError(data.error);
          return;
        }
        setTrends(data);
        setError("");
      })
      .catch((err) => active && setError(err.message));
    return () => {
      active = false;
    };
  }, [api, token, refreshKey, endpoint]);

  const chartData = useMemo(() => {
    if (!trends) return null;
    const metricLabel = METRICS.find(([key]) => key === selectedMetric)?.[1] || selectedMetric;
    const firstScore = trends.first_run?.scores?.[selectedMetric];
    const reference = firstScore == null ? [] : [
      {
        label: "First run",
        data: trends.time_series.runs.map(() => firstScore),
        borderColor: "rgba(139, 192, 213, 0.55)",
        borderWidth: 1.5,
        borderDash: [6, 6],
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: false,
        tension: 0,
      },
    ];
    return {
      labels: trends.time_series.runs.map((run) => `Run ${run}`),
      datasets: [
        ...reference,
        {
          label: metricLabel,
          data: trends.time_series[selectedMetric],
          borderColor: "#6aafff",
          backgroundColor: "rgba(106, 175, 255, 0.08)",
          borderWidth: 2,
          tension: 0.28,
          fill: true,
          // Every run is plotted; keep dots small when the history is long.
          pointRadius: trends.time_series.runs.length > 30 ? 2 : 4,
          pointHoverRadius: 6,
          pointBackgroundColor: "#6aafff",
          pointBorderColor: "#07111f",
          pointBorderWidth: 2,
        },
      ],
    };
  }, [selectedMetric, trends]);

  if (error) {
    return (
      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Run history</p>
            <h2>Score history</h2>
          </div>
        </div>
        <div className="empty-state">{error}</div>
      </section>
    );
  }

  if (!trends || !chartData) {
    return (
      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Run history</p>
            <h2>Score history</h2>
          </div>
        </div>
        <div className="empty-state">Loading score history...</div>
      </section>
    );
  }

  const comparison = trends.current_vs_first_run?.[selectedMetric];
  const firstRun = trends.first_run;
  const currentScore = comparison?.current_score ?? trends.time_series[selectedMetric]?.at(-1) ?? "--";
  const selectedLabel = METRICS.find(([key]) => key === selectedMetric)?.[1];
  const metricIndex = METRICS.findIndex(([key]) => key === selectedMetric);
  const summary = firstRunComparison(comparison, { you, metricLabel: selectedLabel, metricIndex });
  const trendClass = comparison && !comparison.is_first_run && comparison.difference > 0 ? "is-up" : "is-steady";

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      title: { display: false },
      tooltip: {
        backgroundColor: "#07111f",
        titleColor: "#eef6ff",
        bodyColor: "#c8d7ea",
        borderColor: "rgba(106, 175, 255, 0.28)",
        borderWidth: 1,
        padding: 12,
        displayColors: false,
        callbacks: {
          afterTitle: (items) => trends.time_series.dates[items[0]?.dataIndex] || "",
          label: (item) => `${item.dataset.label === "First run" ? "First run" : selectedLabel}: ${item.formattedValue} / 240`,
        },
      },
    },
    scales: {
      y: {
        min: 60,
        max: 240,
        grid: { color: "rgba(255, 255, 255, 0.07)" },
        ticks: { color: "#8fa3bd", stepSize: 30 },
      },
      x: {
        grid: { display: false },
        ticks: { color: "#8fa3bd", maxRotation: 0, autoSkip: true },
      },
    },
  };

  return (
    <section className="section-block trends-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Run history</p>
          <h2>Score history</h2>
          <p>
            {isClient
              ? "See how far you've come since your first run."
              : `See how far ${athleteName || "this athlete"} has come since their first run.`}
          </p>
        </div>
        <div className="metric-tabs" role="tablist" aria-label="Score metric">
          {METRICS.map(([key, label]) => (
            <button
              type="button"
              role="tab"
              aria-selected={selectedMetric === key}
              className={selectedMetric === key ? "active" : ""}
              key={key}
              onClick={() => setSelectedMetric(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="history-chart">
        <Line data={chartData} options={chartOptions} />
      </div>

      <div className="comparison-grid">
        <div className="comparison-card">
          <span>Current run</span>
          <strong>{currentScore}<small>/240</small></strong>
          <p>{trends.date_range.end}</p>
        </div>
        <div className="comparison-card personal-best-card">
          <span>First run</span>
          <strong>{firstRun?.scores?.[selectedMetric] ?? "--"}<small>/240</small></strong>
          <p>{bestContext(firstRun)}</p>
        </div>
        <div className={`comparison-card comparison-copy first-run-progress ${trendClass}`}>
          <span>Since {you.possessive} first run</span>
          <strong>{summary.headline}</strong>
          <p>{summary.detail}</p>
        </div>
      </div>
    </section>
  );
}
