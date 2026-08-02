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
  ["edging", "Edging"],
];

function comparisonText(comparison) {
  if (!comparison) return "No comparison available";
  if (comparison.status === "new_personal_best") {
    const previous = comparison.previous_best?.score;
    const gain = previous == null ? 0 : comparison.current_score - previous;
    return gain > 0 ? `New personal best, ${gain} points higher` : "New personal best";
  }
  if (comparison.status === "baseline") return "First recorded result: baseline established";
  if (comparison.status === "matches_personal_best") return "Matches your personal best";
  return `${comparison.points_below} points below your personal best`;
}

function bestContext(record) {
  if (!record) return "No completed result yet";
  const parts = [];
  if (record.date && record.date !== "Unknown date") parts.push(record.date);
  if (record.session_number) parts.push(`Session ${record.session_number}`);
  if (record.run_number) parts.push(`Run ${record.run_number}`);
  return parts.join(" / ") || "Recorded result";
}

export default function TrendsChart({ api, token, refreshKey }) {
  const [trends, setTrends] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState("blue_iq");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api("/me/trends?limit=15", { token })
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
  }, [api, token, refreshKey]);

  const chartData = useMemo(() => {
    if (!trends) return null;
    const metricLabel = METRICS.find(([key]) => key === selectedMetric)?.[1] || selectedMetric;
    return {
      labels: trends.time_series.runs.map((run) => `Run ${run}`),
      datasets: [
        {
          label: metricLabel,
          data: trends.time_series[selectedMetric],
          borderColor: "#6aafff",
          backgroundColor: "rgba(106, 175, 255, 0.08)",
          borderWidth: 2,
          tension: 0.28,
          fill: true,
          pointRadius: 4,
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

  const comparison = trends.current_vs_personal_best?.[selectedMetric];
  const personalBest = trends.personal_bests?.[selectedMetric];
  const currentScore = comparison?.current_score ?? trends.time_series[selectedMetric]?.at(-1) ?? "--";
  const selectedLabel = METRICS.find(([key]) => key === selectedMetric)?.[1];

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
          label: (item) => `${selectedLabel}: ${item.formattedValue} / 240`,
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
          <p>Compare completed runs against your all-time personal best.</p>
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
          <span>Personal best</span>
          <strong>{personalBest?.score ?? "--"}<small>/240</small></strong>
          <p>{bestContext(personalBest)}</p>
        </div>
        <div className="comparison-card comparison-copy">
          <span>Current comparison</span>
          <strong>{comparisonText(comparison)}</strong>
          <p>Compared with your highest completed result.</p>
        </div>
      </div>
    </section>
  );
}
