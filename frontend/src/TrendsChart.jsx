import { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function TrendsChart({ api, token, refreshKey }) {
  const [trends, setTrends] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('blue_iq');
  const [error, setError] = useState('');

  useEffect(() => {
    api('/me/trends?limit=15', { token })
      .then((data) => {
        if (data.error) {
          setError(data.error);
        } else {
          setTrends(data);
          setError('');
        }
      })
      .catch((err) => setError(err.message));
  }, [api, token, refreshKey]);

  if (error) {
    return (
      <section className="section-block">
        <div className="section-heading">
          <h2>Performance Trends</h2>
          <p>Track your progress over time</p>
        </div>
        <div className="empty-state">{error}</div>
      </section>
    );
  }

  if (!trends) {
    return (
      <section className="section-block">
        <div className="section-heading">
          <h2>Performance Trends</h2>
          <p>Loading...</p>
        </div>
      </section>
    );
  }

  const metricLabels = {
    blue_iq: 'Blue IQ',
    balance: 'Balance',
    rotation: 'Rotation',
    pressure: 'Pressure',
    edging: 'Edging'
  };

  const chartData = {
    labels: trends.time_series.dates,
    datasets: [
      {
        label: metricLabels[selectedMetric],
        data: trends.time_series[selectedMetric],
        borderColor: '#6aafff',
        backgroundColor: 'rgba(106, 175, 255, 0.1)',
        tension: 0.4,
        fill: true,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: '#6aafff',
        pointBorderColor: '#fff',
        pointBorderWidth: 2
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: false
      },
      tooltip: {
        backgroundColor: '#0a1728',
        titleColor: '#eef6ff',
        bodyColor: '#8fa3bd',
        borderColor: 'rgba(106, 175, 255, 0.2)',
        borderWidth: 1,
        padding: 12,
        displayColors: false
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 240,
        grid: {
          color: 'rgba(255, 255, 255, 0.08)'
        },
        ticks: {
          color: '#8fa3bd',
          font: {
            size: 12
          }
        }
      },
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.08)'
        },
        ticks: {
          color: '#8fa3bd',
          font: {
            size: 11
          },
          maxRotation: 45,
          minRotation: 45
        }
      }
    }
  };

  const getTrendEmoji = (trend) => {
    if (trend === 'improving') return '↗️';
    if (trend === 'declining') return '↘️';
    return '→';
  };

  const getTrendClass = (trend) => {
    if (trend === 'improving') return 'trend-up';
    if (trend === 'declining') return 'trend-down';
    return 'trend-stable';
  };

  const currentMetricData = trends.current_vs_first[selectedMetric];

  return (
    <section className="section-block">
      <div className="section-heading">
        <div>
          <h2>Performance Trends</h2>
          <p>Your progress over the last {trends.total_runs} runs</p>
        </div>
        <select
          value={selectedMetric}
          onChange={(e) => setSelectedMetric(e.target.value)}
          style={{
            padding: '8px 12px',
            background: '#091426',
            border: '1px solid var(--line-soft)',
            borderRadius: '8px',
            color: 'var(--text)',
            fontSize: '14px'
          }}
        >
          <option value="blue_iq">Blue IQ</option>
          <option value="balance">Balance</option>
          <option value="rotation">Rotation</option>
          <option value="pressure">Pressure</option>
          <option value="edging">Edging</option>
        </select>
      </div>

      {/* Chart */}
      <div style={{ background: '#0a1728', padding: '24px', borderRadius: '16px', height: '400px' }}>
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginTop: '24px' }}>
        <div style={{ background: '#0a1728', padding: '18px', borderRadius: '12px', border: '1px solid var(--line-soft)' }}>
          <p className="eyebrow">Overall Trend</p>
          <strong className={getTrendClass(trends.trends[selectedMetric])} style={{ fontSize: '24px', display: 'block', marginTop: '8px' }}>
            {trends.trends[selectedMetric].toUpperCase()} {getTrendEmoji(trends.trends[selectedMetric])}
          </strong>
        </div>

        <div style={{ background: '#0a1728', padding: '18px', borderRadius: '12px', border: '1px solid var(--line-soft)' }}>
          <p className="eyebrow">Best Run</p>
          <strong style={{ fontSize: '24px', display: 'block', marginTop: '8px', color: 'var(--green)' }}>
            Run #{trends.highlights[selectedMetric].best.run}
          </strong>
          <small style={{ color: 'var(--muted)', display: 'block', marginTop: '4px' }}>
            Score: {trends.highlights[selectedMetric].best.score} • {trends.highlights[selectedMetric].best.date}
          </small>
        </div>

        <div style={{ background: '#0a1728', padding: '18px', borderRadius: '12px', border: '1px solid var(--line-soft)' }}>
          <p className="eyebrow">Improvement Rate</p>
          <strong style={{ fontSize: '24px', display: 'block', marginTop: '8px', color: trends.improvement_rates[selectedMetric] > 0 ? 'var(--green)' : 'var(--red)' }}>
            {trends.improvement_rates[selectedMetric] > 0 ? '+' : ''}{trends.improvement_rates[selectedMetric]}%
          </strong>
          <small style={{ color: 'var(--muted)', display: 'block', marginTop: '4px' }}>Per run average</small>
        </div>
      </div>

      {/* Progress Summary */}
      <div style={{ marginTop: '24px', padding: '24px', background: '#0e1d31', borderRadius: '16px', border: '1px solid var(--line)' }}>
        <h3 style={{ margin: '0 0 20px 0', fontSize: '18px' }}>First Run vs Current Run</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr auto', gap: '20px', alignItems: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <small style={{ color: 'var(--muted)', display: 'block', marginBottom: '8px' }}>
              First Run ({trends.date_range.start})
            </small>
            <div style={{ fontSize: '48px', fontWeight: 'bold', color: 'var(--blue)' }}>
              {currentMetricData.first}
            </div>
          </div>

          <div style={{ fontSize: '32px', color: 'var(--muted)' }}>→</div>

          <div style={{ textAlign: 'center' }}>
            <small style={{ color: 'var(--muted)', display: 'block', marginBottom: '8px' }}>
              Current Run ({trends.date_range.end})
            </small>
            <div style={{ fontSize: '48px', fontWeight: 'bold', color: 'var(--blue)' }}>
              {currentMetricData.current}
            </div>
          </div>

          <div style={{ textAlign: 'center', padding: '20px', background: 'rgba(106, 175, 255, 0.1)', borderRadius: '12px' }}>
            <small style={{ color: 'var(--muted)', display: 'block', marginBottom: '8px' }}>Total Change</small>
            <div style={{ fontSize: '32px', fontWeight: 'bold', color: currentMetricData.change > 0 ? 'var(--green)' : 'var(--red)' }}>
              {currentMetricData.change > 0 ? '+' : ''}{currentMetricData.change}
            </div>
            <small style={{ fontSize: '18px', color: currentMetricData.change_percent > 0 ? 'var(--green)' : 'var(--red)' }}>
              ({currentMetricData.change_percent > 0 ? '+' : ''}{currentMetricData.change_percent}%)
            </small>
          </div>
        </div>
      </div>
    </section>
  );
}
