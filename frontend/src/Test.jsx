// Simple test component to verify React is working
export default function Test() {
  console.log("Test component rendering");
  
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'column',
      gap: '20px',
      color: 'white',
      textAlign: 'center',
      padding: '20px'
    }}>
      <h1 style={{ fontSize: '48px', margin: 0 }}>✅ React is Working!</h1>
      <p>If you can see this, React is rendering correctly.</p>
      <div style={{
        padding: '20px',
        background: 'rgba(255,255,255,0.1)',
        borderRadius: '12px',
        maxWidth: '600px'
      }}>
        <h2>Next Steps:</h2>
        <ol style={{ textAlign: 'left', lineHeight: '1.8' }}>
          <li>Open browser console (F12) and check for errors</li>
          <li>Verify backend is running at http://localhost:8000</li>
          <li>Check that API_BASE is set correctly</li>
          <li>Replace Test with App in main.jsx</li>
        </ol>
      </div>
      <p style={{ color: '#6aafff' }}>
        API Base: {import.meta.env.VITE_API_BASE_URL || 'Not set'}
      </p>
    </div>
  );
}
