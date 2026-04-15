import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EMOTION_META = {
  Anger:     { emoji: '😠', color: '#e74c3c', bg: 'rgba(231,76,60,0.15)' },
  Disgust:   { emoji: '🤢', color: '#8e44ad', bg: 'rgba(142,68,173,0.15)' },
  Fear:      { emoji: '😨', color: '#2980b9', bg: 'rgba(41,128,185,0.15)' },
  Happiness: { emoji: '😄', color: '#27ae60', bg: 'rgba(39,174,96,0.15)' },
  Neutral:   { emoji: '😐', color: '#95a5a6', bg: 'rgba(149,165,166,0.15)' },
  Sadness:   { emoji: '😢', color: '#3498db', bg: 'rgba(52,152,219,0.15)' },
}

export default function App() {
  const [file, setFile]         = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)
  const [modelOk, setModelOk]   = useState(null)
  const inputRef                = useRef(null)

  useEffect(() => {
    fetch(`${API}/status`)
      .then(r => r.json())
      .then(d => setModelOk(d.model_loaded))
      .catch(() => setModelOk(false))
  }, [])

  const acceptFile = useCallback((f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    acceptFile(e.dataTransfer.files[0])
  }, [acceptFile])

  const onDragOver  = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onInputChange = (e) => acceptFile(e.target.files[0])

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API}/predict`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Server error')
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const meta = result ? EMOTION_META[result.emotion] : null

  return (
    <div className="page">

      {/* ── Header ── */}
      <header className="header">
        <span className="pill">Transformer Encoder · CREMA-D · 6 Emotions</span>
        <h1>Speech Emotion<br />Recognition</h1>
        <p className="subtitle">
          Upload an audio clip and the AI detects the speaker's emotion.
        </p>
      </header>

      {/* ── Warning banner ── */}
      {modelOk === false && (
        <div className="banner">
          <strong>Backend not reachable</strong> — run{' '}
          <code>uvicorn backend:app --port 8000</code> and place{' '}
          <code>best_transformer.pth</code> in the project root.
        </div>
      )}

      <main className="main">

        {/* ── Upload card ── */}
        <section className="card">
          <h2 className="card-title">Upload Audio</h2>

          <div
            className={`drop-zone${dragging ? ' dz-active' : ''}${file ? ' dz-filled' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".wav,.mp3,.flac,.ogg,.m4a"
              onChange={onInputChange}
              style={{ display: 'none' }}
            />

            {file ? (
              <div className="file-info">
                <span className="file-icon">🎵</span>
                <div>
                  <p className="file-name">{file.name}</p>
                  <p className="file-size">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
            ) : (
              <>
                <span className="dz-icon">🎤</span>
                <p className="dz-title">Drop your audio file here</p>
                <p className="dz-sub">or click to browse · WAV, MP3, FLAC, OGG, M4A</p>
              </>
            )}
          </div>

          <div className="btn-row">
            {file && (
              <button className="btn btn-ghost" onClick={reset}>Clear</button>
            )}
            <button
              className="btn btn-primary"
              onClick={analyze}
              disabled={!file || loading}
            >
              {loading
                ? <><span className="spinner" /> Analyzing…</>
                : 'Analyze Emotion'}
            </button>
          </div>

          {error && <p className="error-msg">⚠ {error}</p>}
        </section>

        {/* ── Result card ── */}
        {result && meta && (
          <section
            className="card result-card"
            style={{ '--accent': meta.color, '--accent-bg': meta.bg }}
          >
            <div className="result-hero">
              <span className="result-emoji">{meta.emoji}</span>
              <div>
                <h2 className="result-emotion" style={{ color: meta.color }}>
                  {result.emotion}
                </h2>
                <p className="result-conf">{result.confidence}% confidence</p>
              </div>
            </div>

            <div className="bars">
              {result.all_emotions
                .slice()
                .sort((a, b) => b.probability - a.probability)
                .map(({ label, probability }) => {
                  const m = EMOTION_META[label]
                  const isTop = label === result.emotion
                  return (
                    <div key={label} className={`bar-row${isTop ? ' bar-row-top' : ''}`}>
                      <span className="bar-label">{m.emoji} {label}</span>
                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${probability}%`,
                            background: m.color,
                            opacity: isTop ? 1 : 0.4,
                          }}
                        />
                      </div>
                      <span className="bar-pct">{probability.toFixed(1)}%</span>
                    </div>
                  )
                })}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        ITE 5410 Deep Learning · Humber Polytechnic
      </footer>
    </div>
  )
}
