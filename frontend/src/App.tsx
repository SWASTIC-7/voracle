import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { Flame, Bot, LoaderCircle } from 'lucide-react'
import jsPDF from 'jspdf'
import './App.css'

const FLOATING_TAGS = [
  'Hardware Scan', 'IoT Security', 'CVE Database', 'Deep Scan',
  'Firmware Analysis', 'PDF Reports', 'Network Audit', 'Zero-Day',
  'Penetration Testing', 'OWASP Top 10', 'Cloud Security', 'API Security',
  'Compliance', 'Threat Intel', 'Malware Detection', 'Risk Assessment',
  'Vulnerability', 'Container Security', 'SAST', 'DAST',
]

function FloatingTags() {
  const containerRef = useRef<HTMLDivElement>(null)
  const mouseRef = useRef({ x: -9999, y: -9999 })

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const pills = container.querySelectorAll<HTMLElement>('.floating-pill')
    const velocities: { x: number; y: number }[] = []
    const positions: { x: number; y: number }[] = []

    const cw = container.offsetWidth
    const ch = container.offsetHeight
    const cols = 5
    const rows = Math.ceil(pills.length / cols)

    pills.forEach((pill, i) => {
      const speed = 0.15 + Math.random() * 0.25
      const angle = Math.random() * Math.PI * 2
      velocities.push({ x: Math.cos(angle) * speed, y: Math.sin(angle) * speed })

      const col = i % cols
      const row = Math.floor(i / cols)
      const cellW = cw / cols
      const cellH = ch / rows
      const px = cellW * col + Math.random() * (cellW - pill.offsetWidth)
      const py = cellH * row + Math.random() * (cellH - pill.offsetHeight)

      positions.push({
        x: Math.max(0, Math.min(px, cw - pill.offsetWidth)),
        y: Math.max(0, Math.min(py, ch - pill.offsetHeight)),
      })
      pill.style.transform = `translate(${positions[i].x}px, ${positions[i].y}px)`
    })

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect()
      mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top }
    }

    const handleMouseLeave = () => {
      mouseRef.current = { x: -9999, y: -9999 }
    }

    container.addEventListener('mousemove', handleMouseMove)
    container.addEventListener('mouseleave', handleMouseLeave)

    const REPEL_RADIUS = 150
    const REPEL_STRENGTH = 2.5

    let animId: number
    const animate = () => {
      const cw = container.offsetWidth
      const ch = container.offsetHeight
      const mouse = mouseRef.current

      pills.forEach((pill, i) => {
        const pw = pill.offsetWidth
        const ph = pill.offsetHeight

        // cursor repulsion
        const cx = positions[i].x + pw / 2
        const cy = positions[i].y + ph / 2
        const dx = cx - mouse.x
        const dy = cy - mouse.y
        const dist = Math.sqrt(dx * dx + dy * dy)

        if (dist < REPEL_RADIUS && dist > 0) {
          const force = (1 - dist / REPEL_RADIUS) * REPEL_STRENGTH
          velocities[i].x += (dx / dist) * force
          velocities[i].y += (dy / dist) * force
        }

        // friction to keep things smooth
        velocities[i].x *= 0.985
        velocities[i].y *= 0.985

        // min drift speed so pills don't stop completely
        const speed = Math.sqrt(velocities[i].x ** 2 + velocities[i].y ** 2)
        const minSpeed = 0.12
        if (speed < minSpeed && speed > 0) {
          velocities[i].x = (velocities[i].x / speed) * minSpeed
          velocities[i].y = (velocities[i].y / speed) * minSpeed
        }

        positions[i].x += velocities[i].x
        positions[i].y += velocities[i].y

        // bounce off edges
        if (positions[i].x < 0) { positions[i].x = 0; velocities[i].x *= -0.7 }
        if (positions[i].x + pw > cw) { positions[i].x = cw - pw; velocities[i].x *= -0.7 }
        if (positions[i].y < 0) { positions[i].y = 0; velocities[i].y *= -0.7 }
        if (positions[i].y + ph > ch) { positions[i].y = ch - ph; velocities[i].y *= -0.7 }

        pill.style.transform = `translate(${positions[i].x}px, ${positions[i].y}px)`
      })

      animId = requestAnimationFrame(animate)
    }
    animId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(animId)
      container.removeEventListener('mousemove', handleMouseMove)
      container.removeEventListener('mouseleave', handleMouseLeave)
    }
  }, [])

  const isMobile = window.innerWidth <= 768
  const tags = isMobile ? FLOATING_TAGS.slice(0, 8) : FLOATING_TAGS

  return (
    <div className="floating-tags" ref={containerRef} aria-hidden="true">
      {tags.map((tag) => (
        <span
          key={tag}
          className="floating-pill"
          style={{ position: 'absolute', left: 0, top: 0, opacity: 0.35 + Math.random() * 0.35 }}
        >
          {tag}
        </span>
      ))}
    </div>
  )
}

type ReportSummary = {
  total_matches?: number
  critical_count?: number
  high_count?: number
  medium_count?: number
  low_count?: number
  max_risk_score?: number
  avg_risk_score?: number
}

type Vulnerability = {
  cve_id?: string
  severity?: string
  cvss_score?: number
  risk_score?: number
  problem_type?: string
  description?: string
}

type AssessmentReport = {
  device?: {
    vendor?: string
    model?: string
  }
  assessment_summary?: ReportSummary
  top_vulnerabilities?: Vulnerability[]
}

type ServerEvent =
  | { type: 'log'; level?: string; message?: string }
  | { type: 'status'; status?: string; details?: string }
  | { type: 'error'; message?: string }
  | { type: 'report'; data?: AssessmentReport }

type LogItem = {
  level: 'info' | 'status' | 'error'
  message: string
}

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedFileName, setSelectedFileName] = useState<string>('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [logs, setLogs] = useState<LogItem[]>([])
  const [report, setReport] = useState<AssessmentReport | null>(null)

  const addLog = (message: string, level: LogItem['level'] = 'info') => {
    if (!message) {
      return
    }
    setLogs((prev) => [...prev, { message, level }])
  }

  const handleServerEvent = (payload: ServerEvent) => {
    if (payload.type === 'log') {
      addLog(payload.message ?? 'Log update received', 'info')
      return
    }

    if (payload.type === 'status') {
      const statusText = payload.status ?? 'Status'
      const detailsText = payload.details ? ` ${payload.details}` : ''
      addLog(`[${statusText}]${detailsText}`, 'status')
      return
    }

    if (payload.type === 'error') {
      addLog(payload.message ?? 'Unknown error from server', 'error')
      return
    }

    if (payload.type === 'report') {
      setReport(payload.data ?? null)
      addLog('Assessment report received.', 'status')
      setIsProcessing(false)
    }
  }

  const startAssessment = async (file: File) => {
    setSelectedFileName(file.name)
    setLogs([])
    setReport(null)
    setIsProcessing(true)
    addLog(`Uploading ${file.name}...`)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/assess', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`)
      }

      if (!response.body) {
        throw new Error('No response stream available from server')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() ?? ''

        for (const eventChunk of events) {
          const dataLines = eventChunk
            .split('\n')
            .filter((line) => line.startsWith('data: '))
            .map((line) => line.slice(6))

          if (!dataLines.length) {
            continue
          }

          const eventPayload = dataLines.join('\n')
          try {
            const parsed = JSON.parse(eventPayload) as ServerEvent
            handleServerEvent(parsed)
          } catch {
            addLog('Received malformed stream data from server', 'error')
          }
        }
      }

      if (buffer.trim().startsWith('data: ')) {
        try {
          const parsed = JSON.parse(buffer.trim().slice(6)) as ServerEvent
          handleServerEvent(parsed)
        } catch {
          addLog('Unable to parse final stream payload', 'error')
        }
      }

      setIsProcessing(false)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected request error'
      addLog(`Assessment failed: ${message}`, 'error')
      setIsProcessing(false)
    }
  }

  const onFileSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    if (file.type !== 'application/pdf') {
      addLog('Please upload a valid PDF file.', 'error')
      return
    }

    await startAssessment(file)

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const summary = report?.assessment_summary
  const topVulnerabilities = report?.top_vulnerabilities ?? []
  const latestLog = logs[logs.length - 1]

  const criticalCount = summary?.critical_count ?? topVulnerabilities.filter((v) => v.severity === 'CRITICAL').length
  const highCount = summary?.high_count ?? topVulnerabilities.filter((v) => v.severity === 'HIGH').length
  const mediumCount = summary?.medium_count ?? topVulnerabilities.filter((v) => v.severity === 'MEDIUM').length
  const lowCount = summary?.low_count ?? topVulnerabilities.filter((v) => v.severity === 'LOW').length
  const severityTotal = criticalCount + highCount + mediumCount + lowCount
  const pieTotal = Math.max(severityTotal, 1)
  const criticalEnd = (criticalCount / pieTotal) * 100
  const highEnd = criticalEnd + (highCount / pieTotal) * 100
  const mediumEnd = highEnd + (mediumCount / pieTotal) * 100

  const riskSeries = topVulnerabilities
    .filter((v) => (v.risk_score ?? 0) > 0)
    .map((v) => ({
      cve: v.cve_id ?? 'Unknown',
      risk: Number(v.risk_score ?? 0),
    }))

  const maxRisk = riskSeries.length ? Math.max(...riskSeries.map((item) => item.risk)) * 1.1 : 1
  const formatCveLabel = (cve: string) => (cve.length > 14 ? `${cve.slice(0, 14)}...` : cve)

  const handleDownloadPdf = () => {
    if (!report) {
      return
    }

    const doc = new jsPDF({ unit: 'pt', format: 'a4' })
    const left = 40
    const right = 555
    const lineHeight = 18
    let y = 44

    const addWrappedText = (text: string, indent = 0, fontSize = 11) => {
      doc.setFontSize(fontSize)
      const width = right - (left + indent)
      const lines = doc.splitTextToSize(text, width)
      lines.forEach((line: string) => {
        if (y > 790) {
          doc.addPage()
          y = 44
        }
        doc.text(line, left + indent, y)
        y += lineHeight
      })
    }

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(16)
    doc.text('Vulnerability Assessment Report', left, y)
    y += 26

    doc.setFont('helvetica', 'normal')
    addWrappedText(`Vendor: ${report.device?.vendor ?? 'Unknown'}`)
    addWrappedText(`Model: ${report.device?.model ?? 'Unknown'}`)
    y += 6

    doc.setFont('helvetica', 'bold')
    addWrappedText('Summary', 0, 13)
    doc.setFont('helvetica', 'normal')
    addWrappedText(`Total Matches: ${summary?.total_matches ?? 0}`)
    addWrappedText(`Critical: ${criticalCount} | High: ${highCount} | Medium: ${mediumCount} | Low: ${lowCount}`)
    addWrappedText(`Max Risk: ${summary?.max_risk_score ?? 0} | Avg Risk: ${summary?.avg_risk_score ?? 0}`)
    y += 6

    doc.setFont('helvetica', 'bold')
    addWrappedText('Top Vulnerabilities', 0, 13)
    doc.setFont('helvetica', 'normal')

    topVulnerabilities.forEach((item, index) => {
      addWrappedText(`${index + 1}. ${item.cve_id ?? 'Unknown CVE'} (${item.severity ?? 'Unknown'})`, 0, 11)
      addWrappedText(`CVSS: ${item.cvss_score ?? 0} | Risk: ${item.risk_score ?? 0}`, 12, 10)
      addWrappedText(`Type: ${item.problem_type ?? 'N/A'}`, 12, 10)
      addWrappedText(`Description: ${item.description ?? 'No description provided'}`, 12, 10)
      y += 6
    })

    const baseName = selectedFileName ? selectedFileName.replace(/\.pdf$/i, '') : 'assessment'
    doc.save(`${baseName}-assessment-report.pdf`)
  }

  return (
    <div className="app">
      {/* Navbar */}
      <nav className="navbar">
        <div className="navbar-inner">
          <div className="logo">
            <div className="logo-icon">
              <svg viewBox="0 0 32 32" width="32" height="32" fill="none">
                <circle cx="16" cy="16" r="15" stroke="#E8720C" strokeWidth="2" />
                <circle cx="16" cy="16" r="6" fill="#E8720C" />
              </svg>
            </div>
            <span className="logo-text">Voracle</span>
          </div>

          <ul className="nav-links">
            <li><a href="#solutions">Solutions</a></li>
            <li><a href="#features">Features</a></li>
            <li><a href="#pricing">Pricing</a></li>
            <li><a href="#about">About</a></li>
          </ul>

          <div className="nav-actions">
            {/* <a href="#login" className="login-link">Login</a> */}
            <a href="#scan" className="btn-primary-sm">Start Free Scan</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <FloatingTags />

        <div className="hero-content">
          <div className="badge">
            <Flame size={14} />
            <span>NEW: CVE-2024 REAL-TIME TRACKING</span>
          </div>

          <h1 className="hero-title">
            Secure Your Digital<br />
            Assets with <span className="highlight">AI-Powered</span>
            <br />
            Scanning
          </h1>

          <p className="hero-subtitle">
            Advanced AI-driven vulnerability assessment for Digital Devices in PowerGrid substations.
          </p>

          {/* <div className="hero-buttons">
            <button
              className="btn-primary"
              type="button"
              disabled={isProcessing}
              onClick={() => fileInputRef.current?.click()}
            >
              {isProcessing ? 'Analyzing...' : 'Start Free Scan'} <ArrowRight size={18} />
            </button>
          </div> */}

          {isProcessing && (
            <div className="agent-console">
              <div className="agent-line">
                <Bot size={16} />
                <p className={`agent-current agent-current-${latestLog.level} ${isProcessing ? 'agent-current-processing' : ''}`}>
                  {latestLog.message}
                  {isProcessing && <span className="processing-dots" aria-hidden="true" />}
                </p>
                <LoaderCircle size={14} className="agent-loader" />
              </div>
            </div>
          )}

          <div className="upload-inline">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={onFileSelected}
              disabled={isProcessing}
              className="file-input"
              id="scan-file"
            />
            <button
              className="btn-secondary"
              type="button"
              disabled={isProcessing}
              onClick={() => fileInputRef.current?.click()}
            >
              {isProcessing ? 'Processing PDF...' : 'Upload PDF File'}
            </button>
            {selectedFileName && <span className="selected-file">{selectedFileName}</span>}
          </div>

          {report && (
            <div className="report-panel">
              <div className="report-header">
                <h3>Assessment Report</h3>
                <button className="download-btn" type="button" onClick={handleDownloadPdf}>
                  Download Report
                </button>
              </div>

              <div className="report-grid">
                <div className="report-block">
                  <h4>Device</h4>
                  <p>Vendor: {report.device?.vendor ?? 'Unknown'}</p>
                  <p>Model: {report.device?.model ?? 'Unknown'}</p>
                </div>
                <div className="report-block">
                  <h4>Summary</h4>
                  <p>Total Matches: {summary?.total_matches ?? 0}</p>
                  <p>Critical: {summary?.critical_count ?? 0}</p>
                  <p>High: {summary?.high_count ?? 0}</p>
                  <p>Medium: {summary?.medium_count ?? 0}</p>
                  <p>Low: {summary?.low_count ?? 0}</p>
                  <p>Max Risk: {summary?.max_risk_score ?? 0}</p>
                  <p>Avg Risk: {summary?.avg_risk_score ?? 0}</p>
                </div>
              </div>

              <div className="chart-grid">
                <div className="report-block">
                  <h4>Severity Distribution</h4>
                  <div className="pie-wrap">
                    <div
                      className="severity-pie"
                      style={{
                        background: `conic-gradient(
                          #c43a2f 0% ${criticalEnd}%,
                          #ea8f2f ${criticalEnd}% ${highEnd}%,
                          #f1c04d ${highEnd}% ${mediumEnd}%,
                          #88b04b ${mediumEnd}% 100%
                        )`,
                      }}
                      role="img"
                      aria-label="Pie chart showing critical, high, medium and low CVE counts"
                    >
                      <div className="severity-center">
                        <span className="severity-total">{severityTotal}</span>
                        <span className="severity-caption">Total CVEs</span>
                      </div>
                    </div>
                    <div className="pie-legend">
                      <p><span className="dot dot-critical" />Critical: {criticalCount}</p>
                      <p><span className="dot dot-high" />High: {highCount}</p>
                      <p><span className="dot dot-medium" />Medium: {mediumCount}</p>
                      <p><span className="dot dot-low" />Low: {lowCount}</p>
                    </div>
                  </div>
                </div>

                <div className="report-block">
                  <h4>Risk Score per CVE</h4>
                  <div className="risk-chart">
                    {!riskSeries.length && <p className="empty-text">No risk score data found.</p>}
                    {riskSeries.map((item) => (
                      <div className="risk-col" key={item.cve}>
                        <p className="risk-value">{item.risk.toFixed(2)}</p>
                        <div className="risk-col-track">
                          <div className="risk-col-fill" style={{ height: `${(item.risk / maxRisk) * 100}%` }} />
                        </div>
                        <p className="risk-label" title={item.cve}>{formatCveLabel(item.cve)}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

export default App
