import { useEffect, useRef } from 'react'
import { Flame, Lock, ArrowRight } from 'lucide-react'
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

  return (
    <div className="floating-tags" ref={containerRef} aria-hidden="true">
      {FLOATING_TAGS.map((tag) => (
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

function App() {
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
            Advanced AI-driven vulnerability analysis for hardware, IoT, and
            complex PDF reports. Stay ahead of threats with automated security
            compliance.
          </p>

          <div className="hero-buttons">
            <a href="#scan" className="btn-primary">
              Start Free Scan <ArrowRight size={18} />
            </a>
            <a href="#upload" className="btn-secondary">
              <Lock size={16} />
              Upload PDF
            </a>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App
