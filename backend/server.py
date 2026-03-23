"""
FastAPI server for PDF vulnerability assessment with WebSocket real-time updates.
Processes PDFs, streams progress via WebSocket, returns assessment report.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Optional
import gc
import torch

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Import voracle modules
from voracle import (
    extract_device_features,
    InMemoryVulnerabilitySystem,
    IndexConfig,
    load_device_features,
)

app = FastAPI(title="Vulnerability Assessment Server")

# Store for WebSocket connections (for multi-client support if needed)
active_connections = []


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        """Send message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending message: {e}")

    async def send(self, websocket: WebSocket, message: Dict):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")


manager = ConnectionManager()

# Global state for vulnerability system (cached)
vulnerability_system = None


def log_progress(message: str, level: str = "info"):
    """Helper to create log messages."""
    return {
        "type": "log",
        "level": level,
        "message": message,
    }


def log_status(status: str, details: str = ""):
    """Helper to create status messages."""
    return {
        "type": "status",
        "status": status,
        "details": details,
    }


def log_report(report: Dict):
    """Helper to create report message."""
    return {
        "type": "report",
        "data": report,
    }


def log_error(message: str):
    """Helper to create error messages."""
    return {
        "type": "error",
        "message": message,
    }


class WebSocketProgressHandler:
    """Captures extraction progress and sends via WebSocket."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.current_chunk = 0
        self.total_chunks = 0

    async def on_extraction_start(self, total_chunks: int):
        """Called when extraction starts."""
        self.total_chunks = total_chunks
        await manager.send(
            self.websocket,
            log_status("Extraction started", f"Processing {total_chunks} chunks"),
        )
        await manager.send(
            self.websocket,
            log_progress(f"Split into {total_chunks} chunks"),
        )

    async def on_chunk_processed(self, chunk_num: int, vendor: str = "", model: str = ""):
        """Called after each chunk is processed."""
        self.current_chunk = chunk_num
        msg = f"Processing chunk {chunk_num}/{self.total_chunks}"
        if vendor or model:
            msg += f" - {vendor} {model}".strip()
        await manager.send(self.websocket, log_progress(msg))

    async def on_extraction_complete(self, result_count: int):
        """Called when extraction completes."""
        await manager.send(
            self.websocket,
            log_status("Extraction complete", f"Valid extractions: {result_count}"),
        )

    async def on_index_build_start(self):
        """Called when building FAISS index starts."""
        await manager.send(
            self.websocket,
            log_status("Building vector index", "Generating embeddings..."),
        )

    async def on_index_build_complete(self):
        """Called when FAISS index is built."""
        await manager.send(
            self.websocket,
            log_status("Index built", "Vector index ready"),
        )

    async def on_assessment_start(self):
        """Called when assessment starts."""
        await manager.send(
            self.websocket,
            log_status("Assessment started", "Retrieving vulnerabilities..."),
        )

    async def on_assessment_complete(self):
        """Called when assessment completes."""
        await manager.send(
            self.websocket,
            log_status("Assessment complete", "Generating report..."),
        )


@app.get("/")
async def root():
    """Serve simple HTML frontend."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vulnerability Assessment</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .upload-area {
                border: 2px dashed #3498db;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                background: white;
                margin-bottom: 20px;
                transition: all 0.3s;
            }
            .upload-area:hover { background: #ecf0f1; border-color: #2980b9; }
            .upload-area.dragover { background: #e8f4f8; border-color: #2980b9; }
            input[type="file"] { display: none; }
            button {
                background: #3498db;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s;
            }
            button:hover { background: #2980b9; }
            button:disabled { background: #95a5a6; cursor: not-allowed; }
            .logs {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                max-height: 400px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 13px;
            }
            .log-entry {
                padding: 8px;
                margin: 5px 0;
                border-radius: 4px;
                border-left: 4px solid #3498db;
            }
            .log-info { border-left-color: #3498db; background: #ecf0f1; }
            .log-status { border-left-color: #27ae60; background: #d5f4e6; color: #27ae60; font-weight: bold; }
            .log-error { border-left-color: #e74c3c; background: #fadbd8; color: #e74c3c; }
            .report {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
                display: none;
            }
            .report.show { display: block; }
            .device-info {
                background: #ecf0f1;
                padding: 15px;
                border-radius: 4px;
                margin-bottom: 20px;
            }
            .vulnerability {
                background: #f9f9f9;
                border-left: 4px solid #e74c3c;
                padding: 12px;
                margin: 10px 0;
                border-radius: 4px;
            }
            .severity-critical { border-left-color: #c0392b; }
            .severity-high { border-left-color: #e74c3c; }
            .severity-medium { border-left-color: #f39c12; }
            .severity-low { border-left-color: #f1c40f; }
            .badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
                margin-right: 5px;
            }
            .badge-critical { background: #c0392b; color: white; }
            .badge-high { background: #e74c3c; color: white; }
            .badge-medium { background: #f39c12; color: white; }
            .badge-low { background: #f1c40f; color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔐 Vulnerability Assessment System</h1>
                <p>Upload a relay/device PDF to extract features and assess vulnerabilities</p>
            </div>

            <div class="upload-area" id="uploadArea">
                <h3>📄 Drop PDF here or click to select</h3>
                <p style="margin-top: 10px; color: #7f8c8d;">or</p>
                <button onclick="document.getElementById('fileInput').click()">Select File</button>
                <input type="file" id="fileInput" accept=".pdf" />
            </div>

            <div class="logs" id="logs" style="display: none;">
                <h3>Processing Log</h3>
                <div id="logContent"></div>
            </div>

            <div class="report" id="report">
                <h2>📊 Assessment Report</h2>
                <div id="reportContent"></div>
            </div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const logsDiv = document.getElementById('logs');
            const logContent = document.getElementById('logContent');
            const reportDiv = document.getElementById('report');
            const reportContent = document.getElementById('reportContent');
            let ws = null;

            // File upload handling
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    processFile(files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    processFile(e.target.files[0]);
                }
            });

            function processFile(file) {
                if (file.type !== 'application/pdf') {
                    alert('Please select a PDF file');
                    return;
                }

                uploadFile(file);
            }

            function uploadFile(file) {
                uploadArea.style.display = 'none';
                logsDiv.style.display = 'block';
                reportDiv.classList.remove('show');
                logContent.innerHTML = '';
                reportContent.innerHTML = '';

                // Create FormData
                const formData = new FormData();
                formData.append('file', file);

                // Fetch with streaming
                fetch('/assess', {
                    method: 'POST',
                    body: formData,
                }).then(response => {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    function read() {
                        reader.read().then(({ done, value }) => {
                            if (done) return;

                            const chunk = decoder.decode(value);
                            const lines = chunk.split('\\n');

                            lines.forEach(line => {
                                if (line.startsWith('data: ')) {
                                    try {
                                        const msg = JSON.parse(line.substring(6));
                                        handleMessage(msg);
                                    } catch (e) {
                                        console.error('Error parsing message:', e);
                                    }
                                }
                            });

                            read();
                        });
                    }

                    read();
                }).catch(error => {
                    addLog('Error: ' + error.message, 'error');
                });
            }

            function handleMessage(msg) {
                if (msg.type === 'log') {
                    addLog(msg.message, msg.level);
                } else if (msg.type === 'status') {
                    addLog(`[${msg.status}] ${msg.details}`, 'status');
                } else if (msg.type === 'error') {
                    addLog(msg.message, 'error');
                } else if (msg.type === 'report') {
                    displayReport(msg.data);
                    uploadArea.style.display = 'block';
                    logsDiv.style.display = 'none';
                }
            }

            function addLog(message, level = 'info') {
                const entry = document.createElement('div');
                entry.className = `log-entry log-${level}`;
                entry.textContent = message;
                logContent.appendChild(entry);
                logContent.scrollTop = logContent.scrollHeight;
            }

            function displayReport(report) {
                reportDiv.classList.add('show');

                const device = report.device || {};
                const summary = report.assessment_summary || {};
                const vulns = report.top_vulnerabilities || [];

                let html = `
                    <div class="device-info">
                        <h3>Device Information</h3>
                        <p><strong>Vendor:</strong> ${device.vendor || 'Unknown'}</p>
                        <p><strong>Model:</strong> ${device.model || 'Unknown'}</p>
                    </div>

                    <div class="device-info">
                        <h3>Assessment Summary</h3>
                        <p><strong>Total Vulnerabilities Found:</strong> ${summary.total_matches || 0}</p>
                        <p>
                            <span class="badge badge-critical">CRITICAL: ${summary.critical_count || 0}</span>
                            <span class="badge badge-high">HIGH: ${summary.high_count || 0}</span>
                            <span class="badge badge-medium">MEDIUM: ${summary.medium_count || 0}</span>
                            <span class="badge badge-low">LOW: ${summary.low_count || 0}</span>
                        </p>
                        <p><strong>Max Risk Score:</strong> ${summary.max_risk_score || 0}</p>
                        <p><strong>Avg Risk Score:</strong> ${summary.avg_risk_score || 0}</p>
                    </div>

                    <h3>Top Vulnerabilities</h3>
                `;

                vulns.forEach(v => {
                    const severityClass = `severity-${v.severity.toLowerCase()}`;
                    html += `
                        <div class="vulnerability ${severityClass}">
                            <h4>${v.cve_id} <span class="badge badge-${v.severity.toLowerCase()}">${v.severity}</span></h4>
                            <p><strong>CVSS Score:</strong> ${v.cvss_score} | <strong>Risk Score:</strong> ${v.risk_score}</p>
                            <p><strong>Problem Type:</strong> ${v.problem_type}</p>
                            <p><strong>Description:</strong> ${v.description}</p>
                        </div>
                    `;
                });

                reportContent.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """)


@app.post("/assess")
async def assess_pdf(file: UploadFile = File(...)):
    """
    Assessment endpoint.
    Accepts PDF upload and streams progress + final report via Server-Sent Events.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    if not file.size or file.size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Process in background and stream results
    async def event_generator():
        try:
            yield f"data: {json.dumps(log_status('Processing started', 'Extracting device features...'))}\n\n"

            # Step 1: Extract device features from PDF
            yield f"data: {json.dumps(log_progress('Loading extraction model...'))}\n\n"
            device_features = extract_device_features(temp_path)

            yield f"data: {json.dumps(log_progress('Device features extracted successfully'))}\n\n"
            vendor = device_features.get("vendor")
            model = device_features.get("model")
            vendor_model_msg = f"Vendor: {vendor}, Model: {model}"
            yield f"data: {json.dumps(log_progress(vendor_model_msg))}\n\n"

            # Step 2: Initialize vulnerability system
            yield f"data: {json.dumps(log_status('Building assessment system', 'Initializing in-memory vulnerability database...'))}\n\n"

            global vulnerability_system
            if vulnerability_system is None:
                yield f"data: {json.dumps(log_progress('Loading vulnerability system...'))}\n\n"
                vulnerability_system = InMemoryVulnerabilitySystem(
                    index_config=IndexConfig()
                )
                yield f"data: {json.dumps(log_progress('Building FAISS index from CVE data...'))}\n\n"

                cve_path = "cvelistV5/cves"
                if Path(cve_path).exists():
                    yield f"data: {json.dumps(log_progress('Found local CVE data, building index...'))}\n\n"
                    vulnerability_system.build_index(cve_path)
                else:
                    yield f"data: {json.dumps(log_status('Warning', 'CVE data not found locally, using minimal dataset'))}\n\n"

            # Step 3: Run vulnerability assessment
            yield f"data: {json.dumps(log_status('Assessment started', 'Querying vulnerabilities...'))}\n\n"

            report = vulnerability_system.assess_device(
                device_features=device_features,
                k=20,
                top_n=10,
            )

            yield f"data: {json.dumps(log_status('Assessment complete', 'Generating final report...'))}\n\n"

            # Step 4: Send final report
            yield f"data: {json.dumps(log_report(report))}\n\n"

        except Exception as e:
            yield f"data: {json.dumps(log_error(f'Error during assessment: {str(e)}'))}\n\n"
        finally:
            # Cleanup
            try:
                Path(temp_path).unlink()
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
