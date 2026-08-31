const API_BASE = "http://localhost:8000";

// ── Clock ──
function updateClock() {
    const now = new Date();
    document.getElementById("currentTime").textContent = now.toLocaleTimeString("en-US", { hour12: false });
}
setInterval(updateClock, 1000);

// ── Plotly Initializations ──
const layoutBase = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#888', family: "'JetBrains Mono', monospace" },
    margin: { t: 20, b: 20, l: 20, r: 20 }
};

function initPlots() {
    // 1. Density Matrix Heatmap (16x16)
    Plotly.newPlot('densityMatrixPlot', [{
        z: Array(16).fill(Array(16).fill(0)),
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: false
    }], {
        ...layoutBase,
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false, autorange: 'reversed' }
    }, { displayModeBar: false, responsive: true });

    // 2. Probability Amplitude Landscape (4x4 Surface/Contour)
    Plotly.newPlot('probabilityPlot', [{
        z: Array(4).fill(Array(4).fill(0)),
        type: 'surface', // 3D Surface
        colorscale: 'Electric',
        showscale: false
    }], {
        ...layoutBase,
        scene: {
            xaxis: { showgrid: false, showticklabels: false, zeroline: false },
            yaxis: { showgrid: false, showticklabels: false, zeroline: false },
            zaxis: { showgrid: true, gridcolor: '#333', range: [0, 0.5] },
            camera: { eye: { x: 1.5, y: 1.5, z: 0.5 } }
        }
    }, { displayModeBar: false, responsive: true });
}

// ── Update Plots ──
function updateQuantumVisuals(qml_analysis) {
    const { density_matrix_heatmap, probabilities, circuit_angles } = qml_analysis;

    // Update Density Matrix
    Plotly.update('densityMatrixPlot', { z: [density_matrix_heatmap] });

    // Reshape 16 probabilities into 4x4 grid for topographical rendering
    const z_surface = [];
    for (let i = 0; i < 4; i++) {
        z_surface.push(probabilities.slice(i * 4, (i + 1) * 4));
    }
    Plotly.update('probabilityPlot', { z: [z_surface] });

    // Update Circuit Angles (θ)
    for (let i = 0; i < 4; i++) {
        const el = document.getElementById(`theta${i}`);
        el.textContent = circuit_angles[i].toFixed(4);
        
        // Highlight if angle is shifting significantly from 0
        if (Math.abs(circuit_angles[i]) > 0.5) {
            el.style.color = "var(--neon-yellow)";
        } else {
            el.style.color = "#fff";
        }
    }
}

// ── QML Risk Widget ──
function renderRisk(qml_analysis) {
    const { risk_score, severity } = qml_analysis;
    
    document.getElementById("riskValue").textContent = risk_score;
    
    const circumference = 2 * Math.PI * 54;
    const offset = circumference * (1 - risk_score / 100);
    document.getElementById("riskProgress").style.strokeDashoffset = offset;

    const badge = document.getElementById("severityBadge");
    badge.textContent = severity;
    
    const widget = document.getElementById("riskWidget");
    widget.className = "widget widget-risk severity-" + severity;
    
    const dot = document.getElementById("qmlStatusDot");
    if (severity === "CRITICAL") dot.className = "status-dot red";
    else if (severity === "WARNING") dot.className = "status-dot yellow";
    else dot.className = "status-dot green";
}

// ── Terminal Log ──
function logTerminal(msg) {
    const term = document.getElementById("terminalOutput");
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    term.innerHTML += `<div><span style="color:#555">[${time}]</span> ${msg}</div>`;
    term.scrollTop = term.scrollHeight;
}

// ── API Polling ──
let driftActive = false;

async function fetchLiveData() {
    try {
        const res = await fetch(`${API_BASE}/live-vitals`);
        const data = await res.json();
        
        if (data.status === "starting") return;
        
        updateQuantumVisuals(data.qml_analysis);
        renderRisk(data.qml_analysis);
        
        // Classical Status
        const cDot = document.getElementById("classicalStatusDot");
        const cText = document.getElementById("classicalStatusText");
        
        if (data.is_anomaly_injected) {
            cText.textContent = `VITALS DRIFTING: HR=${data.vitals["Heart Rate"]}, SpO2=${data.vitals["SpO2"]}`;
            // It remains green/yellow to show classical monitors don't see it as critical yet
            cDot.className = "status-dot yellow";
            cText.style.color = "var(--text-main)";
            
            if (!driftActive) {
                driftActive = true;
                logTerminal("<span style='color:var(--neon-yellow)'>Micro-Drift Injected. Classical vitals remain in safe range.</span>");
            }
            
            // Log quantum mechanics
            if (data.qml_analysis.severity === "CRITICAL" && Math.random() < 0.3) {
                logTerminal("<span style='color:var(--neon-red)'>[QML] Mass coherence distortion detected in density matrix.</span>");
            }
        } else {
            cText.textContent = "CLASSICAL VITALS: NORMAL";
            cDot.className = "status-dot green";
            cText.style.color = "var(--text-main)";
            driftActive = false;
        }

    } catch (err) {
        console.error(err);
    }
}

// ── Controls ──
document.getElementById("btnInject").addEventListener("click", async () => {
    await fetch(`${API_BASE}/trigger-drift`, { method: "POST" });
});

document.getElementById("btnReset").addEventListener("click", async () => {
    await fetch(`${API_BASE}/reset`, { method: "POST" });
    logTerminal("Patient reset to baseline. Quantum state returning to equilibrium.");
});

// ── Init ──
window.addEventListener("load", () => {
    initPlots();
    logTerminal("Quantum Hilbert Space mapping initialized.");
    logTerminal("Calculating 16-dimensional Density Matrix...");
    setInterval(fetchLiveData, 1000);
});

// -- TAB SWITCHING --
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    
    event.target.classList.add('active');
    document.getElementById(tabId).classList.add('active');
}

// -- TB COMPARISON LOGIC --
async function fetchTBPatient() {
    const btn = document.getElementById('btnFetchTB');
    btn.textContent = "SIMULATING QUANTUM STATE...";
    btn.disabled = true;
    
    try {
        const res = await fetch(${API_BASE}/compare-tb-patient);
        const data = await res.json();
        
        if(data.error) {
            alert("Error: " + data.error);
            return;
        }

        // 1. Populate Vitals Grid
        const grid = document.getElementById('vitalsGrid');
        grid.innerHTML = '';
        for (const [key, value] of Object.entries(data.patient_data)) {
            const isWarning = (key === 'heart_rate' && value > 85) || (key === 'temperature' && value > 37.2);
            grid.innerHTML += <div class="vital-box  + (isWarning ? 'warning' : '') + ">
                <span></span>
                <b></b>
            </div>;
        }

        // 2. Update Classical Card
        const cl = data.classical_ml;
        document.getElementById('clValue').textContent = (cl.risk_score * 100).toFixed(1);
        const clOffset = (2 * Math.PI * 54) * (1 - cl.risk_score);
        document.getElementById('clProgress').style.strokeDashoffset = clOffset;
        const clBadge = document.getElementById('clDiag');
        clBadge.textContent = cl.diagnosis;
        clBadge.className = "diag-badge " + (cl.risk_score >= 0.5 ? "danger" : "healthy");

        // 3. Update Quantum Card
        const qm = data.quantum_ml;
        document.getElementById('qmValue').textContent = (qm.risk_score * 100).toFixed(1);
        const qmOffset = (2 * Math.PI * 54) * (1 - qm.risk_score);
        document.getElementById('qmProgress').style.strokeDashoffset = qmOffset;
        const qmBadge = document.getElementById('qmDiag');
        qmBadge.textContent = qm.diagnosis;
        qmBadge.className = "diag-badge " + (qm.risk_score >= 0.5 ? "danger" : "healthy");

        // 4. Update Qdrant Results
        const qGrid = document.getElementById('qdrantResults');
        qGrid.innerHTML = '';
        if (data.qdrant_similar && data.qdrant_similar.length > 0) {
            data.qdrant_similar.forEach(p => {
                qGrid.innerHTML += <div class="qdrant-card">
                    <span style="color:#94a3b8; font-size:0.8rem">Patient ID: #</span><br>
                    <b style="color:#fff; font-size:1.1rem"></b><br>
                    <span style="color:#38bdf8; font-size:0.8rem">Vector Match: %</span>
                </div>;
            });
        } else {
            qGrid.innerHTML = '<span style="color:#94a3b8">No similar patients found.</span>';
        }

    } catch (err) {
        console.error(err);
    } finally {
        btn.textContent = "ANALYZE HIDDEN TIER-1 PATIENT";
        btn.disabled = false;
    }
}
