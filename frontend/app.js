const API_BASE = "";

// -- Clock --
function updateClock() {
    const now = new Date();
    document.getElementById("currentTime").textContent = now.toLocaleTimeString("en-US", { hour12: false });
}
setInterval(updateClock, 1000);

// -- Plotly Initializations --
const layoutBase = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#888', family: "'JetBrains Mono', monospace" },
    margin: { t: 20, b: 20, l: 20, r: 20 }
};

function initPlots() {
    Plotly.newPlot('densityMatrixPlot', [{
        z: Array(32).fill(Array(32).fill(0)),
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: false
    }], {
        ...layoutBase,
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false, autorange: 'reversed' }
    }, { displayModeBar: false, responsive: true });

    Plotly.newPlot('probabilityPlot', [{
        z: Array(4).fill(Array(8).fill(0)),
        type: 'surface',
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

// -- Update ICU Plots --
function updateQuantumVisuals(qml_analysis) {
    const { density_matrix_heatmap, probabilities_top, circuit_angles } = qml_analysis;

    Plotly.update('densityMatrixPlot', { z: [density_matrix_heatmap] });

    // Reshape top 32 probabilities into 4x8 grid
    const z_surface = [];
    for (let i = 0; i < 4; i++) {
        z_surface.push(probabilities_top.slice(i * 8, (i + 1) * 8));
    }
    Plotly.update('probabilityPlot', { z: [z_surface] });

    // Update all 8 Circuit Angles
    for (let i = 0; i < 8; i++) {
        const el = document.getElementById(`theta${i}`);
        if (el && circuit_angles[i] !== undefined) {
            el.textContent = circuit_angles[i].toFixed(4);
            el.style.color = Math.abs(circuit_angles[i]) > 0.5 ? "var(--neon-yellow)" : "#fff";
        }
    }
}

// -- QML Risk Widget --
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
    const text = document.getElementById("qmlStatusText");
    if (severity === "CRITICAL") { dot.className = "status-dot red"; text.textContent = "QML RISK: CRITICAL"; }
    else if (severity === "WARNING") { dot.className = "status-dot yellow"; text.textContent = "QML RISK: WARNING"; }
    else { dot.className = "status-dot green"; text.textContent = "QML RISK: LOW"; }
}

// -- Terminal Log --
function logTerminal(msg) {
    const term = document.getElementById("terminalOutput");
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    term.innerHTML += `<div><span style="color:#555">[${time}]</span> ${msg}</div>`;
    term.scrollTop = term.scrollHeight;
}

// -- API Polling (ICU Tab) --
let driftActive = false;

async function fetchLiveData() {
    try {
        const res = await fetch(`${API_BASE}/live-vitals`);
        const data = await res.json();
        
        if (data.status === "starting") return;
        
        updateQuantumVisuals(data.qml_analysis);
        renderRisk(data.qml_analysis);
        
        const cDot = document.getElementById("classicalStatusDot");
        const cText = document.getElementById("classicalStatusText");
        
        if (data.is_anomaly_injected) {
            cText.textContent = `VITALS DRIFTING: HR=${data.vitals["Heart Rate"]}, SpO2=${data.vitals["SpO2"]}`;
            cDot.className = "status-dot yellow";
            
            if (!driftActive) {
                driftActive = true;
                logTerminal("<span style='color:var(--neon-yellow)'>Micro-Drift Injected. Classical vitals remain in safe range.</span>");
            }
            
            if (data.qml_analysis.severity === "CRITICAL" && Math.random() < 0.3) {
                logTerminal("<span style='color:var(--neon-red)'>[QML] Mass coherence distortion detected in density matrix.</span>");
            }
        } else {
            cText.textContent = "CLASSICAL VITALS: NORMAL";
            cDot.className = "status-dot green";
            driftActive = false;
        }

    } catch (err) {
        console.error(err);
    }
}

// -- Controls --
document.getElementById("btnInject").addEventListener("click", async () => {
    await fetch(`${API_BASE}/trigger-drift`, { method: "POST" });
});

document.getElementById("btnReset").addEventListener("click", async () => {
    await fetch(`${API_BASE}/reset`, { method: "POST" });
    logTerminal("Patient reset to baseline. Quantum state returning to equilibrium.");
});

// -- Init --
window.addEventListener("load", () => {
    initPlots();
    logTerminal("Quantum Hilbert Space mapping initialized (256-dim).");
    logTerminal("Calculating 8-qubit Density Matrix...");
    setInterval(fetchLiveData, 1000);
});

// -- TAB SWITCHING --
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    
    event.target.classList.add('active');
    document.getElementById(tabId).classList.add('active');
}

// =============================================
// TB COMPARISON TAB LOGIC (ADVANCED)
// =============================================
function initTBPlots() {
    Plotly.newPlot('tbDensityPlot', [{
        z: Array(32).fill(Array(32).fill(0)),
        type: 'heatmap',
        colorscale: [[0, '#0a0a0c'], [0.3, '#1a0040'], [0.6, '#8b5cf6'], [1, '#ff2244']],
        showscale: false
    }], {
        ...layoutBase,
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false, autorange: 'reversed' }
    }, { displayModeBar: false, responsive: true });

    Plotly.newPlot('tbProbPlot', [{
        z: Array(4).fill(Array(8).fill(0)),
        type: 'surface',
        colorscale: [[0, '#0a0a0c'], [0.25, '#1e0060'], [0.5, '#8b5cf6'], [0.75, '#ffcc00'], [1, '#ff2244']],
        showscale: false
    }], {
        ...layoutBase,
        scene: {
            xaxis: { showgrid: false, showticklabels: false, zeroline: false, title: '' },
            yaxis: { showgrid: false, showticklabels: false, zeroline: false, title: '' },
            zaxis: { showgrid: true, gridcolor: '#222', range: [0, 0.1], title: '' },
            camera: { eye: { x: 1.8, y: 1.2, z: 0.6 } }
        }
    }, { displayModeBar: false, responsive: true });
}

async function fetchTBPatient() {
    const btn = document.getElementById('btnFetchTB');
    btn.textContent = "COMPUTING QUANTUM STATE...";
    btn.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/compare-tb-patient`);
        const data = await res.json();
        
        if(data.error) {
            alert("Error: " + data.error);
            return;
        }

        // 1. Populate Vitals Grid (22 features)
        const grid = document.getElementById('vitalsGrid');
        grid.innerHTML = '';
        const warningFeatures = ['esr', 'crp', 'xray_opacity', 'xray_cavity', 'xray_nodule', 'xray_pleural', 'ada_level', 'mantoux_mm', 'sputum_afb'];
        for (const [key, value] of Object.entries(data.patient_data)) {
            const isWarning = warningFeatures.includes(key) && value > 0.1;
            grid.innerHTML += `<div class="vital-box ${isWarning ? 'warning' : ''}">
                <span>${key.replace(/_/g, ' ').toUpperCase()}</span>
                <b>${typeof value === 'number' ? value.toFixed(2) : value}</b>
            </div>`;
        }

        // 2. Update Classical Card
        const cl = data.classical_ml;
        const clRisk = (cl.risk_score * 100);
        document.getElementById('clValue').textContent = clRisk.toFixed(1);
        const clOffset = (2 * Math.PI * 54) * (1 - cl.risk_score);
        document.getElementById('clProgress').style.strokeDashoffset = clOffset;
        const clBadge = document.getElementById('clDiag');
        clBadge.textContent = cl.diagnosis;
        clBadge.className = "diag-badge " + (cl.risk_score >= 0.5 ? "danger" : "healthy");

        // 3. Update Quantum Card
        const qm = data.quantum_ml;
        document.getElementById('qmValue').textContent = qm.risk_score;
        const qmOffset = (2 * Math.PI * 54) * (1 - qm.risk_score / 100);
        document.getElementById('qmProgress').style.strokeDashoffset = qmOffset;
        const qmBadge = document.getElementById('qmDiag');
        qmBadge.textContent = qm.severity;
        qmBadge.className = "diag-badge " + (qm.risk_score >= 50 ? "danger" : "healthy");

        // 4. X-Ray Quantum Focus
        const xrayPanel = document.getElementById('xrayPanel');
        xrayPanel.style.display = 'block';
        if (qm.xray_quantum_focus) {
            for (const [name, data_xr] of Object.entries(qm.xray_quantum_focus)) {
                const nameCapitalized = name.charAt(0).toUpperCase() + name.slice(1);
                const fill = document.getElementById(`xray${nameCapitalized}Fill`);
                const val = document.getElementById(`xray${nameCapitalized}Val`);
                if (fill) {
                    fill.style.width = data_xr.focus_score + '%';
                    fill.style.background = data_xr.focus_score > 60 ? 'var(--neon-red)' : (data_xr.focus_score > 30 ? 'var(--neon-yellow)' : 'var(--neon-green)');
                }
                if (val) val.textContent = data_xr.focus_score.toFixed(1) + '% (raw: ' + data_xr.raw_value + ')';
            }
        }

        // 5. Per-Qubit Analysis
        const qubitPanel = document.getElementById('qubitPanel');
        qubitPanel.style.display = 'block';
        const qubitGrid = document.getElementById('qubitGrid');
        qubitGrid.innerHTML = '';
        if (qm.per_qubit_analysis) {
            qm.per_qubit_analysis.forEach(q => {
                const statusClass = q.status === 'anomaly' ? 'qubit-anomaly' : (q.status === 'watch' ? 'qubit-watch' : 'qubit-normal');
                qubitGrid.innerHTML += `<div class="qubit-box ${statusClass}">
                    <span class="qubit-label">${q.label}</span>
                    <span class="qubit-z">Z = ${q.pauliz > 0 ? '+' : ''}${q.pauliz.toFixed(4)}</span>
                    <div class="qubit-bar"><div class="qubit-fill" style="width:${q.prob_excited*100}%"></div></div>
                    <span class="qubit-status">${q.status.toUpperCase()}</span>
                </div>`;
            });
        }

        // 6. ZZ Entanglement Interactions
        const zzPanel = document.getElementById('zzPanel');
        zzPanel.style.display = 'block';
        const zzGrid = document.getElementById('zzGrid');
        zzGrid.innerHTML = '';
        if (qm.quantum_attention_map) {
            qm.quantum_attention_map.forEach(zz => {
                const barWidth = Math.min(zz.normalized * 100, 100);
                zzGrid.innerHTML += `<div class="zz-item">
                    <span class="zz-pair">${zz.pair}</span>
                    <div class="zz-bar"><div class="zz-fill" style="width:${barWidth}%"></div></div>
                    <span class="zz-strength">${zz.strength.toFixed(2)}</span>
                </div>`;
            });
        }

        // 7. Update Plotly Charts
        const vizRow = document.getElementById('vizRow');
        vizRow.style.display = 'flex';
        
        if (!window._tbPlotsInitialized) {
            initTBPlots();
            window._tbPlotsInitialized = true;
        }
        
        // Update density matrix
        if (qm.density_matrix_heatmap) {
            Plotly.update('tbDensityPlot', { z: [qm.density_matrix_heatmap] });
        }
        
        // Update probability landscape (16x16 -> surface)
        if (qm.probability_landscape) {
            const landscape = qm.probability_landscape;
            // Take every 4th row for a 4x16 view, then subsample to 4x8
            const sub = [];
            for (let i = 0; i < 16; i += 4) {
                const row = [];
                for (let j = 0; j < 16; j += 2) {
                    row.push(landscape[i][j]);
                }
                sub.push(row);
            }
            Plotly.update('tbProbPlot', { z: [sub] });
        }

    } catch (err) {
        console.error(err);
        alert("Backend not running! Start with: python api.py");
    } finally {
        btn.textContent = "ANALYZE HIDDEN TIER-1 PATIENT";
        btn.disabled = false;
    }
}
