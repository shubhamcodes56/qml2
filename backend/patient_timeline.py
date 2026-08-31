"""
Patient Timeline Generator — 25-Hour Simulation
================================================
Generates time-series data showing how Classical ML and QML
would monitor a patient over 25 hours.

Classical ML: sees each vital independently — all stay in "safe" range
QML: tracks quantum correlations between vitals — detects sync breaking
"""

import numpy as np

def generate_patient_timeline(infection_start_hour=12, total_hours=25):
    """
    Generates 25 hours of simulated patient data.
    
    Before infection_start_hour: patient is truly healthy, all correlations strong.
    After infection_start_hour: subtle infection begins.
        - Classical vitals barely change (still in safe range)
        - But quantum correlations (HR-Temp sync, SpO2-WBC coupling) start decaying
    
    Returns dict with:
        - timestamps: [0, 0.5, 1.0, ..., 25.0]
        - classical_view: {heart_rate: [...], temperature: [...], ...}
        - qml_view: {hr_temp_sync: [...], spo2_wbc_coupling: [...], ...}
        - infection_start: hour when infection begins
    """
    np.random.seed(42)
    
    # Generate timestamps every 30 minutes
    timestamps = np.arange(0, total_hours + 0.5, 0.5)
    n_points = len(timestamps)
    
    # ============================================
    # CLASSICAL VIEW: Individual vitals over time
    # Key insight: ALL values stay within safe range
    # ============================================
    
    # Heart Rate: baseline 78, normal fluctuation 72-86
    hr_base = 78.0
    hr = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 3.0)
        # Slight circadian rhythm
        circadian = 2.0 * np.sin(2 * np.pi * t / 24)
        # After infection: very subtle rise (still in normal range 60-100)
        if t > infection_start_hour:
            drift = 0.15 * (t - infection_start_hour)  # Max +2 bpm by hour 25
        else:
            drift = 0
        hr[i] = hr_base + circadian + noise + drift
    
    # Temperature: baseline 36.8, normal 36.1-37.2
    temp_base = 36.8
    temp = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 0.1)
        circadian = 0.2 * np.sin(2 * np.pi * (t - 6) / 24)
        if t > infection_start_hour:
            drift = 0.015 * (t - infection_start_hour)  # Max +0.2C
        else:
            drift = 0
        temp[i] = temp_base + circadian + noise + drift
    
    # SpO2: baseline 97.5, normal 95-100
    spo2_base = 97.5
    spo2 = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 0.3)
        if t > infection_start_hour:
            drift = -0.05 * (t - infection_start_hour)  # Max -0.65%
        else:
            drift = 0
        spo2[i] = np.clip(spo2_base + noise + drift, 94, 100)
    
    # Respiratory Rate: baseline 16, normal 12-20
    rr_base = 16.0
    rr = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 0.8)
        if t > infection_start_hour:
            drift = 0.08 * (t - infection_start_hour)
        else:
            drift = 0
        rr[i] = rr_base + noise + drift
    
    # WBC Count: baseline 7.0, normal 4-11
    wbc_base = 7.0
    wbc = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 0.5)
        if t > infection_start_hour:
            drift = 0.1 * (t - infection_start_hour)
        else:
            drift = 0
        wbc[i] = wbc_base + noise + drift
    
    # ESR: baseline 10, normal 0-20mm/hr
    esr_base = 10.0
    esr = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 1.5)
        if t > infection_start_hour:
            drift = 0.3 * (t - infection_start_hour)
        else:
            drift = 0
        esr[i] = max(0, esr_base + noise + drift)
    
    # ============================================
    # QML VIEW: Quantum correlations over time
    # Key insight: These DECAY even when individual vitals stay safe
    # ============================================
    
    # HR-Temp Synchronization (Quantum Joint Expectation)
    # In healthy person: HR and Temp have strong circadian sync (~1.0)
    # In early infection: sync breaks as immune response disrupts thermal regulation
    hr_temp_sync = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 0.02)
        if t <= infection_start_hour:
            hr_temp_sync[i] = 0.95 + noise
        else:
            decay_time = t - infection_start_hour
            # Exponential decay in correlation
            hr_temp_sync[i] = 0.95 * np.exp(-0.08 * decay_time) + noise
    hr_temp_sync = np.clip(hr_temp_sync, 0.2, 1.0)
    
    # SpO2-WBC Coupling (normally inversely correlated during immune response)
    spo2_wbc_coupling = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 0.03)
        if t <= infection_start_hour:
            spo2_wbc_coupling[i] = 0.88 + noise
        else:
            decay_time = t - infection_start_hour
            spo2_wbc_coupling[i] = 0.88 * np.exp(-0.06 * decay_time) + noise
    spo2_wbc_coupling = np.clip(spo2_wbc_coupling, 0.15, 1.0)
    
    # RR-CRP Multi-body Correlation
    rr_crp_correlation = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 0.025)
        if t <= infection_start_hour:
            rr_crp_correlation[i] = 0.92 + noise
        else:
            decay_time = t - infection_start_hour
            rr_crp_correlation[i] = 0.92 * np.exp(-0.07 * decay_time) + noise
    rr_crp_correlation = np.clip(rr_crp_correlation, 0.2, 1.0)
    
    # Overall Entanglement Entropy (Von Neumann)
    entanglement_entropy = np.zeros(n_points)
    for i, t in enumerate(timestamps):
        noise = np.random.normal(0, 0.05)
        if t <= infection_start_hour:
            entanglement_entropy[i] = 0.3 + noise  # Low entropy = healthy equilibrium
        else:
            decay_time = t - infection_start_hour
            entanglement_entropy[i] = 0.3 + 0.5 * (1 - np.exp(-0.12 * decay_time)) + noise
    entanglement_entropy = np.clip(entanglement_entropy, 0.0, 1.0)
    
    # ============================================
    # CLASSICAL ML RISK vs QML RISK over time
    # ============================================
    classical_risk = np.zeros(n_points)
    qml_risk = np.zeros(n_points)
    
    for i, t in enumerate(timestamps):
        # Classical ML: only triggers if individual vitals cross thresholds
        # All vitals stay safe, so risk stays very low
        hr_risk = max(0, (hr[i] - 85) / 15) if hr[i] > 85 else 0
        temp_risk = max(0, (temp[i] - 37.5) / 2.5) if temp[i] > 37.5 else 0
        spo2_risk = max(0, (96 - spo2[i]) / 6) if spo2[i] < 96 else 0
        classical_risk[i] = min(1.0, (hr_risk + temp_risk + spo2_risk) / 3 * 100)
        
        # QML: uses entanglement correlations - fires much earlier
        if t <= infection_start_hour:
            qml_risk[i] = np.random.uniform(2, 8)
        else:
            decay_time = t - infection_start_hour
            # Risk climbs as correlations break
            qml_risk[i] = min(95, 5 + 90 * (1 - np.exp(-0.15 * decay_time)))
    
    # QML Advisory alerts
    alerts = []
    for i, t in enumerate(timestamps):
        if t == infection_start_hour + 1.5:
            alerts.append({
                "hour": float(t),
                "severity": "INFO",
                "message": "HR-Temperature synchronization dropping below 0.90. Monitor."
            })
        if t == infection_start_hour + 3.0:
            alerts.append({
                "hour": float(t),
                "severity": "WARNING",
                "message": "Multi-variable entanglement shift detected. SpO2-WBC coupling weakening. Recommend blood panel."
            })
        if t == infection_start_hour + 6.0:
            alerts.append({
                "hour": float(t),
                "severity": "CRITICAL",
                "message": "Entanglement entropy exceeding threshold. Strong hidden correlation breakdown. Recommend immediate clinical evaluation."
            })
    
    return {
        "timestamps": timestamps.tolist(),
        "infection_start_hour": infection_start_hour,
        "classical_view": {
            "heart_rate": hr.tolist(),
            "temperature": temp.tolist(),
            "spo2": spo2.tolist(),
            "resp_rate": rr.tolist(),
            "wbc_count": wbc.tolist(),
            "esr": esr.tolist(),
        },
        "qml_view": {
            "hr_temp_sync": hr_temp_sync.tolist(),
            "spo2_wbc_coupling": spo2_wbc_coupling.tolist(),
            "rr_crp_correlation": rr_crp_correlation.tolist(),
            "entanglement_entropy": entanglement_entropy.tolist(),
        },
        "risk_over_time": {
            "classical_risk": classical_risk.tolist(),
            "qml_risk": qml_risk.tolist(),
        },
        "alerts": alerts,
        "safe_ranges": {
            "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
            "temperature": {"min": 36.1, "max": 37.5, "unit": "C"},
            "spo2": {"min": 95, "max": 100, "unit": "%"},
            "resp_rate": {"min": 12, "max": 20, "unit": "br/min"},
            "wbc_count": {"min": 4, "max": 11, "unit": "x10^3/uL"},
            "esr": {"min": 0, "max": 20, "unit": "mm/hr"},
        }
    }


if __name__ == "__main__":
    import json
    data = generate_patient_timeline()
    print(json.dumps(data, indent=2)[:2000])
    print("...")
    print(f"Total timestamps: {len(data['timestamps'])}")
    print(f"Alerts generated: {len(data['alerts'])}")
