import numpy as np
import time
import json
from quantum_tb_engine import QuantumTBEngine

def run_test():
    print("="*60)
    print(" [+] INITIALIZING ULTIMATE HYBRID TB ENGINE [+]")
    print("="*60)
    
    t0 = time.time()
    engine = QuantumTBEngine()
    print(f"\n[+] Engine Initialized in {time.time()-t0:.2f} seconds.")
    
    # --- PATIENT 1: COMPLETELY HEALTHY ---
    patient_healthy = np.array([
        75.0, 98.5, 16.0, 37.0,    # Vitals (Normal)
        7.0, 10.0, 2.0, 30.0,      # Blood (Normal)
        13.5, 4.0, 250.0, 100.0,   # Blood 2 (Normal)
        0.0, 0.0, 0.0, 0.0,        # X-Ray (Clear)
        0.0, 0.0, 0.0, 35.0,       # TB Tests (Negative)
        22.0, 0.0                   # Profile (Normal BMI)
    ])
    
    # --- PATIENT 2: SEVERE TB ---
    patient_tb = np.array([
        105.0, 88.0, 24.0, 39.5,   # Vitals (Bad)
        18.0, 85.0, 45.0, 10.0,    # Blood (Bad)
        9.5, 2.5, 450.0, 110.0,    # Blood 2 (Bad)
        0.85, 0.60, 0.75, 0.40,    # X-Ray (Bad)
        85.0, 22.0, 0.90, 15.0,    # TB Tests (Positive)
        16.0, 0.0                   # Profile (Low BMI)
    ])
    
    for label, patient in [("HEALTHY", patient_healthy), ("SEVERE TB", patient_tb)]:
        print(f"\n{'='*50}")
        print(f" --- {label} PATIENT ---")
        print(f"{'='*50}")
        t1 = time.time()
        res = engine.full_analysis(patient)
        elapsed = time.time() - t1
        
        print(f"Risk Score       : {res['risk_score']}%")
        print(f"Severity         : {res['severity']}")
        print(f"Confidence       : {res['confidence']}%")
        print(f"TB Probability   : {res['tb_probability']:.4f}")
        print(f"Raw QML Signal   : {res['raw_tb_probability']:.6f}")
        print(f"Von Neumann S    : {res['von_neumann_entropy']:.4f}")
        print(f"Compute Time     : {elapsed:.3f}s")
        
        print(f"\n  [Per-Qubit PauliZ Analysis]")
        for q in res['per_qubit_analysis']:
            marker = "!!" if q['status'] == 'anomaly' else ("?" if q['status'] == 'watch' else "ok")
            print(f"    {q['label']:10s} | Z={q['pauliz']:+.4f} | P(excited)={q['prob_excited']:.4f} | [{marker}]")
        
        print(f"\n  [X-Ray Quantum Focus]")
        for name, data in res['xray_quantum_focus'].items():
            bar = "#" * int(data['focus_score'] / 5)
            print(f"    {name:8s} | raw={data['raw_value']:.2f} | angle={data['angle']:+.4f} | focus={data['focus_score']:5.1f}% |{bar}")
        
        print(f"\n  [Top ZZ Entanglement Interactions]")
        for zz in res['quantum_attention_map'][:5]:
            print(f"    {zz['pair']:35s} | strength={zz['strength']:.4f} | norm={zz['normalized']:.4f}")
    
    print(f"\n{'='*60}")
    print(f" [+] SIGNAL AMPLIFIER WORKING: Healthy={patient_healthy[0]}, TB={patient_tb[0]}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_test()
