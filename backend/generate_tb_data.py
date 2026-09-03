"""
TB Patient Dataset Generator — ULTIMATE "Judge-Proof" Clinical EHR Edition v2
===============================================================================
Generates 3,000,000 unique, medically-accurate TB patient records.

22 Clinical Features (5 Groups) — Based on WHO TB Report 2023, Harrison's 
Principles of Internal Medicine, and Pai et al. (2016) Nature Reviews.

NEW in v2:
- 6 new clinical parameters: BMI, Platelet_Count, Blood_Sugar, Sputum_AFB, GeneXpert_Ct, Treatment_Days
- Smoking history modifier (25% smokers -> worse lung scores)
- Seasonal variation (TB incidence peaks in monsoon/winter)
- False Positives: Pneumonia, Sarcoidosis, Lung Cancer mimics
- Sensor noise (2%) and missing values (5%)
- Full EHR metadata: Patient_ID, Hospital, Age, Gender, Comorbidities, Admission_Date

Output:
- tb_ehr_full.parquet (Full 3M realistic EHR dataset)
- tb_ehr_10k_sample.csv (10k readable sample for Excel inspection)
- tb_features.pt / tb_labels.pt / tb_tiers.pt (Numerical matrices for ML/QML pipeline)
"""

import numpy as np
import pandas as pd
import torch
import os
import time
import json
import uuid

# ═══════════════════════════════════════════════════════════════════════════════
# 22 CLINICAL FEATURE SPECIFICATIONS — Every value is medically grounded
# Sources: WHO TB Report 2023, Harrison's Principles, Pai et al. 2016
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_SPECS = {
    # ── A. CORE VITALS (4) ──
    "heart_rate":     {"mean": 78.0,  "std": 8.0,  "lo": 40,   "hi": 180,  "unit": "bpm",
                       "medical": "Resting HR. TB -> sympathetic activation -> mild tachycardia (~8 bpm/°C fever)."},
    "spo2":           {"mean": 97.5,  "std": 0.8,  "lo": 70,   "hi": 100,  "unit": "%",
                       "medical": "Oxygen saturation. Pulmonary TB damages alveoli -> subtle desaturation."},
    "resp_rate":      {"mean": 16.0,  "std": 2.0,  "lo": 8,    "hi": 40,   "unit": "breaths/min",
                       "medical": "Breathing rate. Compensatory increase when gas exchange impaired."},
    "temperature":    {"mean": 36.8,  "std": 0.3,  "lo": 35.0, "hi": 42.0, "unit": "°C",
                       "medical": "Core temp. TB hallmark: low-grade evening fever (37.2-38.0°C)."},
    # ── B. BLOOD MARKERS (8) ──
    "wbc_count":      {"mean": 7.0,   "std": 1.5,  "lo": 2.0,  "hi": 30.0, "unit": "×10³/µL",
                       "medical": "White blood cells. Mild leukocytosis in active TB."},
    "esr":            {"mean": 10.0,  "std": 5.0,  "lo": 0,    "hi": 120,  "unit": "mm/hr",
                       "medical": "Erythrocyte Sedimentation Rate. Elevated 40-100+ in active TB."},
    "crp":            {"mean": 2.0,   "std": 1.0,  "lo": 0,    "hi": 200,  "unit": "mg/L",
                       "medical": "C-Reactive Protein. Acute phase reactant. Rises sharply in infection."},
    "lymphocyte_pct": {"mean": 30.0,  "std": 5.0,  "lo": 5,    "hi": 60,   "unit": "%",
                       "medical": "Lymphocyte %. TB causes lymphocyte-predominant response."},
    "hemoglobin":     {"mean": 13.5,  "std": 1.2,  "lo": 5.0,  "hi": 20.0, "unit": "g/dL",
                       "medical": "Hb. Chronic TB causes anemia of chronic disease."},
    "albumin":        {"mean": 4.0,   "std": 0.4,  "lo": 1.0,  "hi": 5.5,  "unit": "g/dL",
                       "medical": "Serum albumin. Drops in chronic TB (malnutrition, protein wasting)."},
    "platelet_count": {"mean": 250.0, "std": 60.0, "lo": 50,   "hi": 600,  "unit": "×10³/µL",
                       "medical": "Platelets. Reactive thrombocytosis in active TB (elevated in ~30% cases)."},
    "blood_sugar":    {"mean": 100.0, "std": 15.0, "lo": 50,   "hi": 400,  "unit": "mg/dL",
                       "medical": "Fasting glucose. TB-DM comorbidity is bidirectional risk factor."},
    # ── C. X-RAY RADIOLOGICAL SCORES (4) ──
    "xray_opacity":   {"mean": 0.08,  "std": 0.04,  "lo": 0.0, "hi": 1.0, "unit": "CNN score",
                       "medical": "Overall lung opacity. Upper lobe predominance in reactivation TB."},
    "xray_cavity":    {"mean": 0.04,  "std": 0.025, "lo": 0.0, "hi": 1.0, "unit": "CNN score",
                       "medical": "Cavitation. Hallmark of active pulmonary TB. High bacterial load."},
    "xray_nodule":    {"mean": 0.06,  "std": 0.03,  "lo": 0.0, "hi": 1.0, "unit": "CNN score",
                       "medical": "Micronodular pattern. Miliary TB: random 1-3mm nodules."},
    "xray_pleural":   {"mean": 0.05,  "std": 0.025, "lo": 0.0, "hi": 1.0, "unit": "CNN score",
                       "medical": "Pleural effusion/thickening. Common in primary TB pleurisy."},
    # ── D. TB-SPECIFIC DIAGNOSTIC TESTS (4) ──
    "ada_level":      {"mean": 15.0,  "std": 8.0,  "lo": 0,    "hi": 150, "unit": "U/L",
                       "medical": "Adenosine Deaminase. Gold standard for TB pleural effusion. >40 U/L = TB."},
    "mantoux_mm":     {"mean": 5.0,   "std": 4.0,  "lo": 0,    "hi": 30,  "unit": "mm",
                       "medical": "Tuberculin Skin Test induration. >10mm immunocompetent = positive."},
    "sputum_afb":     {"mean": 0.02,  "std": 0.02, "lo": 0.0,  "hi": 1.0, "unit": "score",
                       "medical": "Sputum AFB smear microscopy score. Positive in ~50-60% active pulmonary TB."},
    "genexpert_ct":   {"mean": 35.0,  "std": 3.0,  "lo": 10,   "hi": 45,  "unit": "Ct value",
                       "medical": "GeneXpert MTB/RIF cycle threshold. Lower Ct = higher bacterial load. <28 = positive."},
    # ── E. CLINICAL STATUS (2) ──
    "bmi":            {"mean": 22.0,  "std": 3.5,  "lo": 12.0, "hi": 45.0, "unit": "kg/m²",
                       "medical": "Body Mass Index. TB causes wasting -> BMI drop. Low BMI = poor prognosis."},
    "treatment_days": {"mean": 0.0,   "std": 0.0,  "lo": 0,    "hi": 365,  "unit": "days",
                       "medical": "Days since treatment started. 0 = new diagnosis. Used for treatment response."},
}

FEATURE_NAMES = list(FEATURE_SPECS.keys())
N_FEATURES = len(FEATURE_NAMES)  # 22

HOSPITALS = [
    "AIIMS Delhi", "Apollo Hospital Chennai", "Fortis Healthcare Mumbai",
    "CMC Vellore", "Tata Memorial Hospital Mumbai", "Medanta Gurugram",
    "Max Super Speciality Delhi", "PGIMER Chandigarh", "NIMHANS Bangalore",
    "Lilavati Hospital Mumbai", "Narayana Health Bangalore", "Manipal Hospital",
    "Sir Ganga Ram Hospital Delhi", "King Edward Memorial Mumbai",
    "Safdarjung Hospital Delhi", "LRS Institute of TB Delhi",
]

# TB Pathology Drift Direction (22 features)
DRIFT_DIRECTION = np.array([
    +1.0,   # heart_rate ↑
    -1.0,   # spo2 ↓
    +1.0,   # resp_rate ↑
    +1.0,   # temperature ↑
    +1.0,   # wbc_count ↑
    +1.0,   # esr ↑
    +1.0,   # crp ↑
    +1.0,   # lymphocyte_pct ↑
    -1.0,   # hemoglobin ↓
    -1.0,   # albumin ↓
    +1.0,   # platelet_count ↑ (reactive thrombocytosis)
    +1.0,   # blood_sugar ↑ (stress hyperglycemia)
    +1.0,   # xray_opacity ↑
    +1.0,   # xray_cavity ↑
    +1.0,   # xray_nodule ↑
    +1.0,   # xray_pleural ↑
    +1.0,   # ada_level ↑
    +1.0,   # mantoux_mm ↑
    +1.0,   # sputum_afb ↑
    -1.0,   # genexpert_ct ↓ (lower Ct = more bacteria)
    -1.0,   # bmi ↓ (wasting)
    +1.0,   # treatment_days ↑ (gets assigned treatment)
], dtype=np.float32)

TIER_MAGNITUDE = {
    1: (0.15, 0.40),   # Ultra-subtle
    2: (0.40, 0.75),   # Very subtle
    3: (0.75, 1.20),   # Subtle
    4: (1.20, 1.70),   # Moderate
    5: (1.70, 2.40),   # Clear
    6: (2.40, 3.80),   # Severe
}


# ═══════════════════════════════════════════════════════════════════════════════
# 22×22 BIOLOGICAL CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_medical_correlation_matrix():
    C = np.eye(N_FEATURES, dtype=np.float64)
    pairs = [
        # Vital-Vital
        (0,2,0.35), (0,3,0.42), (1,2,-0.32), (1,3,-0.18), (2,3,0.25),
        # Blood-Blood
        (4,5,0.40), (5,6,0.62), (4,6,0.38), (4,7,0.30), (8,9,0.35),
        (5,8,-0.28), (6,9,-0.25), (10,5,0.30), (10,6,0.25), (11,6,0.20),
        # Vital-Blood
        (3,5,0.32), (3,6,0.35), (0,8,-0.15), (1,8,0.20),
        # X-ray inter-correlations
        (12,13,0.52), (12,14,0.48), (12,15,0.30), (13,14,0.28),
        # Pleural-ADA
        (15,16,0.45),
        # Cross-domain: Vitals-Xray
        (1,12,-0.28), (2,12,0.22), (3,12,0.18),
        # Blood-Xray
        (5,12,0.38), (6,13,0.32), (8,12,-0.22), (9,12,-0.20),
        # TB tests
        (16,5,0.40), (16,6,0.35), (16,7,0.42), (16,12,0.30),
        (17,3,0.20), (17,7,0.35), (17,16,0.50), (17,12,0.18),
        # Sputum AFB correlations
        (18,13,0.55), (18,12,0.45), (18,6,0.30), (18,5,0.35),
        # GeneXpert (negative = worse)
        (19,18,-0.60), (19,13,-0.40), (19,12,-0.35),
        # BMI
        (20,8,0.25), (20,9,0.40), (20,12,-0.15),
    ]
    for i, j, val in pairs:
        if i < N_FEATURES and j < N_FEATURES:
            C[i,j] = C[j,i] = val

    eigvals = np.linalg.eigvalsh(C)
    if np.min(eigvals) < 0:
        C += (-np.min(eigvals) + 0.02) * np.eye(N_FEATURES)
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
    return C


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tb_dataset(n_total=3_000_000, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()

    print("=" * 78)
    print("  TB DATASET GENERATOR v2 — ULTIMATE JUDGE-PROOF EHR EDITION")
    print(f"  {N_FEATURES} Features | 7 Severity Tiers | Age/Gender/Comorbidity/Smoking/Season")
    print("=" * 78)

    corr = build_medical_correlation_matrix()
    means = np.array([FEATURE_SPECS[f]["mean"] for f in FEATURE_NAMES])
    stds  = np.array([FEATURE_SPECS[f]["std"]  for f in FEATURE_NAMES])
    # treatment_days has std=0 so we handle it separately
    stds_safe = stds.copy()
    stds_safe[stds_safe == 0] = 1.0
    cov = np.outer(stds_safe, stds_safe) * corr

    tier_dist = {0: 0.30, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.10, 5: 0.10, 6: 0.05}
    tier_counts = {t: int(n_total * f) for t, f in tier_dist.items()}
    tier_counts[0] += n_total - sum(tier_counts.values())

    print(f"\n  Total patients: {n_total:,}")
    for t, c in tier_counts.items():
        label = "Healthy" if t == 0 else f"Tier-{t}"
        print(f"    {label:12s}: {c:>10,}")

    all_dfs = []
    chunk = 500_000

    for tier, count in tier_counts.items():
        remaining = count
        while remaining > 0:
            batch = min(chunk, remaining)

            # 1. Base generation from 22-dim multivariate normal
            samples = np.random.multivariate_normal(means, cov, size=batch).astype(np.float32)
            # treatment_days: 0 for healthy, random for TB
            samples[:, 21] = 0  # Reset treatment_days

            # 2. Age & Gender
            ages = np.clip(np.concatenate([
                np.random.normal(30, 8, batch // 2),
                np.random.normal(58, 10, batch - batch // 2)
            ]), 18, 90)
            np.random.shuffle(ages)
            is_male = np.random.random(batch) < 0.60

            age_f = (ages - 45) / 30
            samples[:, 0] += age_f * 3.0      # HR ↑ with age
            samples[:, 1] -= age_f * 0.3       # SpO2 ↓ with age
            samples[:, 5] += age_f * 4.0       # ESR ↑ with age
            samples[:, 8] -= age_f * 0.5       # Hb ↓ with age
            samples[:, 9] -= age_f * 0.2       # Albumin ↓ with age
            samples[:, 17] += age_f * 1.5      # Mantoux ↑ with age (cumulative exposure)
            samples[:, 20] -= age_f * 0.8      # BMI ↓ slightly with age

            samples[is_male, 8] += 1.0         # Males higher Hb
            samples[~is_male, 8] -= 0.5
            samples[is_male, 5] -= 2.0         # Males lower ESR baseline
            samples[~is_male, 5] += 3.0

            # 3. Smoking modifier (25% smokers)
            is_smoker = np.random.random(batch) < 0.25
            samples[is_smoker, 1] -= np.random.uniform(0.5, 2.0, is_smoker.sum())  # SpO2 ↓
            samples[is_smoker, 2] += np.random.uniform(1.0, 3.0, is_smoker.sum())  # RR ↑
            samples[is_smoker, 12] += np.random.uniform(0.02, 0.08, is_smoker.sum())  # Opacity ↑
            samples[is_smoker, 20] -= np.random.uniform(0.5, 2.0, is_smoker.sum())  # BMI ↓

            # 4. TB Pathology Drift
            if tier > 0:
                lo, hi = TIER_MAGNITUDE[tier]
                mag = np.random.uniform(lo, hi, size=(batch, 1)).astype(np.float32)
                noise = 1.0 + np.random.normal(0, 0.20, (batch, N_FEATURES))
                leader = np.random.random((batch, N_FEATURES)) > 0.3
                samples += DRIFT_DIRECTION * stds_safe * mag * noise * leader

                # Sputum AFB: positive more likely in higher tiers
                if tier >= 3:
                    samples[:, 18] = np.random.uniform(0.3, 0.9, batch)
                elif tier >= 1:
                    samples[:, 18] = np.random.uniform(0.0, 0.3, batch)

                # GeneXpert Ct: lower in higher tiers (more bacteria)
                if tier >= 4:
                    samples[:, 19] = np.random.uniform(15, 25, batch)
                elif tier >= 2:
                    samples[:, 19] = np.random.uniform(22, 32, batch)

                # Treatment days: some patients already on treatment
                on_treatment = np.random.random(batch) < (0.1 * tier)
                samples[on_treatment, 21] = np.random.uniform(1, 180, on_treatment.sum())

            # 5. Comorbidities
            has_dm = np.random.random(batch) < 0.15
            has_hiv = np.random.random(batch) < 0.05

            samples[has_dm, 6] += np.random.uniform(0.5, 2.0, has_dm.sum())     # CRP ↑
            samples[has_dm, 5] += np.random.uniform(2.0, 5.0, has_dm.sum())     # ESR ↑
            samples[has_dm, 11] += np.random.uniform(30, 150, has_dm.sum())     # Blood sugar ↑↑
            samples[has_dm, 20] += np.random.uniform(2.0, 8.0, has_dm.sum())    # BMI ↑ (obesity)

            samples[has_hiv, 7] -= np.random.uniform(5.0, 15.0, has_hiv.sum())  # Lymphocyte ↓↓
            samples[has_hiv, 4] -= np.random.uniform(1.0, 3.0, has_hiv.sum())   # WBC ↓
            samples[has_hiv, 17] -= np.random.uniform(3.0, 6.0, has_hiv.sum())  # Mantoux ↓ (anergy)
            samples[has_hiv, 20] -= np.random.uniform(1.0, 4.0, has_hiv.sum())  # BMI ↓↓

            # 6. FALSE POSITIVES — Diseases that mimic TB
            if tier == 0:
                # 5% Pneumonia (high fever, WBC, CRP but NO cavities/AFB/GeneXpert)
                is_pneum = np.random.random(batch) < 0.05
                samples[is_pneum, 0] += np.random.uniform(10, 30, is_pneum.sum())
                samples[is_pneum, 3] += np.random.uniform(1.0, 2.5, is_pneum.sum())
                samples[is_pneum, 4] += np.random.uniform(5.0, 15.0, is_pneum.sum())
                samples[is_pneum, 6] += np.random.uniform(10.0, 50.0, is_pneum.sum())
                samples[is_pneum, 12] += np.random.uniform(0.1, 0.4, is_pneum.sum())  # Opacity ↑

                # 2% Sarcoidosis (high ADA, high Lymph% but NO sputum AFB)
                is_sarc = np.random.random(batch) < 0.02
                samples[is_sarc, 16] += np.random.uniform(20.0, 60.0, is_sarc.sum())  # ADA ↑↑
                samples[is_sarc, 7] += np.random.uniform(5.0, 15.0, is_sarc.sum())    # Lymph ↑
                samples[is_sarc, 14] += np.random.uniform(0.05, 0.2, is_sarc.sum())   # Nodules ↑

                # 1% Lung Cancer (mass on X-ray, weight loss)
                is_cancer = np.random.random(batch) < 0.01
                samples[is_cancer, 12] += np.random.uniform(0.2, 0.6, is_cancer.sum())  # Opacity ↑↑
                samples[is_cancer, 20] -= np.random.uniform(3.0, 8.0, is_cancer.sum())  # BMI ↓↓
                samples[is_cancer, 8] -= np.random.uniform(1.0, 3.0, is_cancer.sum())   # Hb ↓

            # Clamp to physiological limits
            for i, fname in enumerate(FEATURE_NAMES):
                spec = FEATURE_SPECS[fname]
                samples[:, i] = np.clip(samples[:, i], spec["lo"], spec["hi"])

            # Seasonal variation in admission dates
            # TB incidence peaks in Jan-Mar (winter) and Jul-Sep (monsoon) in India
            month_weights = [0.12, 0.11, 0.10, 0.07, 0.06, 0.06, 0.10, 0.11, 0.10, 0.06, 0.05, 0.06]
            months = np.random.choice(range(1, 13), size=batch, p=month_weights)
            years = np.random.choice([2021, 2022, 2023, 2024, 2025], size=batch)
            days = np.random.randint(1, 29, size=batch)
            dates = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)]

            # Build DataFrame
            df = pd.DataFrame(samples, columns=FEATURE_NAMES)
            df["Patient_ID"] = [f"PT-{str(uuid.uuid4())[:8].upper()}" for _ in range(batch)]
            df["Hospital"] = np.random.choice(HOSPITALS, batch)
            df["Age"] = ages.astype(int)
            df["Gender"] = np.where(is_male, "Male", "Female")
            df["Smoker"] = np.where(is_smoker, "Yes", "No")
            df["Diabetes"] = np.where(has_dm, "Yes", "No")
            df["HIV"] = np.where(has_hiv, "Positive", "Negative")
            df["Admission_Date"] = dates
            df["Diagnosis"] = "TB Positive" if tier > 0 else "Healthy"
            df["Severity_Tier"] = tier

            # 7. Sensor Noise (2%)
            noise_mask = np.random.random(df[FEATURE_NAMES].shape) < 0.02
            noise_vals = np.random.normal(0, stds_safe * 0.1, df[FEATURE_NAMES].shape)
            vals = df[FEATURE_NAMES].values
            vals = np.where(noise_mask, vals + noise_vals, vals)
            df[FEATURE_NAMES] = vals

            # 8. Missing Values (5% in lab tests — realistic)
            for col in ["crp", "esr", "ada_level", "mantoux_mm", "sputum_afb", "genexpert_ct"]:
                miss = np.random.random(batch) < 0.05
                df.loc[miss, col] = np.nan

            all_dfs.append(df)
            remaining -= batch

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── SAVING ──
    final_df.head(10000).to_csv(os.path.join(save_dir, "tb_ehr_10k_sample.csv"), index=False)
    final_df.to_parquet(os.path.join(save_dir, "tb_ehr_full.parquet"))

    # ML-ready tensors (NaN filled with column mean)
    ml_df = final_df[FEATURE_NAMES].copy()
    ml_df = ml_df.fillna(ml_df.mean())
    X = ml_df.values.astype(np.float32)
    y = np.where(final_df["Diagnosis"] == "TB Positive", 1.0, 0.0).astype(np.float32)
    tiers = final_df["Severity_Tier"].values.astype(np.int32)

    torch.save(torch.tensor(X), os.path.join(save_dir, "tb_features.pt"))
    torch.save(torch.tensor(y), os.path.join(save_dir, "tb_labels.pt"))
    torch.save(torch.tensor(tiers), os.path.join(save_dir, "tb_tiers.pt"))

    elapsed = time.time() - start
    print(f"\n  Generated in {elapsed:.1f}s | Shape: X=({n_total}, {N_FEATURES})")
    print(f"  EHR Metadata: Patient_ID, Hospital, Age, Gender, Smoker, DM, HIV, Date")
    print(f"  Realism: Pneumonia/Sarcoidosis/Cancer False Positives, 2% Sensor Noise, 5% Missing")
    print(f"  Files saved to '{save_dir}/':")
    print(f"   - tb_ehr_full.parquet ({n_total:,} rows)")
    print(f"   - tb_ehr_10k_sample.csv (10k readable)")
    print(f"   - tb_features.pt / tb_labels.pt / tb_tiers.pt (ML-ready)")
    print("=" * 78)
    return X, y, tiers


if __name__ == "__main__":
    generate_tb_dataset(n_total=3_000_000)
