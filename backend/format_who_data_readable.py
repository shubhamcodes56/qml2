import pandas as pd

def format_for_doctors():
    input_file = r'c:\Users\shubham dixit\OneDrive\Desktop\New folder (37)\QML_Report\backend\data\tb_patients_10k_readable.csv'
    output_file = r'c:\Users\shubham dixit\OneDrive\Desktop\New folder (37)\QML_Report\who_parameters_readable_10k.csv'
    
    # Read the data
    df = pd.read_csv(input_file)
    
    # Limit to 10k just in case it's larger
    df = df.head(10000)
    
    # Create a new DataFrame for the formatted strings
    formatted_df = pd.DataFrame()
    
    # Format each column with clinical units
    formatted_df['Heart Rate'] = df['Heart_Rate'].apply(lambda x: f"{x:.0f} bpm")
    formatted_df['SpO2 (Oxygen)'] = df['SpO2'].apply(lambda x: f"{x:.1f}%")
    formatted_df['Respiratory Rate'] = df['Respiratory_Rate'].apply(lambda x: f"{x:.0f} breaths/min")
    formatted_df['Temperature'] = df['Temperature'].apply(lambda x: f"{x:.1f} °C")
    formatted_df['WBC Count'] = df['WBC_Count'].apply(lambda x: f"{x:.1f} x10^9/L")
    formatted_df['ESR'] = df['ESR'].apply(lambda x: f"{x:.1f} mm/hr")
    formatted_df['CRP'] = df['CRP'].apply(lambda x: f"{x:.2f} mg/L")
    formatted_df['Lymphocytes'] = df['Lymphocyte_Pct'].apply(lambda x: f"{x:.1f}%")
    formatted_df['Hemoglobin'] = df['Hemoglobin'].apply(lambda x: f"{x:.1f} g/dL")
    formatted_df['Albumin'] = df['Albumin'].apply(lambda x: f"{x:.1f} g/dL")
    
    # Format X-ray findings from probabilities/fractions to percentages
    formatted_df['X-Ray Opacity'] = df['Xray_Opacity'].apply(lambda x: f"{x*100:.1f}% Area")
    formatted_df['X-Ray Cavitation'] = df['Xray_Cavity'].apply(lambda x: f"{x*100:.1f}% Area")
    formatted_df['X-Ray Nodules'] = df['Xray_Nodule'].apply(lambda x: f"{x*100:.1f}% Area")
    formatted_df['Pleural Effusion'] = df['Xray_Pleural'].apply(lambda x: f"{x*100:.1f}% Area")
    
    # Other markers
    formatted_df['Pleural ADA'] = df['ADA_Level'].apply(lambda x: f"{x:.1f} U/L")
    formatted_df['Mantoux Test (TST)'] = df['Mantoux_mm'].apply(lambda x: f"{x:.1f} mm")
    
    # Clinical Outcomes
    formatted_df['Diagnosis'] = df['Diagnosis']
    
    # Map Severity Tier to text descriptions
    tier_map = {
        0: "Tier 0 (No TB)",
        1: "Tier 1 (Latent/Early)",
        2: "Tier 2 (Mild Active)",
        3: "Tier 3 (Moderate)",
        4: "Tier 4 (Severe)",
        5: "Tier 5 (Critical/Fatal Risk)"
    }
    formatted_df['Severity Level'] = df['Severity_Tier'].map(tier_map)
    
    # Save the formatted data
    formatted_df.to_csv(output_file, index=False)
    print(f"Successfully saved formatted WHO parameters to {output_file}")

if __name__ == '__main__':
    format_for_doctors()
