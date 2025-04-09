import pandas as pd
import numpy as np
import re
import math

def parse_range(value):
    
    if pd.isna(value) or value == '':
        return None, None
    
    
    range_match = re.search(r'\[([\d.]+):([\d.]+)\]', str(value))
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    
    
    array_match = re.search(r'\[([\d\s.]+)\]', str(value))
    if array_match:
        values = [float(val) for val in array_match.group(1).split() if val.strip()]
        if values:
            return min(values), max(values)
    
    
    try:
        val = float(value)
        return val, val
    except (ValueError, TypeError):
        return None, None

def calculate_acoustic_impedance(density, youngs_modulus):
    #PSI to Pa
    youngs_modulus_pa = youngs_modulus * 6894.76  # 1 psi = 6894.76 Pa
    
    # wave speed: c = sqrt(E/ρ)
    wave_speed = math.sqrt(youngs_modulus_pa / density)
    
    # Z = ρ * c
    impedance = density * wave_speed
    
    return impedance

def calculate_coefficients(df):
    """Calculate reflection and transmission coefficients for all materials"""
    # Print available columns to help with debugging
    print("Available columns in the CSV:")
    print(df.columns.tolist())
    
    # Find skull cortical bones as the reference material
    # Try multiple ways to find the skull cortical bones row
    scb_row = None
    
    # Try using 'abbrv' column if it exists
    if 'abbrv' in df.columns:
        scb_rows = df[df['abbrv'] == 'SCB']
        if not scb_rows.empty:
            scb_row = scb_rows.iloc[0]
    
    # If not found, try looking in the 'material' column
    if scb_row is None and 'material' in df.columns:
        scb_rows = df[df['material'].str.contains('skull cortical bones', case=False, na=False)]
        if not scb_rows.empty:
            scb_row = scb_rows.iloc[0]
    
    # If still not found, try other column names that might contain this info
    if scb_row is None:
        for col in df.columns:
            if any(df[col].astype(str).str.contains('skull cortical', case=False, na=False)):
                potential_rows = df[df[col].astype(str).str.contains('skull cortical', case=False, na=False)]
                if not potential_rows.empty:
                    scb_row = potential_rows.iloc[0]
                    break
    
    if scb_row is None:
        raise ValueError("Could not find skull cortical bones data in the CSV")
    
    print(f"Found skull cortical bones row: {scb_row.to_dict()}")
    
    # Find the columns for density and Young's modulus
    density_col = next((col for col in df.columns if 'density' in col.lower()), None)
    youngs_col = next((col for col in df.columns if 'youngs' in col.lower() or 'modulus' in col.lower()), None)
    
    if not density_col:
        raise ValueError("Could not find density column in the CSV")
    if not youngs_col:
        raise ValueError("Could not find Young's modulus column in the CSV")
    
    print(f"Using columns: {density_col} and {youngs_col}")
    
    # Calculate Z2 (reference impedance) for skull cortical bones
    density_z2, _ = parse_range(scb_row[density_col])
    youngs_z2, _ = parse_range(scb_row[youngs_col])
    
    if density_z2 is None or youngs_z2 is None:
        raise ValueError("Could not parse density or Young's modulus for skull cortical bones")
    
    Z2 = calculate_acoustic_impedance(density_z2, youngs_z2)
    print(f"Reference impedance (Z2) for skull cortical bones: {Z2:.2f} kg/m²/s")
    
    # Find output columns or create them if they don't exist
    if 'opt_ref_coeff' not in df.columns:
        df['opt_ref_coeff'] = ""
    if 'opt_tans_coeff' not in df.columns:
        df['opt_tans_coeff'] = ""
    
    # Get the material column name
    material_col = next((col for col in df.columns if 'material' in col.lower()), df.columns[0])
    
    # Process each row in the dataframe
    for index, row in df.iterrows():
        # Parse density and Young's modulus ranges using the detected column names
        density_min, density_max = parse_range(row[density_col])
        youngs_min, youngs_max = parse_range(row[youngs_col])
        
        # Get material name for logging
        material_name = row[material_col] if material_col in row else f"Row {index}"
        
        # Skip rows with missing data
        if density_min is None or youngs_min is None:
            print(f"Skipping {material_name} due to missing or invalid data")
            continue
        
        # Calculate Z1 (min and max)
        Z1_min = calculate_acoustic_impedance(density_min, youngs_min)
        Z1_max = calculate_acoustic_impedance(density_max, youngs_max)
        
        # Calculate reflection coefficient (R) bounds
        # R = (Z2 - Z1) / (Z1 + Z2)
        R_values = [
            (Z2 - Z1_min) / (Z1_min + Z2),
            (Z2 - Z1_max) / (Z1_max + Z2)
        ]
        R_min, R_max = min(R_values), max(R_values)
        
        # Calculate energy transmission coefficient (T) bounds
        # CORRECTED FORMULA: T = 4 * Z1 * Z2 / (Z1 + Z2)²
        T_values = [
            4 * Z1_min * Z2 / (Z1_min + Z2)**2,
            4 * Z1_max * Z2 / (Z1_max + Z2)**2
        ]
        T_min, T_max = min(T_values), max(T_values)
        
        # Format as [min:max] string and update the dataframe
        df.at[index, 'opt_ref_coeff'] = f"[{R_min:.4f}:{R_max:.4f}]"
        df.at[index, 'opt_tans_coeff'] = f"[{T_min:.4f}:{T_max:.4f}]"
        
        # Print detailed information for verification
        print(f"Processed {material_name}:")
        print(f"  Z1 range: {Z1_min:.2f} - {Z1_max:.2f} kg/m²/s")
        print(f"  R = {df.at[index, 'opt_ref_coeff']} (reflection coefficient)")
        print(f"  T = {df.at[index, 'opt_tans_coeff']} (energy transmission coefficient)")
    
    return df

def main():
    import sys
    import os
    
   
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'material_properties.csv'
    
   
    print(f"Reading CSV file: {input_file}")
    try:
    
        try:
            df = pd.read_csv(input_file)
        except pd.errors.ParserError:
            
            print("Standard parsing failed, trying more flexible parsing...")
            df = pd.read_csv(input_file, sep=None, engine='python', on_bad_lines='warn')
        
        
        print("First few rows of the CSV:")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return
    
    #coeffs
    print("Calculating reflection and transmission coefficients...")
    try:
        df = calculate_coefficients(df)
    except Exception as e:
        print(f"Error calculating coefficients: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_updated.csv"
    
    
    df.to_csv(output_file, index=False)
    print(f"Updated data written to: {output_file}")

if __name__ == "__main__":
    main()