import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os
import matplotlib as mpl
from matplotlib.patches import Patch

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'

def read_fea_csv(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    materials_line = lines[0].strip().split(',')
    materials = []
    
    for item in materials_line:
        if item and item not in ['', ' ']:
            materials.append(item)
    
    header_types = lines[1].strip().split(',')
    
    column_mapping = {}
    current_material_idx = -1
    
    for i, col_type in enumerate(header_types):
        if i == 0:  # First col: time
            continue
            
        if col_type and "Deformation" in col_type:
            current_material_idx += 1
            if current_material_idx < len(materials):
                current_material = materials[current_material_idx]
                column_mapping[i] = (current_material, "Deformation")
        elif col_type and ("Von misses" in col_type or "Von Mises" in col_type):
            if current_material_idx >= 0 and current_material_idx < len(materials):
                current_material = materials[current_material_idx]
                column_mapping[i] = (current_material, "VonMises")
        elif col_type and "Acceleration" in col_type:
            if current_material_idx >= 0 and current_material_idx < len(materials):
                current_material = materials[current_material_idx]
                column_mapping[i] = (current_material, "Acceleration")
    
    df = pd.read_csv(filename, skiprows=2, header=None)
    

    rename_dict = {0: 'Time'}  # First col: time
    
    for col_idx, (material, measurement) in column_mapping.items():
        if col_idx < len(df.columns):
            new_name = f"{material}_{measurement}"
            rename_dict[col_idx] = new_name
    
    df = df.rename(columns=rename_dict)
    

    valid_columns = ['Time'] + [value for value in rename_dict.values() if value != 'Time']
    df = df[[col for col in valid_columns if col in df.columns]]
    

    for col in df.columns:
        if col != 'Time':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_hic(time, acceleration, window_size=3):

    window_size_sec = window_size / 1000  #ms to sec
    max_hic = 0
    max_interval = (0, 0)
    

    acceleration_ms2 = acceleration
    
    n = len(time)
    
    for i in range(n):
        t1 = time[i]
        
  
        j = i
        while j < n and (time[j] - t1) <= window_size_sec:
            j += 1
            
        if j >= n:
            j = n - 1
            
        if j <= i:  # Skip if the interval is too small
            continue
            
        t2 = time[j]
        
        if t2 - t1 < 0.001:  # Skip if the interval is too small
            continue
            
        interval_time = time[i:j+1]
        interval_accel = acceleration_ms2[i:j+1]
        
        integral = integrate.simpson(interval_accel, x=interval_time)
        avg_accel = integral / (t2 - t1)
        
        hic = (avg_accel ** 2.5) * (t2 - t1)
        
        if hic > max_hic:
            max_hic = hic
            max_interval = (t1, t2)
    
    return max_hic, max_interval

def analyze_fea_results(filenames):

    results = {}
    
    for filename in filenames:
        try:
            material_thickness = float(filename.split('_')[-1].replace('.csv', ''))
        except:
            material_thickness = filename  
        
        print(f"Processing file: {filename}")
        df = read_fea_csv(filename)
        
        print(f"Columns in dataframe: {df.columns.tolist()}")
        
        material_results = {}
        
        materials = set(col.split('_')[0] for col in df.columns if '_' in col)
        
        for material in materials:
   
            accel_col = f"{material}_Acceleration"
            vonmises_col = f"{material}_VonMises"
            deform_col = f"{material}_Deformation"
            
            material_data = {
                'HIC': 0,
                'Interval': (0, 0),
                'Max Acceleration': 0,
                'Max VonMises': 0,
                'Max Deformation': 0,
                'Time Series': {}
            }
            
            if accel_col in df.columns:
                time = df['Time'].values
                accel = df[accel_col].values
                
                accel = np.nan_to_num(accel)
                
                # Calculate HIC
                hic_value, (t1, t2) = calculate_hic(time, accel)
                
                material_data['HIC'] = hic_value
                material_data['Interval'] = (t1, t2)
                material_data['Max Acceleration'] = np.max(accel) * 9.81  # Convert to m/s^2
                material_data['Time Series']['Time'] = time
                material_data['Time Series']['Acceleration'] = accel
                
                # Extract maximum Von Mises and Deformation values during the HIC interval
                if vonmises_col in df.columns:
                    interval_indices = (df['Time'] >= t1) & (df['Time'] <= t2)
                    interval_df = df[interval_indices]
                    
                    if not interval_df.empty and vonmises_col in interval_df.columns:
                        max_vonmises = interval_df[vonmises_col].max()
                        material_data['Max VonMises'] = max_vonmises
                    else:
                        material_data['Max VonMises'] = df[vonmises_col].max()
                
                # same for Deformation
                if deform_col in df.columns:
                    interval_indices = (df['Time'] >= t1) & (df['Time'] <= t2)
                    interval_df = df[interval_indices]

                    if not interval_df.empty and deform_col in interval_df.columns:
                        max_deform = interval_df[deform_col].max()
                        material_data['Max Deformation'] = max_deform
                    else:
                        material_data['Max Deformation'] = df[deform_col].max()
            else:
                if vonmises_col in df.columns:
                    material_data['Max VonMises'] = df[vonmises_col].max()
                
                if deform_col in df.columns:
                    material_data['Max Deformation'] = df[deform_col].max()
            
            material_results[material] = material_data
            
        results[material_thickness] = material_results
    
    return results

def plot_hic_results(results, output_dir="FEA_HIC_Results"):

    os.makedirs(output_dir, exist_ok=True)
    
    risk_categories = [
        (0, 100, 'Very Low Risk', 'lightgreen'),
        (100, 200, 'Low Risk', 'limegreen'),
        (200, 500, 'Moderate Risk', 'yellow'),
        (500, 700, 'Substantial Risk', 'orange'),
        (700, 1000, 'High Risk', 'coral'),
        (1000, 1500, 'Severe Risk', 'red'),
        (1500, float('inf'), 'Extreme Risk', 'darkred')
    ]
    
    # Create plots for HIC vs Material Thickness for each material
    all_materials = set()
    for thickness in results:
        for material in results[thickness]:
            all_materials.add(material)
    
    material_colors = {
        'PXD': '#1f77b4',  # Blue
        'EVA': '#ff7f0e',  # Orange
        'VNF': '#2ca02c',  # Green
        'FPF': '#d62728',  # Red
        'SBT': '#9467bd',  # Purple
        'MFM': '#8c564b',  # Brown
        'RPF': '#e377c2',  # Pink
        'MYF': '#8c564b',  # Brown- vary between MYM and MFM
        'Other': '#bcbd22'  # Yellow-green
    }
    

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.yscale('log')
    
    # shaded regions for risk
    risk_handles = []
    for start, end, label, color in risk_categories:
        ax.axhspan(start, min(end, 10000), alpha=0.2, color=color)
        risk_handles.append(Patch(facecolor=color, alpha=0.2, label=label))
    
    # Plot HIC vs Material Thickness
    material_handles = []
    for material in all_materials:
        thicknesses = []
        hic_values = []
        
        for thickness in results:
            if material in results[thickness]:
                thicknesses.append(thickness)
                hic_values.append(results[thickness][material]['HIC'])
        
        sort_idx = np.argsort(thicknesses)
        thicknesses = [thicknesses[i] for i in sort_idx]
        hic_values = [hic_values[i] for i in sort_idx]
        
        if len(hic_values) == 0 or np.all(np.array(hic_values) == 0):
            continue
        
        color = material_colors.get(material, material_colors['Other'])
        
        line, = ax.plot(thicknesses, hic_values, '-o', 
                        color=color, linewidth=3, markersize=10)
        material_handles.append(line)
    
    ax.set_xlabel('Material Thickness (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('HIC Value (logarithmic scale)', fontsize=14, fontweight='bold')
    ax.set_title('Head Injury Criterion vs Material Thickness', 
              fontsize=16, fontweight='bold')
    

    material_legend = ax.legend(material_handles, all_materials, 
                               loc='upper right', fontsize=10, 
                               title='Materials', title_fontsize=12)
    

    ax.add_artist(material_legend)
    
    risk_legend = ax.legend(handles=risk_handles, loc='lower left', 
                           fontsize=10, title='Risk Categories', 
                           title_fontsize=12)
    

    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hic_vs_thickness_log.png'), dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # shaded regions for risk
    risk_handles = []
    for start, end, label, color in risk_categories:
        ax.axhspan(start, min(end, 10000), alpha=0.2, color=color)
        risk_handles.append(Patch(facecolor=color, alpha=0.2, label=label))
    
    # Plot HIC vs Material Thickness
    material_handles = []
    for material in all_materials:
        thicknesses = []
        hic_values = []
        
        for thickness in results:
            if material in results[thickness]:
                thicknesses.append(thickness)
                hic_values.append(results[thickness][material]['HIC'])
        
        sort_idx = np.argsort(thicknesses)
        thicknesses = [thicknesses[i] for i in sort_idx]
        hic_values = [hic_values[i] for i in sort_idx]
        

        if len(hic_values) == 0 or np.all(np.array(hic_values) == 0):
            continue
        

        color = material_colors.get(material, material_colors['Other'])
        
        line, = ax.plot(thicknesses, hic_values, '-o', 
                        color=color, linewidth=3, markersize=10)
        material_handles.append(line)
    
    # Set labels and title
    ax.set_xlabel('Material Thickness (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('HIC Value', fontsize=14, fontweight='bold')
    ax.set_title('Head Injury Criterion vs Material Thickness', 
              fontsize=16, fontweight='bold')
    
    material_legend = ax.legend(material_handles, all_materials, 
                               loc='upper right', fontsize=10, 
                               title='Materials', title_fontsize=12)
    

    ax.add_artist(material_legend)
    
    risk_legend = ax.legend(handles=risk_handles, loc='lower left', 
                           fontsize=10, title='Risk Categories', 
                           title_fontsize=12)
    
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hic_vs_thickness.png'), dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot Von Mises vs Material Thickness
    material_handles = []
    for material in all_materials:
        thicknesses = []
        vonmises_values = []
        
        for thickness in results:
            if material in results[thickness]:
                thicknesses.append(thickness)
                vonmises_values.append(results[thickness][material]['Max VonMises'])
        
        sort_idx = np.argsort(thicknesses)
        thicknesses = [thicknesses[i] for i in sort_idx]
        vonmises_values = [vonmises_values[i] for i in sort_idx]
        
        if len(vonmises_values) == 0 or np.all(np.array(vonmises_values) == 0):
            continue
            
        color = material_colors.get(material, material_colors['Other'])
        
        line, = ax.plot(thicknesses, vonmises_values, '-o',
                       color=color, linewidth=3, markersize=10)
        material_handles.append(line)
    
    ax.set_xlabel('Material Thickness (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Max Von Mises Stress', fontsize=14, fontweight='bold')
    ax.set_title('Maximum Von Mises Stress vs Material Thickness', 
              fontsize=16, fontweight='bold')
    ax.legend(material_handles, all_materials, fontsize=10, loc='upper right', 
             title='Materials', title_fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vonmises_vs_thickness.png'), dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot Deformation vs Material Thickness
    material_handles = []
    for material in all_materials:
        thicknesses = []
        deform_values = []
        
        for thickness in results:
            if material in results[thickness]:
                thicknesses.append(thickness)
                deform_values.append(results[thickness][material]['Max Deformation'])
        
        sort_idx = np.argsort(thicknesses)
        thicknesses = [thicknesses[i] for i in sort_idx]
        deform_values = [deform_values[i] for i in sort_idx]
        
        if len(deform_values) == 0 or np.all(np.array(deform_values) == 0):
            continue
            
        color = material_colors.get(material, material_colors['Other'])
        
        line, = ax.plot(thicknesses, deform_values, '-o',
                       color=color, linewidth=3, markersize=10)
        material_handles.append(line)
    
    ax.set_xlabel('Material Thickness (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Max Deformation (mm)', fontsize=14, fontweight='bold')
    ax.set_title('Maximum Deformation vs Material Thickness', 
              fontsize=16, fontweight='bold')
    ax.legend(material_handles, all_materials, fontsize=10, loc='upper right',
             title='Materials', title_fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deformation_vs_thickness.png'), dpi=300)
    plt.close()

def export_results_to_csv(results, output_dir="FEA_HIC_Results"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_data = []
    
    for thickness in sorted(results.keys()):
        for material in sorted(results[thickness].keys()):
            data = results[thickness][material]
            
            row = {
                'Material': material,
                'Thickness': thickness,
                'HIC': data['HIC'],
                'Max_Acceleration': data['Max Acceleration'],
                'Max_VonMises': data['Max VonMises'],
                'Max_Deformation': data['Max Deformation'],
                'HIC_Interval_Start': data['Interval'][0],
                'HIC_Interval_End': data['Interval'][1]
            }
            
            csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_file = os.path.join(output_dir, 'fea_analysis_results.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"Results exported to {csv_file}")
    
    return csv_file

def main():

    filenames = [
        'FEA_initial_results_0.5.csv',
        'FEA_initial_results_1.csv',
        'FEA_initial_results_2.csv',
        'FEA_initial_results_3.csv'
    ]
    
    output_dir = "FEA_HIC_Results_Final"
    os.makedirs(output_dir, exist_ok=True)
    
    results = analyze_fea_results(filenames)
    plot_hic_results(results, output_dir)
    csv_file = export_results_to_csv(results, output_dir)
    summary_file = os.path.join(output_dir, 'fea_summary.txt')
    with open(summary_file, 'w') as f:
        import sys
        original_stdout = sys.stdout
        sys.stdout = f

        print("HIC Values Summary Table")
        print(f"{'Material':<10}", end="")
        for thickness in sorted(results.keys()):
            print(f"{thickness} mm".center(15), end="")
        print()
        
        print("-" * (10 + 15 * len(results)))
        
        for material in sorted(set(mat for thickness in results for mat in results[thickness])):
            print(f"{material:<10}", end="")
            for thickness in sorted(results.keys()):
                if material in results[thickness]:
                    hic = results[thickness][material]['HIC']
                    print(f"{hic:^15.6f}", end="")
                else:
                    print(f"{'N/A':^15}", end="")
            print()
        
        print("\n\nMax Von Mises Stress Summary Table")
        print(f"{'Material':<10}", end="")
        for thickness in sorted(results.keys()):
            print(f"{thickness} mm".center(15), end="")
        print()
        
        print("-" * (10 + 15 * len(results)))
        
        for material in sorted(set(mat for thickness in results for mat in results[thickness])):
            print(f"{material:<10}", end="")
            for thickness in sorted(results.keys()):
                if material in results[thickness]:
                    vonmises = results[thickness][material]['Max VonMises']
                    print(f"{vonmises:^15.6f}", end="")
                else:
                    print(f"{'N/A':^15}", end="")
            print()
        
        print("\n\n------ Max Deformation Summary Table ------")
        print(f"{'Material':<10}", end="")
        for thickness in sorted(results.keys()):
            print(f"{thickness} mm".center(15), end="")
        print()
        
        print("-" * (10 + 15 * len(results)))
        
        for material in sorted(set(mat for thickness in results for mat in results[thickness])):
            print(f"{material:<10}", end="")
            for thickness in sorted(results.keys()):
                if material in results[thickness]:
                    deform = results[thickness][material]['Max Deformation']
                    print(f"{deform:^15.6f}", end="")
                else:
                    print(f"{'N/A':^15}", end="")
            print()

        sys.stdout = original_stdout
    
    print(f"Analysis complete. Results saved in {output_dir}")
    print(f"Summary tables saved to {summary_file}")
    print(f"CSV file saved to {csv_file}")

if __name__ == "__main__":
    main()