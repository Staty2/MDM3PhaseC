import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import random

def parse_fea_summary(file_path):
    current_table = None
    tables = {
        'HIC': [],
        'Stress': [],
        'Deformation': []
    }
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                line = line.strip()
                
                if not line:
                    continue
                
                if "HIC Values Summary Table" in line:
                    current_table = 'HIC'
                    continue
                elif "Max Von Mises Stress Summary Table" in line:
                    current_table = 'Stress'
                    continue
                elif "Max Deformation Summary Table" in line:
                    current_table = 'Deformation'
                    continue

                if line.startswith("Material") or line.startswith("----"):
                    continue

                if current_table and line:
                    tables[current_table].append(line)

        dfs = {}
        for table_name, table_lines in tables.items():
            data_rows = []
            for line in table_lines:
                parts = line.split()
                if len(parts) >= 5:
                    row = [parts[0]]  # Mat name
                    row.extend([float(val) for val in parts[1:5]])  
                    data_rows.append(row)
            
            dfs[table_name] = pd.DataFrame(data_rows, 
                                         columns=['Material', '0.5 cm', '1.0 cm', '2.0 cm', '3.0 cm'])
        
        return dfs
    
    except Exception as e:
        print(f"Error reading or parsing file: {str(e)}")
        return None
    
def create_decision_matrix_for_thickness(hic_df, stress_df, deform_df, thickness, weights):
    materials = hic_df['Material'].tolist()
    criteria = ['HIC Value', 'Max Stress', 'Deformation']
    
    raw_data = {
        'Material': materials,
        'HIC Value': hic_df[thickness].tolist(),
        'Max Stress': stress_df[thickness].tolist(),
        'Deformation': deform_df[thickness].tolist()
    }
    
    df = pd.DataFrame(raw_data)
    

    for criterion in criteria:
        min_val = df[criterion].min()
        max_val = df[criterion].max()
        if max_val > min_val:
            df[f'{criterion} (Normalized)'] = 1 - ((df[criterion] - min_val) / (max_val - min_val))
        else:
            df[f'{criterion} (Normalized)'] = 1
    

    weight_map = {'HIC Value': weights['HIC'], 'Max Stress': weights['Stress'], 'Deformation': weights['Deformation']}
    

    for criterion in criteria:
        df[f'{criterion} (Weighted)'] = df[f'{criterion} (Normalized)'] * weight_map[criterion]
    
    # Calc tot score
    df['Total Score'] = sum(df[f'{criterion} (Weighted)'] for criterion in criteria)
    
    # Add rank
    df['Rank'] = df['Total Score'].rank(ascending=False).astype(int)
    
    return df.sort_values('Rank')

# Main function to analyze all thicknesses
def analyze_all_thicknesses(file_path, weights={'HIC': 0.6, 'Stress': 0.2, 'Deformation': 0.2}):
    print(f"Reading data from: {file_path}")
    dfs = parse_fea_summary(file_path)
    
    if not dfs:
        print("Failed to read data. Please check the file path and format.")
        return
    
    hic_df = dfs['HIC']
    stress_df = dfs['Stress']
    deform_df = dfs['Deformation']
    
    print(f"Successfully read data for {len(hic_df)} materials.")
    
    output_dir = "material_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    thicknesses = ['0.5 cm', '1.0 cm', '2.0 cm', '3.0 cm']
    all_results = {}
    top_materials = {}
    all_sensitivity = []
    
    plt.figure(figsize=(14, 10))
    
    # Loop through each thickness and analyze
    for i, thickness in enumerate(thicknesses):
        print(f"\n\nAnalyzing {thickness} thickness...")
        print("=" * 80)
        
        #decision matrix for this thickness
        decision_df = create_decision_matrix_for_thickness(
            hic_df, stress_df, deform_df, thickness, weights
        )
        
        all_results[thickness] = decision_df

        top_material = decision_df.iloc[0]['Material']
        top_score = decision_df.iloc[0]['Total Score']
        top_materials[thickness] = (top_material, top_score)
        
        print(f"Top material for {thickness}: {top_material} (score: {top_score:.4f})")
        
        # Create visualizations:
        
        # 1. Bar chart
        plt.figure(figsize=(12, 8))
        bar_width = 0.2
        index = np.arange(len(decision_df))
        criteria = ['HIC Value', 'Max Stress', 'Deformation']
        
        for j, criterion in enumerate(criteria):
            plt.bar(index + j*bar_width, 
                   decision_df[f'{criterion} (Normalized)'], 
                   bar_width, 
                   alpha=0.8, 
                   label=criterion)
        
        plt.xlabel('Material')
        plt.ylabel('Normalized Score (Higher is Better)')
        plt.title(f'Decision Matrix Analysis for {thickness} Thickness')
        plt.xticks(index + bar_width, decision_df['Material'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/decision_matrix_{thickness.replace(".", "p").replace(" ", "")}.png', dpi=300)
        plt.close()
        
        # 2. Heatmap
        plt.figure(figsize=(10, 8))
        heatmap_data = decision_df[['Material'] + [f'{criterion} (Normalized)' for criterion in criteria] + ['Total Score']].copy()
        heatmap_data = heatmap_data.set_index('Material')
        heatmap_data.columns = [col.replace(' (Normalized)', '') for col in heatmap_data.columns]
        
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".2f")
        plt.title(f'Material Decision Matrix Heatmap for {thickness} Thickness')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/decision_heatmap_{thickness.replace(".", "p").replace(" ", "")}.png', dpi=300)
        plt.close()
        
        # 3. Radar chart for top 3 materials
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)
        
        categories = criteria
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        top3_materials = decision_df['Material'].head(3).tolist()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for k, material in enumerate(top3_materials):
            values = []
            for criterion in criteria:
                values.append(decision_df.loc[decision_df['Material'] == material, f'{criterion} (Normalized)'].values[0])
            values += values[:1] 
            
            ax.plot(angles, values, color=colors[k], linewidth=2, label=material)
            ax.fill(angles, values, color=colors[k], alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)
        
        plt.title(f"Top 3 Materials Comparison for {thickness} Thickness", size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(f'{output_dir}/radar_chart_{thickness.replace(".", "p").replace(" ", "")}.png', dpi=300)
        plt.close()
        
        weighting_schemes = {
            "Balanced": {'HIC': 0.33, 'Stress': 0.33, 'Deformation': 0.34},
            "HIC Priority": {'HIC': 0.6, 'Stress': 0.2, 'Deformation': 0.2},
            "Stress Priority": {'HIC': 0.2, 'Stress': 0.6, 'Deformation': 0.2},
            "Deformation Priority": {'HIC': 0.2, 'Stress': 0.2, 'Deformation': 0.6}
        }
        
        for scheme_name, scheme_weights in weighting_schemes.items():
            scheme_df = create_decision_matrix_for_thickness(
                hic_df, stress_df, deform_df, thickness, scheme_weights
            )
            
            top_scheme_material = scheme_df.iloc[0]['Material']
            top_scheme_score = scheme_df.iloc[0]['Total Score']
            
            all_sensitivity.append({
                'Thickness': thickness,
                'Scheme': scheme_name,
                'Top Material': top_scheme_material,
                'Score': top_scheme_score,
                'HIC Weight': scheme_weights['HIC'],
                'Stress Weight': scheme_weights['Stress'],
                'Deformation Weight': scheme_weights['Deformation']
            })
            
            print(f"  {scheme_name}: {top_scheme_material} (score: {top_scheme_score:.4f})")
        

        plt.subplot(2, 2, i+1)
        materials = decision_df['Material'].tolist()
        scores = decision_df['Total Score'].tolist()
        
        sorted_indices = np.argsort(scores)[::-1]
        sorted_materials = [materials[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        bars = plt.bar(sorted_materials, sorted_scores, color='skyblue')
        
        top_idx = sorted_materials.index(top_material)
        bars[top_idx].set_color('navy')
        
        plt.title(f'{thickness} Thickness')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Performance Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # summary comparison
    plt.suptitle('Material Performance Comparison Across All Thicknesses', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'{output_dir}/all_thicknesses_comparison.png', dpi=300)
    plt.close()
    
    # Create comparison chart for top materials by thickness
    plt.figure(figsize=(10, 6))
    thickness_labels = list(top_materials.keys())
    top_material_names = [info[0] for info in top_materials.values()]
    top_material_scores = [info[1] for info in top_materials.values()]
    
    bars = plt.bar(thickness_labels, top_material_scores, color='lightblue')
    
    for i, (bar, material) in enumerate(zip(bars, top_material_names)):
        plt.text(i, bar.get_height() + 0.02, material, 
                ha='center', va='bottom', fontweight='bold', color='navy')
    
    plt.title('Top Performing Material by Thickness')
    plt.xlabel('Thickness')
    plt.ylabel('Performance Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_materials_by_thickness.png', dpi=300)
    plt.close()
    
    excel_filename = f'{output_dir}/all_thickness_analysis.xlsx'
    attempt = 1
    max_attempts = 3
    
    while attempt <= max_attempts:
        try:
            if attempt > 1:
                timestamp = int(time.time())
                random_suffix = random.randint(1000, 9999)
                excel_filename = f'{output_dir}/all_thickness_analysis_{timestamp}_{random_suffix}.xlsx'
                print(f"Attempt {attempt}: Trying alternate filename: {excel_filename}")
            
            with pd.ExcelWriter(excel_filename) as writer:
                summary_data = []
                for thickness, (material, score) in top_materials.items():
                    summary_data.append({
                        'Thickness': thickness,
                        'Best Material': material,
                        'Score': score
                    })
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual thickness sheets
                for thickness, df in all_results.items():
                    sheet_name = thickness.replace('.', 'p').replace(' ', '')
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Sensitivity analysis
                pd.DataFrame(all_sensitivity).to_excel(writer, sheet_name='Sensitivity', index=False)
            
            print(f"Successfully saved Excel file to: {excel_filename}")
            break  # Success!! - exit the loop
            
        except PermissionError as e:
            print(f"Permission error (attempt {attempt}/{max_attempts}): {str(e)}")
            if attempt == max_attempts:
                print(f"Could not save Excel file after {max_attempts} attempts. CSV files will be created instead.")
                pd.DataFrame(summary_data).to_csv(f'{output_dir}/summary.csv', index=False)
                for thickness, df in all_results.items():
                    csv_name = thickness.replace('.', 'p').replace(' ', '')
                    df.to_csv(f'{output_dir}/{csv_name}.csv', index=False)
                pd.DataFrame(all_sensitivity).to_csv(f'{output_dir}/sensitivity.csv', index=False)
            attempt += 1
            time.sleep(2)  
        except Exception as e:
            print(f"Error saving Excel file: {str(e)}")
            print("Saving as CSV files instead.")
            pd.DataFrame(summary_data).to_csv(f'{output_dir}/summary.csv', index=False)
            for thickness, df in all_results.items():
                csv_name = thickness.replace('.', 'p').replace(' ', '')
                df.to_csv(f'{output_dir}/{csv_name}.csv', index=False)
            pd.DataFrame(all_sensitivity).to_csv(f'{output_dir}/sensitivity.csv', index=False)
            break
    
    # material performance table across thicknesses
    material_performance = []
    materials = hic_df['Material'].unique()
    
    for material in materials:
        material_data = {'Material': material}
        
        scores_by_thickness = {}
        for thickness in thicknesses:
            score = all_results[thickness].loc[all_results[thickness]['Material'] == material, 'Total Score'].values[0]
            material_data[thickness] = score
            scores_by_thickness[thickness] = score
        
        best_thickness = max(scores_by_thickness, key=scores_by_thickness.get)
        best_score = scores_by_thickness[best_thickness]
        
        material_data['Best Thickness'] = best_thickness
        material_data['Best Score'] = best_score
        
        material_performance.append(material_data)
    
    # Create DataFrame
    material_perf_df = pd.DataFrame(material_performance)
    material_perf_df = material_perf_df.sort_values('Best Score', ascending=False)
    
    try:
        material_perf_df.to_excel(f'{output_dir}/material_performance_comparison.xlsx', index=False)
    except Exception as e:
        print(f"Error saving material performance Excel file: {str(e)}")
        print("Saving as CSV instead.")
        material_perf_df.to_csv(f'{output_dir}/material_performance_comparison.csv', index=False)
    
    plt.figure(figsize=(12, 8))
    
    # Create a bar plot with materials on x-axis and best score on y-axis
    materials = material_perf_df['Material'].tolist()
    best_scores = material_perf_df['Best Score'].tolist()
    best_thicknesses = material_perf_df['Best Thickness'].tolist()
    
    thickness_colors = {
        '0.5 cm': '#FF9999',  # Light red
        '1.0 cm': '#99FF99',  # Light green
        '2.0 cm': '#9999FF',  # Light blue
        '3.0 cm': '#FFFF99'   # Light yellow
    }
    

    bar_colors = [thickness_colors[thickness] for thickness in best_thicknesses]
    bars = plt.bar(materials, best_scores, color=bar_colors)

    for i, (bar, thickness) in enumerate(zip(bars, best_thicknesses)):
        plt.text(i, bar.get_height() + 0.02, thickness, 
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Best Thickness by Material (with Optimal Performance Score)')
    plt.xlabel('Material')
    plt.ylabel('Optimal Performance Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=thickness) 
                      for thickness, color in thickness_colors.items()]
    plt.legend(handles=legend_elements, title="Thickness")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_thickness_by_material.png', dpi=300)
    plt.close()
    
    print("\n\nOVERALL RECOMMENDATIONS")
    print("=" * 80)
    print("Best Material by Thickness:")
    for thickness, (material, score) in top_materials.items():
        print(f"For {thickness} thickness: {material} (score: {score:.4f})")
    
    print("\nBest Thickness by Material:")
    for _, row in material_perf_df.iterrows():
        print(f"{row['Material']}: {row['Best Thickness']} (score: {row['Best Score']:.4f})")
    
    best_material_idx = material_perf_df['Best Score'].idxmax()
    best_material = material_perf_df.iloc[best_material_idx]['Material']
    best_thickness = material_perf_df.iloc[best_material_idx]['Best Thickness']
    best_score = material_perf_df.iloc[best_material_idx]['Best Score']
    
    print("\nOPTIMAL MATERIAL-THICKNESS COMBINATION:")
    print(f"Material: {best_material}")
    print(f"Thickness: {best_thickness}")
    print(f"Performance Score: {best_score:.4f}")
    
    print("\nDetailed results and visualizations have been saved to the 'material_analysis_results' directory.")
    
    return top_materials, all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Material Decision Matrix Analysis for All Thicknesses')
    parser.add_argument('--file', type=str, default="FEA_HIC_Results_Final/FEA_summary.txt",
                        help='Path to FEA_summary.txt file')
    parser.add_argument('--hic-weight', type=float, default=0.6,
                        help='Weight for HIC values (0-1)')
    parser.add_argument('--stress-weight', type=float, default=0.125,
                        help='Weight for Stress values (0-1)')
    parser.add_argument('--deform-weight', type=float, default=0.125,
                        help='Weight for Deformation values (0-1)')
    
    args = parser.parse_args()
    
    #Set weights: ##can change here
    weights = {'HIC': 0.6, 'Stress': 0.15, 'Deformation': 0.05}
    
    requested_weights = {'HIC': 0.6, 'Stress': 0.15, 'Deformation': 0.05}
    total = sum(requested_weights.values())
    
    # Normalize if don't sum to 1
    if total != 1.0:
        print(f"Normalizing weights that sum to {total}")
        for key in requested_weights:
            requested_weights[key] /= total
    
    analyze_all_thicknesses(args.file, requested_weights)