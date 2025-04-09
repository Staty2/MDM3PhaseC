import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


def parse_range(value):
    if pd.isna(value) or value == '':
        return None, None
    
    # Check if it's a range in [min:max] format
    match = re.search(r'\[([0-9.]+):([0-9.]+)\]', str(value))
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # Check if it's a list of values [val1 val2 val3]
    match = re.search(r'\[([0-9. ]+)\]', str(value))
    if match:
        values = [float(v) for v in match.group(1).split() if v]
        if values:
            return min(values), max(values)
    
    # If it's a single value, return it as both min and max
    try:
        val = float(value)
        return val, val
    except (ValueError, TypeError):
        return None, None


df = pd.read_csv('material_properties_updated.csv', skipinitialspace=True)

print("CSV Column names:", df.columns.tolist())

trans_col_name = 'opt_tans_coeff.1'  # second one in csv
refl_col_name = 'opt_ref_coeff.1'    # second one in csv

print("\nSample values from coefficient columns:")
for i, row in df.head(3).iterrows():
    print(f"Material: {row['abbrv']}")
    print(f"  Transmission column: {row[trans_col_name]}")
    print(f"  Reflection column: {row[refl_col_name]}")


materials = []
material_types = []
material_fullnames = []  
trans_values = []
refl_values = []
trans_min_values = []
trans_max_values = []
refl_min_values = []
refl_max_values = []


materials_of_interest = ['EVA', 'VNF', 'PXD', 'MYF', 'RPF', 'FPF', 'SBT']


for idx, row in df.iterrows():
    material_name = row['material']
    abbrv = row['abbrv']
    mat_type = row['material_type']
    
    #trans coeff
    t_min, t_max = parse_range(row[trans_col_name])
    
    #reflec coeff
    r_min, r_max = parse_range(row[refl_col_name])
    
   
    if t_min is None or r_min is None:
        print(f"Skipping {abbrv} - Trans: {row[trans_col_name]}, Refl: {row[refl_col_name]}")
        continue
    
    
    t_median = (t_min + t_max) / 2
    r_median = (r_min + r_max) / 2
    
    
    materials.append(abbrv)
    material_fullnames.append(material_name)
    material_types.append(mat_type)
    trans_values.append(t_median)
    refl_values.append(r_median)
    trans_min_values.append(t_min)
    trans_max_values.append(t_max)
    refl_min_values.append(r_min)
    refl_max_values.append(r_max)


plot_df = pd.DataFrame({
    'Material': materials,
    'Full_Name': material_fullnames,
    'Material_Type': material_types,
    'Transmission': trans_values,
    'Reflection': refl_values,
    'TransMin': trans_min_values,
    'TransMax': trans_max_values,
    'ReflMin': refl_min_values,
    'ReflMax': refl_max_values,
    'IsOfInterest': [m in materials_of_interest for m in materials]
})

print(f"\nProcessed data for {len(plot_df)} materials")

# unique material types for color mapping
unique_material_types = plot_df['Material_Type'].unique()
color_palette = sns.color_palette("viridis", len(unique_material_types))
color_dict = dict(zip(unique_material_types, color_palette))


# figure showing acronyms and their full names
plt.figure(figsize=(12, 8))
acronym_table = plot_df[['Material', 'Full_Name', 'Material_Type']].sort_values(by=['Material_Type', 'Material'])

fig, ax = plt.subplots(figsize=(10, len(acronym_table) * 0.4))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=acronym_table.values,
    colLabels=['Acronym', 'Full Name', 'Material Type'],
    loc='center',
    cellLoc='left',
    colWidths=[0.2, 0.6, 0.2]
)


table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)  


ax.set_title('Material Acronym Reference', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('material_acronym_reference.png', dpi=300, bbox_inches='tight')

#box plots
if len(plot_df) > 0:
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16), dpi=100)
    
    
    plot_df_sorted = plot_df.sort_values(by=['Material_Type', 'Material'])
    
    
    x = np.arange(len(plot_df_sorted))
    width = 0.35
    
    
    colors = [color_dict[t] for t in plot_df_sorted['Material_Type']]
    
    
    bars1 = ax1.bar(x - width/2, plot_df_sorted['ReflMin'], width, label='Min', 
                   color=[c + (0.7,) for c in colors], edgecolor='black')
    bars2 = ax1.bar(x + width/2, plot_df_sorted['ReflMax'], width, label='Max', 
                   color=colors, edgecolor='black')
    
    ax1.set_ylabel('Reflection Coefficient', fontsize=12)
    ax1.set_title('Minimum and Maximum Reflection Coefficients by Material', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df_sorted['Material'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.05)
    
    
    ax1.legend(title='Value Type', loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Second box plot: Transmission Coefficients (min and max)
    bars3 = ax2.bar(x - width/2, plot_df_sorted['TransMin'], width, label='Min', 
                   color=[c + (0.7,) for c in colors], edgecolor='black')
    bars4 = ax2.bar(x + width/2, plot_df_sorted['TransMax'], width, label='Max', 
                   color=colors, edgecolor='black')
    
    ax2.set_ylabel('Transmission Coefficient', fontsize=12)
    ax2.set_title('Minimum and Maximum Transmission Coefficients by Material', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_df_sorted['Material'], rotation=45, ha='right')
    ax2.set_ylim(0, 1.05)
    ax2.legend(title='Value Type', loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    
    fig_legend = plt.figure(figsize=(10, 2))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    
    
    handles = [plt.Rectangle((0,0), 1, 1, color=color_dict[t]) for t in unique_material_types]
    labels = list(unique_material_types)
    
    # Add legend to separate fig
    ax_legend.legend(handles=handles, labels=labels, loc='center', 
                    title='Material Type Legend', ncol=len(unique_material_types))
    fig_legend.savefig('material_type_legend.png', dpi=300, bbox_inches='tight')
    
    
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig('material_coefficient_boxplots.png', dpi=300, bbox_inches='tight')
    
 
    plt.figure(figsize=(16, 10))
    
    # Num materials
    n_materials = len(plot_df)
    x = np.arange(n_materials)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
 
    for i, (_, row) in enumerate(plot_df.iterrows()):
        mat_type = row['Material_Type']
        color = color_dict[mat_type]
        
        # Reflec coeff
        ax.bar(i - width/2, row['Reflection'], width, color=color, edgecolor='black', alpha=0.8)
        
        # Trans coeffr
        ax.bar(i + width/2, row['Transmission'], width, color=color, edgecolor='black', alpha=0.4)
    
    # Add error bars to show min-max ranges
    ax.errorbar(x - width/2, plot_df['Reflection'], 
                yerr=[plot_df['Reflection'] - plot_df['ReflMin'], plot_df['ReflMax'] - plot_df['Reflection']], 
                fmt='none', color='black', capsize=5)
    
    ax.errorbar(x + width/2, plot_df['Transmission'], 
                yerr=[plot_df['Transmission'] - plot_df['TransMin'], plot_df['TransMax'] - plot_df['Transmission']], 
                fmt='none', color='black', capsize=5)
    
    # ref lines
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal Reflection (1.0)')
    ax.axhline(y=0.0, color='blue', linestyle='--', alpha=0.7, label='Ideal Transmission (0.0)')
    
    
    reflection_proxy = plt.Rectangle((0, 0), 1, 1, color='grey', alpha=0.8)
    transmission_proxy = plt.Rectangle((0, 0), 1, 1, color='grey', alpha=0.4)
    
    
    legend1 = ax.legend([reflection_proxy, transmission_proxy], 
                       ['Reflection', 'Transmission'], 
                       loc='upper right')
    ax.add_artist(legend1)
    
    
    ax.set_xlabel('Material', fontsize=14)
    ax.set_ylabel('Coefficient Value', fontsize=14)
    ax.set_title('Transmission and Reflection Coefficients by Material', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Material'], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('coefficient_comparison.png', dpi=300, bbox_inches='tight')
    

    interest_df = plot_df[plot_df['IsOfInterest']].copy()
    
    if len(interest_df) > 0:
        
        interest_df.loc[:, 'Performance'] = interest_df['Reflection'] + (1 - interest_df['Transmission'])
        
        
        interest_df = interest_df.sort_values('Performance', ascending=False)
        
        
        plt.figure(figsize=(14, 10))
        
        x2 = np.arange(len(interest_df))
        
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        
        
        for i, row in enumerate(interest_df.itertuples()):
            mat_type = row.Material_Type
            color = color_dict[mat_type]
            
            # Reflec coeff
            ax2.bar(i - width/2, row.Reflection, width, color=color, edgecolor='black', alpha=0.8)
            
            # Trans coeff
            ax2.bar(i + width/2, row.Transmission, width, color=color, edgecolor='black', alpha=0.4)
        
        # Add error bars to show min-max ranges
        ax2.errorbar(x2 - width/2, interest_df['Reflection'], 
                    yerr=[interest_df['Reflection'] - interest_df['ReflMin'], interest_df['ReflMax'] - interest_df['Reflection']], 
                    fmt='none', color='black', capsize=5)
        
        ax2.errorbar(x2 + width/2, interest_df['Transmission'], 
                    yerr=[interest_df['Transmission'] - interest_df['TransMin'], interest_df['TransMax'] - interest_df['Transmission']], 
                    fmt='none', color='black', capsize=5)
        
        
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal Reflection (1.0)')
        ax2.axhline(y=0.0, color='blue', linestyle='--', alpha=0.7, label='Ideal Transmission (0.0)')
        
        
        reflection_proxy = plt.Rectangle((0, 0), 1, 1, color='grey', alpha=0.8)
        transmission_proxy = plt.Rectangle((0, 0), 1, 1, color='grey', alpha=0.4)
        
        
        legend1 = ax2.legend([reflection_proxy, transmission_proxy], 
                           ['Reflection', 'Transmission'], 
                           loc='upper right')
        ax2.add_artist(legend1)
        
        
        interest_types = interest_df['Material_Type'].unique()
        interest_colors = [color_dict[t] for t in interest_types]
        material_type_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in interest_colors]
        material_type_labels = list(interest_types)
        
        
        legend2 = ax2.legend(material_type_handles, material_type_labels, 
                           title='Material Type', loc='upper center', 
                           bbox_to_anchor=(0.5, -0.15), ncol=len(interest_types))
        
        
        for i, row in enumerate(interest_df.itertuples()):
            ax2.text(i, max(row.Reflection, row.Transmission) + 0.15, 
                    f'Score: {row.Performance:.2f}', ha='center', va='bottom', fontweight='bold')
        
        
        ax2.set_xlabel('Material', fontsize=14)
        ax2.set_ylabel('Coefficient Value', fontsize=14)
        ax2.set_title('Materials of Interest - Reflection vs Transmission Properties', fontsize=16)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(interest_df['Material'])
        

        
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('material_interest_comparison.png', dpi=300, bbox_inches='tight')
        
        print("\nMaterial ranking by performance (reflection + (1-transmission)):")
        for i, row in enumerate(interest_df.itertuples()):
            print(f"{i+1}. {row.Material}: {row.Performance:.2f}")
    else:
        print("No materials of interest found in the processed data")
else:
    print("No valid materials found for plotting")

print("\nAnalysis complete. Generated files:")
print("- material_acronym_reference.png (acronym to full name reference)")
print("- material_type_legend.png (separate material type legend)")
print("- material_coefficient_boxplots.png (box plots for min/max values)")
print("- coefficient_comparison.png (all materials comparison)")
print("- material_interest_comparison.png (materials of interest comparison)")