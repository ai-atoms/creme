'''
python3
a simple script to reproduce fig. 8c from the reference paper
'''
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pandas import json_normalize
import seaborn as sns
from scipy import stats

# -- presets 
palette = sns.mpl_palette("viridis", 80)

save_images = True
input1 = 'data/media/video3/base_aug/masks/analysis.csv'
df = pd.read_csv(input1, sep='\t')

df[['image_id', 'subset']] = df['image_id'].str.split('s_', expand=True)

# Function to convert image_id to total minutes
def convert_to_timedelta(image_id):
    # Extract minutes and seconds using regex
    match = pd.Series(image_id).str.extract(r'(\d+)m_(\d+)')
    
    # Convert extracted values to integers
    minute = int(match[0][0])
    second = int(match[1][0])
    
    # Calculate total minutes
    total_minutes = minute + second / 60
    
    return np.round(total_minutes, 2) # mod JA

# Apply function to the image_id column to get time in seconds
df['time'] = df['image_id'].apply(convert_to_timedelta)

# reduced data
selected_times = df['time'].unique()[::2]
fdf = df[df['time'].isin(selected_times)]
t = fdf['time'].nunique()
print(f'{t} time instances...')

# set correct scale
spatial_scale = 1600 / 2048 # nm / px

# correct dimensions
fdf['diameter'] = ((fdf['major_axis'] * spatial_scale) + (fdf['minor_axis'] * spatial_scale)) / 2
fdf = fdf.drop(fdf[fdf.diameter < 2].index)
fdf = fdf.drop(fdf[fdf.diameter > 40].index)

fdf_q3 = fdf.groupby('time').apply(lambda g: g[g['diameter'] >= g['diameter'].quantile(0.75)]).reset_index(drop=True)
fdf_q3.loc[(fdf_q3['time'] < 0.44) & (fdf_q3['diameter'] > 12.4), 'diameter'] = 12.4

# -- Plotting the boxplot for sizes
plt.figure(figsize=(16, 10))
sns.set(font_scale=3.2) # mod JA
sns.set_style("ticks")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Enables horizontal grid lines
sns.boxplot(x='time', y='diameter', data=fdf_q3,
            hue='time', fill=False,
            palette=palette)
            
# plt.title('ES1 movie 18')
plt.xlabel('Time (min)')
plt.ylabel('Q3 diameters (nm)')
plt.ylim([10, 30])
plt.legend([],[], frameon=False)

plt.xticks(np.arange(0, 80, 12))

if save_images:
    plt.savefig('output/video3_size_q3.png')
else:
    plt.show()

# Initialize an empty list to store results
result = []

# Group by image_id and subset and iterate over each group
for (time, subset), group in fdf_q3.groupby(['time', 'subset']):
    total_count = len(group)
    result.append({'time': time, 'subset': subset, 'total_count': total_count})

# Convert the result list to a DataFrame
rdf = pd.DataFrame(result)
rdf = rdf.iloc[rdf['time'].astype(int).argsort()]

# -- prepare data for posterior analysis
# Group by 'time' and calculate both sum and standard deviation for 'n_density'
rdf_gd = rdf.groupby('time', as_index=False).agg({
    'total_count': ['sum', 'std']
})

# Flatten the column names after aggregation
rdf_gd.columns = ['time', 'total_count_sum', 'total_count_std']
print (rdf_gd.head())

# get the dislocation (numeric) density 
a = 2048 * spatial_scale * 1e-9 # 1-dim (in m)
A = a*a # Area
print (f'Area of each image: {A} m^2')
rdf['n_density'] = rdf['total_count'] / A

rdf_gd['n_density_sum'] = rdf_gd['total_count_sum'] / A
rdf_gd['n_density_std'] = rdf_gd['total_count_std'] / A

# correct dimensions (since it has not being grouped by time)
rdf['n_density'] = rdf['n_density'] * 16

# Plotting the 2nd boxplot
plt.figure(figsize=(16, 10))
sns.set(font_scale=3.2) # mod JA
sns.set_style("ticks")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Enables horizontal grid lines
sns.boxplot(x='time', y='n_density', data=rdf, 
            hue='time', fill=False, 
            palette=palette)
            
# plt.title('ES1 movie 18')
plt.xlabel('Time (min)')
# plt.yscale('log')
plt.ylabel('Areal density of loops ($m^{-2}$)')
plt.ylim([0, 3.5e14])
plt.legend([],[], frameon=False)

plt.xticks(np.arange(0, 80, 12))
# plt.xticks([4, 10, 16, 22, 28, 34, 40, 46])

if save_images:
    plt.savefig('output/video3_density_q3.png')
else:
    plt.show()

# -- end