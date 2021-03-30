import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = df['weight'] / (df['height'] / 100) ** 2
df.loc[bmi > 25, 'overweight'] = 1
df.loc[bmi <= 25, 'overweight'] = 0
df['overweight'] = df['overweight'].astype(np.int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['gluc'] = df['gluc'].map({1: 0, 2: 1, 3: 1}).astype(np.int)
df['cholesterol'] = df['cholesterol'].map({1: 0, 2: 1, 3: 1}).astype(np.int)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars=['id', 'cardio'],
                     value_vars=['active', 'alco', 'cholesterol', 'gluc',
                                 'overweight', 'smoke'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    # df_cat = None (done all in the passage above)

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x='variable',
                    data=df_cat,
                    hue='value',
                    col='cardio',
                    kind='count',
                    sharey=True)

    fig, axs = g.fig, g.axes[0]

    fig.suptitle('Categorical plot for people with (1) or without (0) cardiovascular disease',
                 y=1.06)

    axs[0].set_ylabel('total')
    
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    mask = (df['ap_hi'] < df['ap_lo']) | \
           (df['height'] < df['height'].quantile(0.025)) | \
           (df['height'] > df['height'].quantile(0.975)) | \
           (df['weight'] < df['weight'].quantile(0.025)) | \
           (df['weight'] > df['weight'].quantile(0.975))
    df_heat = df.drop(df.loc[mask].index)

    # Calculate the correlation matrix
    corr = df_heat.corr().round(1)

    # Generate a mask for the upper triangle
    mask = np.tril(np.ones(corr.shape), k=-1).astype(bool)

    # apply the mask to the correlation matrix
    corr.where(mask, inplace=True)  # transform elements in upper diag. to Nan

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, center=0, annot=True, fmt='.1f')  # fmt controls how annotations are displayed
    plt.xticks(rotation=90)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
