import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#----Task 1: Data Loading and Cleaning----

def load_data():

    filename_list = []
    pattern = "world_happiness/world_happiness_{}.csv"
    for i in range(2015, 2025):
        filename = pattern.format(i)
        filename_list.append(filename)

    df_list = []

    for file in filename_list:
        df = pd.read_csv(file, sep = ';', decimal=',')
        df['Year'] = file.split('_')[-1].split('.')[0] 
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['Happiness score'] = combined_df['Happiness score'].combine_first(combined_df['Ladder score'])
    combined_df.drop(columns=['Ladder score'], inplace=True)
#    print(combined_df.head())
#    print(combined_df.columns)
#    print(combined_df.shape)
    
    combined_df['Year'] = combined_df['Year'].astype(int)
    print(combined_df.dtypes)
    combined_df.to_csv("outputs/merged_happiness.csv", index=False)
    return combined_df

combined_df = load_data()

#----Task 2: Compute Statistics----

def compute_stats(df):
    happiness_score_stats = {
        "mean": df['Happiness score'].mean(),
        "median": df['Happiness score'].median(),
        "std": df['Happiness score'].std()
    }
    stats_by_year_region = df.groupby(['Year', 'Regional indicator'])['Happiness score'].agg(['mean', 'median', 'std'])
    print(happiness_score_stats)
    print(stats_by_year_region)

compute_stats(combined_df)

#----Task 3: Visual Exploration----

def visualize_data(df):

# A histogram of all happiness scores across all years. Save as happiness_histogram.png.
    plt.figure(figsize=(10,6))
    plt.hist(df['Happiness score'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Happiness Scores (2015-2024)')
    plt.xlabel('Happiness Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('outputs/happiness_histogram.png')
    plt.close()

# A boxplot comparing happiness score distributions across years (one box per year).
    plt.figure(figsize=(10,6))
    scores_by_year = df.groupby("Year")['Happiness score']
    scores = []
    years = []
    for year, score_range in scores_by_year:
        scores.append(score_range)
        years.append(year)        
    plt.boxplot(scores, labels=years)
    plt.title('Happiness score distribution across years')
    plt.ylabel('Happiness score')
    plt.savefig('outputs/happiness_by_year.png')
    plt.close()

# A scatter plot showing the relationship between GDP per capita and happiness score.

    plt.figure(figsize=(10,6))
    plt.scatter(df['GDP per capita'], df['Happiness score'], color='salmon')
    plt.title('GDP per capita vs. Happiness score')
    plt.xlabel('GDP per capita')
    plt.ylabel('Happiness score')
    plt.savefig('outputs/gdp_vs_happiness.png')
    plt.close()

# A correlation heatmap (using sns.heatmap() with annot=True) showing the Pearson correlations between all numeric columns. 

    plt.figure(figsize=(10,6))
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()

visualize_data(combined_df)

#----Task 4: Hypothesis Testing----

# run an independent samples t-test comparing happiness scores from 2019 to 2020
# Log the t-statistic, p-value, the mean happiness for each group, and a 
# plain-language interpretation of the result at alpha = 0.05. 
def hypothesis_testing(df):
    
    scores_2019 = df[df['Year'] == 2019]['Happiness score']
    scores_2020 = df[df['Year'] == 2020]['Happiness score']

    t_stat, p_val = stats.ttest_ind(scores_2019, scores_2020)

    print("t-statistic:", t_stat)
    print("p-value:", p_val)
    print("Mean happiness score in 2019:", scores_2019.mean())
    print("Mean happiness score in 2020:", scores_2020.mean())

    if p_val < 0.05:
        print("The difference is statistically significant.")
    else:
        print("No statistically significant difference detected.")

    w_europe = df[(df['Year'] == 2019)&(df['Regional indicator'] == 'Western Europe')]['Happiness score']
    l_america = df[(df['Year'] == 2019)&(df['Regional indicator'] == 'Latin America and Caribbean')]['Happiness score']
    t_stat, p_val = stats.ttest_ind(w_europe, l_america)
    print("t-statistic:", t_stat)
    print("p-value:", p_val)
    print("Mean happiness score in Western Europe in 2019:", w_europe.mean())
    print("Mean happiness score in Latin America and Caribbean in 2019:", l_america.mean())
    if p_val < 0.05:
        print("The difference is statistically significant.")
    else:
        print("No statistically significant difference detected.")

hypothesis_testing(combined_df)

#-----Task 5: Correlation and Multiple Comparisons-----

# For each numeric explanatory variable, compute the Pearson correlation with happiness score using 
# scipy.stats.pearsonr and log the coefficient and p-value.


def correlation_analysis(df):
    numeric_df = df.select_dtypes(include='number')
    counter = 0
    results = []
    for column in numeric_df.columns:
        if column != 'Happiness score':
            r, p = stats.pearsonr(df['Happiness score'], numeric_df[column])
            results.append((column, r, p))
            counter += 1
    
    adjusted_alpha = 0.05 / counter
    print(f"Adjusted alpha for multiple comparisons: {adjusted_alpha:.4f}")

    for column, r, p in results:
        sig_original = p < 0.05
        sig_adjusted = p < adjusted_alpha

        print(
            f"{column}: r={r:.2f}, p={p:.4f}, "
            f"significant (0.05): {sig_original}, "
            f"significant (adjusted): {sig_adjusted}"
        )

correlation_analysis(combined_df)

#------Task 6: Summary Report------

def summary_report(df):

    # Total number of countries and years in the merged dataset.
    countries = df['Country'].unique()
    years = df['Year'].unique()
    print(f"Number of countries: {len(countries)}")
    print(f"Number of years: {len(years)}")

    # The top 3 and bottom 3 regions by mean happiness score.
    region_means = df.groupby('Regional indicator')['Happiness score'].mean()
    top_3 = region_means.nlargest(3)
    bottom_3 = region_means.nsmallest(3)

    print("\nTop 3 regions by mean happiness score:")
    for region, mean_score in top_3.items():
        print(f"  {region}: {mean_score:.2f}")

    print("\nBottom 3 regions by mean happiness score:")
    for region, mean_score in bottom_3.items():
        print(f"  {region}: {mean_score:.2f}")

    # A plain-language interpretation of the t-test comparing 2019 and 2020 happiness scores.
    print('The independent samples t-test comparing 2019 and 2020 happiness scores did not find ' \
    'a statistically significant difference at the 0.05 level. ' \
    'In this dataset, average happiness in 2020 was not meaningfully different from 2019 overall.)')
    
    # The variable most strongly correlated with happiness score (after Bonferroni correction)
    numeric_df = df.select_dtypes(include='number')
    results = []
    for column in numeric_df.columns:
        if column != 'Happiness score':
            r, p = stats.pearsonr(df['Happiness score'], numeric_df[column])
            results.append((column, r, p))

    adjusted_alpha = 0.05 / len(results)
    bonferroni_significant = [
    (column, r, p) for column, r, p in results if p < adjusted_alpha
]

    if bonferroni_significant:
        strongest = max(bonferroni_significant, key=lambda x: abs(x[1]))
        print(
            f"Strongest correlation after Bonferroni correction: {strongest[0]} "
            f"(r={strongest[1]:.2f}, p={strongest[2]:.4f})"
        )
    else:
        print("No variables remained significant after Bonferroni correction.")

summary_report(combined_df)
