import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    import seaborn as sns
    plt.figure(figsize=(10,6))
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()


visualize_data(combined_df)