import pandas as pd

#----Task 1: Data Loading and Cleaning----

combined_df = None

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

load_data()

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
    return stats_by_year_region

compute_stats(combined_df)