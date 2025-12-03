import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data (replace with actual file path)
file_path = '30 exp data.xlsx'  # Supports csv/xlsx formats
df = pd.read_excel(file_path)  # Use pd.read_csv() if CSV

# Check dataset dimensions
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {len(df)}, Number of features: {df.shape[1] - 2}")

# Prepare stratified sampling
np.random.seed(55537)  # Fix random seed for reproducibility
target = df.iloc[:, 1]  # Target variable (second column)

# Compute quantiles of target variable for stratification
df['strata'] = pd.qcut(target,
                       q=5,  # Divide into 5 strata (adjustable based on sample size)
                       duplicates='drop',
                       labels=False)


# Stratified sampling function
def continuous_stratified_split(df, strata_col, test_size=0.2):
    test_indices = []

    # Iterate through each stratum
    for stratum in sorted(df[strata_col].unique()):
        stratum_df = df[df[strata_col] == stratum]
        n_test = max(1, round(len(stratum_df) * test_size))  # At least 1 sample

        # Random sampling from current stratum
        test_indices.extend(np.random.choice(stratum_df.index,
                                             size=n_test,
                                             replace=False))

    # Create boolean masks
    test_mask = df.index.isin(test_indices)
    train_mask = ~test_mask

    return train_mask, test_mask


# Perform stratified sampling
train_mask, test_mask = continuous_stratified_split(df, 'strata', test_size=0.2)

# Split dataset
train_df = df[train_mask].copy().reset_index(drop=True)
test_df = df[test_mask].copy().reset_index(drop=True)

# Remove temporary stratification column
for dataset in [train_df, test_df]:
    dataset.drop(columns=['strata'], inplace=True, errors='ignore')

# Check distribution consistency of the target variable
print("\nDistribution consistency check of the target variable:")
print(f"{'Statistic':<15} | {'Original':>10} | {'Train':>10} | {'Test':>10}")
print("-" * 60)
print(
    f"{'Mean':<15} | {target.mean():>10.4f} | {train_df.iloc[:, 1].mean():>10.4f} | {test_df.iloc[:, 1].mean():>10.4f}")
print(
    f"{'Std':<15} | {target.std():>10.4f} | {train_df.iloc[:, 1].std():>10.4f} | {test_df.iloc[:, 1].std():>10.4f}")
print(
    f"{'Min':<15} | {target.min():>10.4f} | {train_df.iloc[:, 1].min():>10.4f} | {test_df.iloc[:, 1].min():>10.4f}")
print(
    f"{'25% Quantile':<15} | {target.quantile(0.25):>10.4f} | {train_df.iloc[:, 1].quantile(0.25):>10.4f} | {test_df.iloc[:, 1].quantile(0.25):>10.4f}")
print(
    f"{'Median':<15} | {target.median():>10.4f} | {train_df.iloc[:, 1].median():>10.4f} | {test_df.iloc[:, 1].median():>10.4f}")
print(
    f"{'75% Quantile':<15} | {target.quantile(0.75):>10.4f} | {train_df.iloc[:, 1].quantile(0.75):>10.4f} | {test_df.iloc[:, 1].quantile(0.75):>10.4f}")
print(
    f"{'Max':<15} | {target.max():>10.4f} | {train_df.iloc[:, 1].max():>10.4f} | {test_df.iloc[:, 1].max():>10.4f}")

# Save results
train_df.to_excel('train_data.xlsx', index=False)
test_df.to_excel('test_data.xlsx', index=False)

print(f"\nSplit results: Train set {len(train_df)} samples, Test set {len(test_df)} samples")
print("Files saved: train_data.xlsx, test_data.xlsx")
