import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def remove_outliers(df, column):
    """Remove outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def compute_log_log_regressions_with_min_time(data):
    """Compute log-log linear regression coefficients, R^2, and cleaned mean time for the smallest data size."""
    results = []
    
    # Loop over unique GPU groups and operations
    gpu_groups = data['group_name'].unique()
    operations = data['operation'].unique()
    transposes = data['transpose'].unique()

    for gpu_group in gpu_groups:
        for operation in operations:
            for transpose in transposes:
                # Get the cleaned data for this GPU group and operation
                operation_cleaned_data = data[(data['group_name'] == gpu_group) & (data['operation'] == operation) & (data['transpose'] == transpose)]
                
                # Prepare the cleaned data for log-log linear regression
                X_clean = np.log2(operation_cleaned_data['data_size_mb'].values).reshape(-1, 1)
                y_clean = np.log2(operation_cleaned_data['duration_sec'].values)

                # Calculate the mean time for the smallest data element
                smallest_data_size = operation_cleaned_data['data_size_mb'].min()
                smallest_data_mean_time = operation_cleaned_data[operation_cleaned_data['data_size_mb'] == smallest_data_size]['duration_sec'].mean()

                # Only perform regression if there are enough points
                if len(X_clean) > 1:
                    # Perform the linear regression on log-log scale
                    reg = LinearRegression().fit(X_clean, y_clean)
                    r_squared = reg.score(X_clean, y_clean)
                    coef = reg.coef_[0]
                    intercept = reg.intercept_

                    # Store the result with the cleaned mean time for the smallest data element
                    results.append({
                        'gpu_group': gpu_group,
                        'operation': operation,
                        'log_coef': coef,
                        'log_intercept': intercept,
                        'log_r_squared': r_squared,
                        'smallest_data_size': smallest_data_size,
                        'smallest_data_mean_time': smallest_data_mean_time,
                        'transpose': transpose
                    })

    # Return results as a DataFrame
    return pd.DataFrame(results)

def main(input_csv, output_csv):
    # Load the input CSV file
    data = pd.read_csv(input_csv)

    # Remove outliers from the data
    data_cleaned = data.groupby(['group_name', 'operation', 'data_size_mb']).apply(lambda x: remove_outliers(x, 'duration_sec')).reset_index(drop=True)

    # Compute log-log linear regression results with the smallest data time
    regression_results = compute_log_log_regressions_with_min_time(data_cleaned)

    # Save the results to CSV
    regression_results.to_csv(output_csv, index=False)
    print(f"Log-log linear regression results with mean time for smallest data size saved to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv> <output_csv>")
    else:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        main(input_csv, output_csv)
