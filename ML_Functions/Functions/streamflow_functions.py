import os
import pandas as pd
import glob

def check_streamflow_data(directory_path, start_date, end_date):
    """
    Check CSV files for NaN values in streamflow between specified dates. This means all the basins have entries in these dates, which means we can randomly chose date-basin combinations in training 
    
    Args:
        directory_path (str): Path to directory containing CSV files
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        tuple: (valid_files, invalid_files) where each is a list of filenames
    """
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    valid_files = []
    invalid_files = []
    
    for file_path in csv_files:
        try:
            # Read the CSV file
            # print(file_path)
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            # Filter for date range
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df_filtered = df.loc[mask]
            filename = os.path.splitext(os.path.basename(file_path))[0]
            # Check for NaN values in streamflow column
            if df_filtered['streamflow'].isna().any():
                invalid_files.append(os.path.basename(filename))
            else:
                valid_files.append(os.path.basename(filename))
                # print("found one!")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            filename = os.path.splitext(os.path.basename(file_path))[0]
            invalid_files.append(os.path.basename(filename))
    
    return valid_files, invalid_files



def check_streamflow_data_recursively(directory_path, start_date, end_date): # Takes about 25 seconds to process 100 stations, 
    """
    Recursively check CSV files in all subdirectories for NaN values in streamflow between specified dates.
    
    Args:
        directory_path (str): Path to root directory containing CSV files and subdirectories
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        tuple: (valid_files, invalid_files) where each is a list of filenames with relative paths
    """
    valid_files = []
    invalid_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory_path):
        counter = 0
        for file in files:
            if file.endswith('.csv'):
                counter += 1
                if counter % 1000 == 0:
                    print("1000 stations processed")
                file_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path, parse_dates=['date'])
                    
                    # Filter for date range
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    df_filtered = df.loc[mask]
                    
                    # Get relative path from the root directory
                    relative_path = os.path.relpath(file_path, directory_path)
                    filename = os.path.splitext(relative_path)[0]
                    
                    # Check for NaN values in streamflow column
                    if df_filtered['streamflow'].isna().any():
                        invalid_files.append(filename)
                    else:
                        valid_files.append(filename)
                        # print("found a suitable basin")
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    relative_path = os.path.relpath(file_path, directory_path)
                    filename = os.path.splitext(relative_path)[0]
                    invalid_files.append(filename)
    
    return valid_files, invalid_files