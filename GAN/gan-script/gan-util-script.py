import pandas as pd
from nfstream import NFStreamer
import os, ast, re, yaml
import numpy as np

class gan_util:
    def __init__(self):
        pass
    def list_folders(self, directory):
        try:
            # Get a list of all items in the directory
            items = os.listdir(directory)
            # Filter out items that are not folders
            folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
            return folders
        except Exception as e:
            return 
    def list_csv_files(self, directory):
        try:
            # Get a list of all items in the directory
            items = os.listdir(directory)
            # Filter out items that are not CSV files
            csv_files = [item for item in items if item.endswith('.csv')]
            return csv_files
        except Exception as e:
            return f"An error occurred: {e}"
    
    def calculate_iqr_and_filter(self, data):
        # Calculate the 1st (Q1) and 3rd (Q3) percentiles
        Q1 = np.percentile(data, 25)  # 25th percentile
        Q3 = np.percentile(data, 75)  # 75th percentile
        IQR = Q3 - Q1  # Interquartile range
        
        # Calculate lower and upper bounds for filtering
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter data based on IQR range (values within lower and upper bounds)
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        return filtered_data

    def find_common_values_in_range(self, df):
        # Check if the dataframe has 2 columns
        if df.shape[1] == 2:
            # Apply filtering to both columns independently
            filtered_col1 = self.calculate_iqr_and_filter(df.iloc[:, 0])  # Filter for column 1
            filtered_col2 = self.calculate_iqr_and_filter(df.iloc[:, 1])  # Filter for column 2
            
            # Find common indices where both columns have valid filtered values
            common_indices = filtered_col1.index.intersection(filtered_col2.index)
            
            # Get the common values from both columns based on the common indices
            common_values = pd.DataFrame({
                df.columns[0]: df.iloc[common_indices, 0],
                df.columns[1]: df.iloc[common_indices, 1]
            })
            
            return common_values
        elif df.shape[1] == 1:
            # Filter the single column based on IQR
            filtered_col1 = self.calculate_iqr_and_filter(df.iloc[:, 0])
            
            # Return the filtered values as a DataFrame
            return pd.DataFrame({df.columns[0]: filtered_col1})
        
        else:
            raise ValueError("The DataFrame should have one or two columns.")

    def extract_data_from_filename(self, filename):
        # Regular expression pattern to match the filename structure
        pattern = re.compile(r"\[(.*?)\]_epoch_(\d+)\.csv")
        
        match = pattern.match(filename)
        if match:
            # Extract the list and epoch (remove quotes around the feature list)
            feature_list = match.group(1).replace("'", "")  # Remove quotes if present
            epoch = int(match.group(2))
            return feature_list, epoch
        else:
            return None, None  # Return None if the filename doesn't match the pattern

    def update_yaml_with_file(self, filename, yaml_file, data:list):
        # Extract data from the filename
        feature_list, epoch = self.extract_data_from_filename(filename)
        
        if feature_list is None or epoch is None:
            print(f"Filename '{filename}' doesn't match expected pattern.")
            return

        # Check if the YAML file exists and load it, otherwise create an empty dictionary
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as file:
                data = yaml.safe_load(file) or {}  # If file is empty, start with an empty dict
        else:
            data = {}

        # Check if the feature list already exists in the data
        if feature_list not in data:
            data[feature_list] = {}

        # Add the epoch to the feature list's dictionary (create an empty list for the epoch)
        data[feature_list][epoch] = data

        # Save the updated data back to the YAML file
        with open(yaml_file, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        print(f"Data from '{filename}' has been added to {yaml_file}")
        
    def df_to_yaml(self, df, yaml_file, filename):
        # Convert the DataFrame to a dictionary where columns are keys and values are lists
        df_dict = df.to_dict(orient='list')
        
        # Check if the YAML file exists and load it, otherwise create an empty dictionary
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as file:
                existing_data = yaml.safe_load(file) or {}
        else:
            existing_data = {}

        # Add the new data under the filename key (unique for each file)
        existing_data[filename] = df_dict

        # Write the updated data to the YAML file
        with open(yaml_file, 'w') as file:
            # Using 'default_flow_style=True' to write the dictionary in a single line (compact format)
            yaml.dump(existing_data, file, default_flow_style=True, allow_unicode=True)

if __name__ == "__main__":
    #main()
    pass



utile = gan_util()
addr = "/home/mehrdad/PycharmProjects/C2_communication/GAN/results/02-03-2025-23-16-20/selected_perturbed/"
yaml_file = '/home/mehrdad/PycharmProjects/C2_communication/GAN/output.yaml'
csv_list = utile.list_csv_files(addr)

for csv in csv_list:
    data = pd.read_csv(addr+csv)
    print(utile.extract_data_from_filename(csv))
    common_values = utile.find_common_values_in_range(data)
    utile.df_to_yaml(yaml_file="/home/mehrdad/PycharmProjects/C2_communication/GAN/note-book/output.yaml",
                     df= common_values.sample(100), filename=csv)
    #print(common_values.sample(100).to_dict(orient='list'))



    