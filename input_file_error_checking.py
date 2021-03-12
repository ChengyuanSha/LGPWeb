import pandas as pd
import numpy as np

csv_file_name = 'sample_alzheimer_vs_normal_data.csv' # change your file name here

def check_csv_file(csv_file_name):
    try:
        df = pd.read_csv(csv_file_name)
        if 'category' not in df:
            print("Error. Please name your class column 'category' ")
        names = df.columns[1:].values
        if 'category' in names:
            print("Error. Please put 'category in the first column")
        X = df.iloc[:, 1:].values
        y = df['category'].values
        print('Successful!')
        print('Feature names: ', names)
        print('Feature category: ', np.unique(y))
        print('Feature value: ', X)
    except Exception as error:
        print("Error: " + repr(error) )


if __name__ == "__main__":
    check_csv_file(csv_file_name)