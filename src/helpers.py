import os
import pandas as pd

def export_to_csv_from_dict(data_dict, output_dir = '.', file_name = 'train.csv'):
    train_patients, train_outcomes = data_dict[0], data_dict[1]
    df1 = pd.DataFrame(train_patients, columns=["Patient"])
    df2 = pd.DataFrame(train_outcomes, columns=["Status", "Time"])
    df = pd.concat([df1, df2], axis=1)
    df["Status"] = df["Status"].astype(int)
    
    # Export the DataFrame to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, file_name), index=False)

    return
