import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import argparse

def main(data_root):
    metadata_path = os.path.join(data_root, 'metadata.csv')
    df = pd.read_csv(metadata_path)

    train_df, tmp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    output_dir = os.path.join(data_root, 'metadata') 
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # The code splits the dataset into Train: 428, Val: 92, Test: 92
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train, val, test sets')
    parser.add_argument('--data_root', type=str, default='/data1/yoojinoh/codes/mlops/data', help='Root directory of the dataset')
    args = parser.parse_args()
    main(args.data_root)