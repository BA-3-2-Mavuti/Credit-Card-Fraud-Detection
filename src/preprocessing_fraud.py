import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load and preprocess credit card fraud data.
    Returns: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(file_path)

    # Scale 'Time' and 'Amount'
    scaler = RobustScaler()
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train-test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

# Example
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('../data/creditcard.csv')
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
