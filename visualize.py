import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data(filepath):
    col_names = ['Category', 'Type', 'Year', 'MonthCol', 'Value']

    try:
        df = pd.read_csv(filepath, usecols=[0, 1, 2, 3, 4], names=col_names, header=0, low_memory=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    df = df[df['Type'] == 'insgesamt']
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)

    df = df[df['Year'] <= 2020]

    df['MonthCol'] = df['MonthCol'].astype(str)
    df['Month_numeric'] = pd.to_numeric(df['MonthCol'].str[-2:], errors='coerce')

    invalid_month_rows = df[df['Month_numeric'].isna()]
    if not invalid_month_rows.empty:
        df.dropna(subset=['Month_numeric'], inplace=True)

    df['Month'] = df['Month_numeric'].astype(int)
    df = df.drop(columns=['Month_numeric'])

    df = df[df['Month'].between(1, 12)]

    df['Value_numeric'] = pd.to_numeric(df['Value'], errors='coerce')
    invalid_values = df[df['Value_numeric'].isna() & df['Value'].notna()]
    if not invalid_values.empty:
        df.dropna(subset=['Value_numeric'], inplace=True)
        df['Value'] = df['Value_numeric'].astype(int)
    else:
        df['Value'] = df['Value_numeric'].fillna(0).astype(int)

    if 'Value_numeric' in df.columns:
       df = df.drop(columns=['Value_numeric'])

    try:
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
    except Exception as e:
        print(f"Error creating Date column: {e}")
        return None

    df = df[['Date', 'Category', 'Year', 'Month', 'Value']]
    df = df.sort_values(by=['Category', 'Date'])
    df = df.reset_index(drop=True)

    print("Data loaded and preprocessed successfully.")
    print(f"Data shape after preprocessing: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print("Categories found:", df['Category'].unique())

    return df

def visualize_data(df):
    if df is None or df.empty:
        print("No data available for visualization.")
        return

    plt.figure(figsize=(15, 10))

    categories = df['Category'].unique()

    for category in categories:
        df_cat = df[df['Category'] == category]
        plt.plot(df_cat['Date'], df_cat['Value'], label=category)

    plt.title('Monthly Accidents per Category (until 2020)')
    plt.xlabel('Date')
    plt.ylabel('Number of Accidents (insgesamt)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plot_filename = 'plots/monthly_accidents_per_category.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == '__main__':
    data_file = 'data.csv'
    processed_data = load_and_preprocess_data(data_file)
    visualize_data(processed_data)
