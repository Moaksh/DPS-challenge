import matplotlib.pyplot as plt
import os
from data_utils import load_and_preprocess_data

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
