import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def perform_eda(data_path: str, output_dir: str):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dataset Information
    dataset_info = data.info(buf=None)
    dataset_info_str = str(dataset_info)

    # Summary statistics of the dataset
    summary_stats = data.describe().to_dict()

    # Check for missing values
    missing_values = data.isnull().sum().to_dict()

    # Distribution of the target variable (label)
    label_distribution = data['label'].value_counts().to_dict()

    # Compile the information into a JSON object
    eda_results = {
        "dataset_info": dataset_info_str,
        "summary_statistics": summary_stats,
        "missing_values": missing_values,
        "label_distribution": label_distribution
    }

    # Save the JSON file
    with open(os.path.join(output_dir, 'eda_results.json'), 'w') as f:
        json.dump(eda_results, f, indent=4)

    # Plotting the distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=data)
    plt.title('Distribution of Emotion Labels')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    
    # Save the plot to the output directory
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
    plt.show()

    print("EDA results saved to eda_results.json and label_distribution.png")

    # Generate HTML report
    with open(os.path.join(output_dir, 'eda_report.html'), 'w') as f:
        f.write('<html><head><title>EDA Report</title></head><body>')
        f.write('<h1>Exploratory Data Analysis</h1>')
        f.write('<h2>Dataset Information</h2><pre>{}</pre>'.format(dataset_info_str))
        f.write('<h2>Summary Statistics</h2><pre>{}</pre>'.format(json.dumps(summary_stats, indent=4)))
        f.write('<h2>Missing Values</h2><pre>{}</pre>'.format(json.dumps(missing_values, indent=4)))
        f.write('<h2>Label Distribution</h2><pre>{}</pre>'.format(json.dumps(label_distribution, indent=4)))
        f.write('<img src="label_distribution.png" alt="Label Distribution">')
        f.write('</body></html>')


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/emotions.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output'
    perform_eda(data_path, output_dir)
