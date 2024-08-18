import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def perform_correlation_analysis(data_path: str, output_dir: str):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Encode the target variable
    label_encoder = LabelEncoder()
    data['label_encoded'] = label_encoder.fit_transform(data['label'])

    # Compute the correlation matrix for all numeric features, including the encoded label
    correlation_matrix = data.corr()

    # Get the correlation values for 'label_encoded'
    correlation_with_label = correlation_matrix['label_encoded'].abs().sort_values(ascending=False)

    # Convert the correlation values to a dictionary
    correlation_dict = correlation_with_label.to_dict()

    # Save the correlation values to a JSON file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'correlation_values.json'), 'w') as f:
        json.dump(correlation_dict, f, indent=4)

    print("Correlation values saved to correlation_values.json")

    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    
    # Save the plot to the output directory
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.show()

    # Generate HTML report
    with open(os.path.join(output_dir, 'correlation_report.html'), 'w') as f:
        f.write('<html><head><title>Correlation Analysis Report</title></head><body>')
        f.write('<h1>Correlation Analysis</h1>')
        f.write('<img src="correlation_matrix.png" alt="Correlation Matrix">')
        f.write('<h2>Correlation Values</h2><pre>{}</pre>'.format(json.dumps(correlation_dict, indent=4)))
        f.write('</body></html>')

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/emotions.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output'
    perform_correlation_analysis(data_path, output_dir)
