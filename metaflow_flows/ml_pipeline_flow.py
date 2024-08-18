from metaflow import FlowSpec, step, Parameter, current, card
import pandas as pd
from sklearn.model_selection import train_test_split
from modules.feature_transformation import preprocess_data, apply_pca
from modules.prediction import make_predictions, evaluate_model, PredictionInput, EvaluationInput
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.metrics import precision_score, recall_score, f1_score

class MLPipelineFlow(FlowSpec):

    data_path = Parameter('data_path', help='Path to the dataset', default='EEG_Brainwave/data/emotions.csv')
    tuning_flow_run_id = Parameter('tuning_flow_run_id', help='Run ID of the tuning flow', default='latest')

    def plot_confusion_matrix(self, cm, title, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, f'{title}.png'))
        plt.close()

    @step
    def start(self):
        from metaflow import Flow, Run
        mlflow.start_run(run_name="MLPipelineFlow")
        print(f"Retrieving artifacts from tuning flow {self.tuning_flow_run_id}")
        run = Flow('TuningFlow').latest_run if self.tuning_flow_run_id == 'latest' else Run(f'TuningFlow/{self.tuning_flow_run_id}')
        self.best_params_svm = run.data.best_params_svm
        self.best_params_dnn = run.data.best_params_dnn
        print("Retrieved best parameters for SVM and DNN from tuning flow.")
        mlflow.log_params(self.best_params_svm)
        mlflow.log_params(self.best_params_dnn)

        # Load the dataset
        data = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully with shape {data.shape}.")
        mlflow.log_param("dataset_shape", data.shape)

        # Preprocess the data
        data_preprocessed, labels = preprocess_data(data)
        print("Data preprocessing completed.")
        mlflow.log_param("preprocessed_shape", data_preprocessed.shape)

        # Apply PCA
        data_pca = apply_pca(data_preprocessed)
        print("PCA transformation applied.")
        mlflow.log_param("pca_shape", data_pca.shape)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_pca, labels, test_size=0.3, random_state=42)
        print(f"Data split into training and testing sets. Training set shape: {self.X_train.shape}, Testing set shape: {self.X_test.shape}.")
        mlflow.log_param("train_shape", self.X_train.shape)
        mlflow.log_param("test_shape", self.X_test.shape)

        self.next(self.train_svm, self.train_dnn)

    @step
    def train_svm(self):
        from sklearn.svm import SVC
        self.model_svm = SVC(**self.best_params_svm)
        self.model_svm.fit(self.X_train, self.y_train)
        print("SVM model trained.")
        mlflow.sklearn.log_model(self.model_svm, "SVM_Model")
        self.next(self.join_train)

    @step
    def train_dnn(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        self.model_dnn = Sequential()
        self.model_dnn.add(Dense(self.best_params_dnn['neurons'], input_dim=self.X_train.shape[1], activation='relu'))
        self.model_dnn.add(Dropout(self.best_params_dnn['dropout_rate']))
        self.model_dnn.add(Dense(self.best_params_dnn['neurons'], activation='relu'))
        self.model_dnn.add(Dense(3, activation='softmax'))

        optimizer = Adam(learning_rate=0.001)
        self.model_dnn.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model_dnn.fit(self.X_train, self.y_train, epochs=self.best_params_dnn['epochs'], batch_size=self.best_params_dnn['batch_size'], verbose=2)
        print("DNN model trained.")
        mlflow.tensorflow.log_model(self.model_dnn, "DNN_Model")
        self.next(self.join_train)

    @step
    def join_train(self, inputs):
        self.model_svm = inputs.train_svm.model_svm
        self.model_dnn = inputs.train_dnn.model_dnn
        self.X_test = inputs.train_svm.X_test
        self.y_test = inputs.train_svm.y_test
        self.best_params_svm = inputs.train_svm.best_params_svm
        self.best_params_dnn = inputs.train_dnn.best_params_dnn
        self.next(self.evaluate_models)

    @step
    def evaluate_models(self):
        try:
            # Evaluating SVM model
            print("Evaluating SVM model.")
            prediction_input_svm = PredictionInput(model=self.model_svm, X_test=self.X_test.tolist())
            prediction_output_svm = make_predictions(prediction_input_svm)
            evaluation_input_svm = EvaluationInput(y_test=self.y_test.tolist(), y_pred=prediction_output_svm.predictions)
            self.evaluation_output_svm = evaluate_model(evaluation_input_svm)
            print("SVM evaluation completed.")
            print("SVM Evaluation Output:", self.evaluation_output_svm)

            # Log SVM evaluation metrics
            mlflow.log_metrics({
                "svm_accuracy": self.evaluation_output_svm.accuracy,
                "svm_precision": precision_score(self.y_test, prediction_output_svm.predictions, average='weighted'),
                "svm_recall": recall_score(self.y_test, prediction_output_svm.predictions, average='weighted'),
                "svm_f1_score": f1_score(self.y_test, prediction_output_svm.predictions, average='weighted')
            })
            mlflow.log_dict({"confusion_matrix": self.evaluation_output_svm.confusion_matrix}, "svm_confusion_matrix.json")
            mlflow.log_text(self.evaluation_output_svm.classification_report, "svm_classification_report.txt")

            # Evaluating DNN model
            print("Evaluating DNN model.")
            prediction_input_dnn = PredictionInput(model=self.model_dnn, X_test=self.X_test.tolist())
            prediction_output_dnn = make_predictions(prediction_input_dnn)
            evaluation_input_dnn = EvaluationInput(y_test=self.y_test.tolist(), y_pred=prediction_output_dnn.predictions)
            self.evaluation_output_dnn = evaluate_model(evaluation_input_dnn)
            print("DNN evaluation completed.")
            print("DNN Evaluation Output:", self.evaluation_output_dnn)

            # Log DNN evaluation metrics
            mlflow.log_metrics({
                "dnn_accuracy": self.evaluation_output_dnn.accuracy,
                "dnn_precision": precision_score(self.y_test, prediction_output_dnn.predictions, average='weighted'),
                "dnn_recall": recall_score(self.y_test, prediction_output_dnn.predictions, average='weighted'),
                "dnn_f1_score": f1_score(self.y_test, prediction_output_dnn.predictions, average='weighted')
            })
            mlflow.log_dict({"confusion_matrix": self.evaluation_output_dnn.confusion_matrix}, "dnn_confusion_matrix.json")
            mlflow.log_text(self.evaluation_output_dnn.classification_report, "dnn_classification_report.txt")

        except Exception as e:
            print(f"Error in evaluate_models step: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        self.next(self.plot_svm_results)

    @card(type='matplotlib')
    @step
    def plot_svm_results(self):
        self.plot_confusion_matrix(self.evaluation_output_svm.confusion_matrix, 'SVM Confusion Matrix', 'output')
        current.card.append(f"<h3>SVM Classification Report</h3><pre>{self.evaluation_output_svm.classification_report}</pre>")
        mlflow.log_artifact(os.path.join('output', 'SVM Confusion Matrix.png'))
        self.next(self.plot_dnn_results)

    @card(type='matplotlib')
    @step
    def plot_dnn_results(self):
        self.plot_confusion_matrix(self.evaluation_output_dnn.confusion_matrix, 'DNN Confusion Matrix', 'output')
        current.card.append(f"<h3>DNN Classification Report</h3><pre>{self.evaluation_output_dnn.classification_report}</pre>")
        mlflow.log_artifact(os.path.join('output', 'DNN Confusion Matrix.png'))
        self.next(self.generate_report)

    @card
    @step
    def generate_report(self):
        output_dir = 'output'
        report_path = os.path.join(output_dir, 'model_report.html')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(report_path, 'w') as f:
            f.write('<html><head><title>Model Training Report</title></head><body>')
            f.write('<h1>Model Training Report</h1>')

            # SVM results
            f.write('<h2>SVM Results</h2>')
            f.write('<p>Best Parameters: {}</p>'.format(json.dumps(self.best_params_svm, indent=4)))
            f.write('<p>Accuracy: {}</p>'.format(self.evaluation_output_svm.accuracy))
            f.write('<h3>Classification Report</h3><pre>{}</pre>'.format(self.evaluation_output_svm.classification_report))
            f.write('<h3>Confusion Matrix</h3><img src="SVM Confusion Matrix.png" alt="SVM Confusion Matrix">')

            # DNN results
            f.write('<h2>DNN Results</h2>')
            f.write('<p>Best Parameters: {}</p>'.format(json.dumps(self.best_params_dnn, indent=4)))
            f.write('<p>Accuracy: {}</p>'.format(self.evaluation_output_dnn.accuracy))
            f.write('<h3>Classification Report</h3><pre>{}</pre>'.format(self.evaluation_output_dnn.classification_report))
            f.write('<h3>Confusion Matrix</h3><img src="DNN Confusion Matrix.png" alt="DNN Confusion Matrix">')

            f.write('</body></html>')

        print("Report generated at", report_path)

        # Log the report as an artifact in MLflow
        mlflow.log_artifact(report_path)

        # Append the HTML report to the card
        with open(report_path, 'r') as f:
            report_content = f.read()
        current.card.append(report_content)

        self.next(self.end)

    @step
    def end(self):
        print("Model training and evaluation completed.")
        mlflow.end_run()


if __name__ == '__main__':
    MLPipelineFlow()