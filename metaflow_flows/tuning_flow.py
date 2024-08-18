from metaflow import FlowSpec, step, Parameter
from modules.tune_SVM import tune_svm_model
from modules.tune_DNN import tune_dnn_model
from modules.feature_transformation import preprocess_data
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TuningFlow(FlowSpec):
    
    data_path = Parameter('data_path', help='Path to the dataset', default='EEG_Brainwave/data/emotions.csv')

    @step
    def start(self):
        print("Starting the tuning flow.")
        mlflow.start_run(run_name="TuningFlow")
        
        # Log the data path
        mlflow.log_param("data_path", self.data_path)
        
        # Load the dataset
        data = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully with shape {data.shape}.")
        
        # Log dataset info
        mlflow.log_param("dataset_shape", data.shape)
        
        self.data_preprocessed, self.labels = preprocess_data(data)
        print("Data preprocessing completed.")
        
        # Log preprocessing info
        mlflow.log_param("preprocessed_shape", self.data_preprocessed.shape)
        
        self.next(self.split_data)

    @step
    def split_data(self):
        print("Splitting the data into training and testing sets.")
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_preprocessed, self.labels, test_size=0.3, random_state=42)
        print(f"Data split completed. Training set shape: {self.X_train.shape}, Testing set shape: {self.X_test.shape}.")
        
        # Log split info
        mlflow.log_param("train_shape", self.X_train.shape)
        mlflow.log_param("test_shape", self.X_test.shape)
        
        self.next(self.tune_svm, self.tune_dnn)

    @step
    def tune_svm(self):
        print("Tuning the SVM model.")
        mlflow.start_run(run_name="Tune SVM")
        self.best_params_svm, self.best_model_svm = tune_svm_model(self.X_train, self.y_train)
        print(f"SVM tuning completed. Best parameters: {self.best_params_svm}")
        
        # Log parameters and model to MLflow
        mlflow.log_params(self.best_params_svm)
        mlflow.sklearn.log_model(self.best_model_svm, "SVM_Model")
        
        # Evaluate and log metrics
        y_pred = self.best_model_svm.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        mlflow.end_run()
        self.next(self.join)

    @step
    def tune_dnn(self):
        print("Tuning the DNN model.")
        mlflow.start_run(run_name="Tune DNN")
        self.best_params_dnn, self.best_model_dnn = tune_dnn_model(self.X_train, self.y_train)
        print(f"DNN tuning completed. Best parameters: {self.best_params_dnn}")
        
        # Log parameters and model to MLflow
        mlflow.log_params(self.best_params_dnn)
        
        # Log DNN model
        input_example = np.array([self.X_train[0]])
        signature = mlflow.models.infer_signature(self.X_train, self.best_model_dnn.model.predict(self.X_train))
        mlflow.tensorflow.log_model(self.best_model_dnn.model, "DNN_Model", signature=signature, input_example=input_example)
        
        # Evaluate and log metrics
        y_pred = self.best_model_dnn.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(self.y_test, y_pred_classes)
        precision = precision_score(self.y_test, y_pred_classes, average='weighted')
        recall = recall_score(self.y_test, y_pred_classes, average='weighted')
        f1 = f1_score(self.y_test, y_pred_classes, average='weighted')
        
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        mlflow.end_run()
        self.next(self.join)

    @step
    def join(self, inputs):
        print("Joining the results from SVM and DNN tuning.")
        self.best_params_svm = inputs.tune_svm.best_params_svm
        self.best_model_svm = inputs.tune_svm.best_model_svm
        self.best_params_dnn = inputs.tune_dnn.best_params_dnn
        self.best_model_dnn = inputs.tune_dnn.best_model_dnn
        self.next(self.end)

    @step
    def end(self):
        print("Tuning completed.")
        print("Best SVM Parameters:", self.best_params_svm)
        print("Best DNN Parameters:", self.best_params_dnn)
        mlflow.end_run()

if __name__ == '__main__':
    TuningFlow()