from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pydantic import BaseModel
from typing import List, Union, Any
import numpy as np

class PredictionInput(BaseModel):
    model: Any
    X_test: Union[np.ndarray, List[List[float]]]

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types

class PredictionOutput(BaseModel):
    predictions: List[int]

class EvaluationInput(BaseModel):
    y_test: List[int]
    y_pred: List[int]

class EvaluationOutput(BaseModel):
    accuracy: float
    classification_report: str
    confusion_matrix: List[List[int]]

def make_predictions(input_data: PredictionInput) -> PredictionOutput:
    try:
        # Ensure X_test is a numpy array
        if not isinstance(input_data.X_test, np.ndarray):
            input_data.X_test = np.array(input_data.X_test)

        # Check if the model is a TensorFlow Keras model
        if hasattr(input_data.model, 'predict'):
            predictions = input_data.model.predict(input_data.X_test)
            # If the model outputs probabilities, convert them to class predictions
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                predictions = predictions.argmax(axis=1)
        else:
            # Assuming the model is a scikit-learn model
            predictions = input_data.model.predict(input_data.X_test)
        
        return PredictionOutput(predictions=predictions.tolist())
    except Exception as e:
        print(f"Error in make_predictions function: {e}")
        raise

def evaluate_model(input_data: EvaluationInput) -> EvaluationOutput:
    # Convert y_pred to integer if necessary
    if isinstance(input_data.y_pred[0], np.integer):
        y_pred = input_data.y_pred
    else:
        y_pred = [int(pred) for pred in input_data.y_pred]

    # Evaluate the model performance
    accuracy = accuracy_score(input_data.y_test, y_pred)
    classification_rep = classification_report(input_data.y_test, y_pred)
    conf_matrix = confusion_matrix(input_data.y_test, y_pred).tolist()
    
    return EvaluationOutput(
        accuracy=accuracy,
        classification_report=classification_rep,
        confusion_matrix=conf_matrix
    )
