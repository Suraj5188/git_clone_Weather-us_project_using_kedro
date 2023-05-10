from kedro.pipeline import Pipeline, node

from .nodes import split_data,train_model,evaluate_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["df2","y_training"],
                outputs=["X_train","X_test","y_train","y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train","X_test","y_train","y_test"],
                outputs="logreg",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["logreg","X_test","y_test"],
                outputs=["y_pred_test","acc"],
                name="evaluate_model_node",
            ),
        ]
    )