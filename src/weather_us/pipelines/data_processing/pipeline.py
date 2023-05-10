from kedro.pipeline import Pipeline, node

from .nodes import extract_training_data,treat_missing,training_data_split,lebel_encoding_filling_null

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=extract_training_data,
                inputs="weather_aus_raw",
                outputs="df1",
                name="extract_training_data_node",
            ),
            node(
                func=treat_missing,
                inputs="df1",
                outputs="df1_treat_training_data",
                name="treat_missing_node",
            ),
            node(
                func=training_data_split,
                inputs="df1_treat_training_data",
                outputs=["X_training","y_training"],
                name="training_data_node",
            ),
            node(
                func=lebel_encoding_filling_null,
                inputs="X_training",
                outputs="df2",
                name="label_encoding_filling_null_node",
            ),
        ]
    )