from kedro.pipeline import Pipeline, node

from .nodes import extracting_inference_data,splitting_inference_data,inference_data_treat_missing_val,inference_data_label_encoding,log_reg_Algorithm

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                 func = extracting_inference_data,
                 inputs="weather_aus_raw",
                 outputs="df_inference",
                 name = "extracting_inference_data_node",
            ),
            node(
                 func = splitting_inference_data,
                 inputs="df_inference",
                 outputs=["X_inference","y_inference"],
                 name = "splitting_inference_data",
            ),
            node(
                 func = inference_data_treat_missing_val,
                 inputs="X_inference",
                 outputs="Xinf_treat_missing_value",
                 name = "inference_data_treat_missing_val_node",
            ),
            node(
                  func = inference_data_label_encoding,
                  inputs = "Xinf_treat_missing_value",
                  outputs = "Xinf_treat_missing_values",
                  name = "inference_data_label_encoding_node",
            ),
            node(
                  func = log_reg_Algorithm,
                  inputs = ["Xinf_treat_missing_values","logreg"],
                  outputs = "data_y_pred_inf",
                  name = "log_reg_Algorithm_node",
            )   
          ]
    )