from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from typing import Dict
from kedro.pipeline import Pipeline 
from kedro.extras.datasets.pickle import PickleDataSet

from weather_us.pipelines import data_processing as dp
from weather_us.pipelines import data_science as ds
from weather_us.pipelines import inferance as infer


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    inferance_pipeline = infer.create_pipeline()

    return {
        "__default__": data_processing_pipeline+data_science_pipeline+inferance_pipeline,
        "dp": data_processing_pipeline,
        "ds": data_science_pipeline,
        "infer":inferance_pipeline,
    }