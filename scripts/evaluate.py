import lighteval, os, load_dotenv
load_dotenv.load_dotenv()
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_package_available

if is_package_available("accelerate"):
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def main():
    print("Starting evaluation...")
    evaluation_tracker = EvaluationTracker(
        output_dir="./.results",
        save_details=True,
        push_to_hub=True,
        hub_results_org=os.getenv("HF_ORG"), 
    )
    print("Evaluation tracker initialized.")
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory=None,  # Set to path if using custom tasks
        # Remove the parameter below once your configuration is tested
        max_samples=10
    )
    print("Pipeline parameters set.")
    model_config = VLLMModelConfig(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        dtype="float16",
    )
    print("Model configuration set.")
    task = "gsm8k|5"

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    print("Pipeline initialized.")
    pipeline.evaluate()
    print("Evaluation completed.")
    pipeline.save_and_push_results()
    print("Results saved and pushed.")
    pipeline.show_results()
    print("Results displayed.")

if __name__ == "__main__":
    print("Running script...")
    main()