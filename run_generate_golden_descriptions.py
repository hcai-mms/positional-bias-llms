import argparse
import asyncio
import weave
import weave.trace
import weave.trace.vals
from models import model_library
from helper.batch_loading import custom_remote_iter
import subprocess
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Creator argument parser')

    parser.add_argument('-p','--project_name', 
                        type=str, 
                        default="golden_descriptions", 
                        help="wandb project name in which the evaluation will be logged in")
    parser.add_argument('-e','--entity', 
                        type=str, 
                        default="highly-biased", 
                        help="optional, wandb team name, defaults to 'highly-biased'")
    parser.add_argument('-n','--name', 
                        type=str, 
                        help="optional, defaults to '{{model}}-{{dataset}}'")
    parser.add_argument('-m','--model', 
                        type=str, 
                        required=True,
                        help=f"model name or comma seperated list of models for which to generate golden descriptions. available models: '{'\', \''.join(model_library.keys())}'")
    parser.add_argument('-d','--dataset', 
                        type=str, 
                        required=True,
                        help="name of dataset in weave, optional with version in the format 'dataset_name:v0'")
    parser.add_argument('-t','--trials', 
                        type=int, 
                        default=3,
                        help=f"Specifies how often each entry in the dataset should be evaluated")
    parser.add_argument('--page_size', 
                        type=int, 
                        default=10000,
                        help=f"Page size for downloading dataset from weave")
    parser.add_argument('--max_retries', 
                        type=int, 
                        default=3,
                        help=f"If the model output cannot be parsed how often should the model be called with the particular prompt")
    parser.add_argument('--retry_wait_sec', 
                        type=int, 
                        default=60,
                        help=f"How long should we wait before sending the prompt to the model again")

    args = parser.parse_args()

    models = [m.strip().lower() for m in args.model.split(",")]
    for model in models:
        if model not in model_library.keys():
            raise ValueError(f"'{args.model}' is not a valid model. Available Models: '{'\', \''.join(model_library.keys())}'")
    
    # overwrite weave method to enable faster dataset loading
    weave.trace.vals.WeaveTable._remote_iter = custom_remote_iter(page_size=args.page_size)


    weave.init(f"{args.entity}/{args.project_name}")

    # if dataset doesnt exists this thorws a ValueError
    dataset_ref  = weave.ref(args.dataset).get()

    for model in models:
        if hasattr(args, "name"):
            name = args.name
        else:
            name = f"{model}-{dataset_ref.name}".replace(":","-")


        evaluation = weave.Evaluation(
            name=name,
            evaluation_name=name, 
            dataset=dataset_ref,
            scorers=[],
            trials=args.trials
        )

        model_obj = model_library[model](max_retries=args.max_retries,
                                                                    retry_wait_time=args.retry_wait_sec)

        print(asyncio.run(evaluation.evaluate(model_obj)))
        if not model.startswith("gemini") and not model.startswith("gpt"):
            subprocess.run(["ollama", "stop", model_obj.model_name])
            time.sleep(10) # wait for a few seconds for model to be deleted from GPU memory
