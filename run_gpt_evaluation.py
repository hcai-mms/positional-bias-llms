import argparse
import asyncio
import weave
import weave.trace
import weave.trace.vals
from scorer.library import scorer_library
from helper.batch_loading import load_dataset, custom_remote_iter
from models.gpt import *

models = {
    "gpt-4o-mini-2024-07-18":{
        "class":GPT4oMini,
        "cost":{
            "prompt_token_cost":0.075/1_000_000,
            "completion_token_cost":0.3/1_000_000,
        }
    },
    "gpt-4o-2024-11-20":{
        "class":GPT4o,
        "cost":{
            "prompt_token_cost":1.25/1_000_000,
            "completion_token_cost":5.0/1_000_000,
        }
    },
    "gpt-3.5-turbo-0125":{
        "class":GPT35Turbo,
        "cost":{
            "prompt_token_cost":0.25/1_000_000,
            "completion_token_cost":0.75/1_000_000,
        }
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Creator argument parser')

    parser.add_argument('-p','--project_name', 
                        type=str, 
                        required=True, 
                        help="wandb project name in which the evaluation will be logged in")
    parser.add_argument('-e','--entity', 
                        type=str, 
                        default="highly-biased", 
                        help="optional, wandb team name, defaults to 'highly-biased'")
    parser.add_argument('-n','--name', 
                        type=str, 
                        help="optional, defaults to '{{model}}-{{dataset}}'")
    parser.add_argument('-s','--scorer', 
                        type=str, 
                        help=f"Scorer, comma-seperated list of Scorers or 'all', to be used for evaluating the model output. available scorers: '{'\', \''.join(scorer_library.keys())}'")
    parser.add_argument('-m','--model', 
                        type=str, 
                        required=True,
                        help=f"model name which will be evaluated. available models: '{'\', \''.join(models.keys())}'")
    parser.add_argument('-d','--dataset', 
                        type=str, 
                        required=True,
                        help="name of dataset in weave, optional with version in the format 'dataset_name:v0'")
    parser.add_argument('-t','--trials', 
                        type=int, 
                        default=1,
                        help=f"Specifies how often each entry in the dataset should be evaluated")
    parser.add_argument('--split-method', 
                        type=str, 
                        default="cosine",
                        help=f"Specifies the method used to split the model output into topics Available 'cosine' and 'first'. ")
    parser.add_argument('--last-line-min-len', 
                        type=int, 
                        default=3,
                        help=f"The last line will be filted if the number of sentences is below this threshold")
    
    parser.add_argument('file_path', 
                        type=str, 
                        help="Path to the json file where the dataset is specified (./datasets/xyz.json)")
    

    args = parser.parse_args()

    if args.model.lower() not in models.keys():
        raise ValueError(f"'{args.model}' is not a valid model. Available Models: '{'\', \''.join(models.keys())}'")

    if args.scorer is None:
        scorers = []
    elif args.scorer.lower() == "all":
        scorers = scorer_library.keys()
    else:
        scorers = args.scorer.split(',')
        available_scorers = scorer_library.keys()
        invalid_scorers = [scorer for scorer in scorers if scorer not in available_scorers]
        if len(invalid_scorers) > 0:
            raise ValueError(f"'{'\', \''.join(invalid_scorers)}' is/are not a valid scorer(s). Available Scorers: '{'\', \''.join(scorer_library.keys())}'")
    
    if args.split_method.lower() not in ("cosine", "first"):
         raise ValueError(f"'{args.split_method}' is not a valid split method. Available methods: 'cosine' and 'first'")

    # overwrite weave method to enable faster dataset loading
    weave.trace.vals.WeaveTable._remote_iter = custom_remote_iter(page_size=1000)

    client = weave.init(f"{args.entity}/{args.project_name}")
    model = models[args.model.lower()]

    client.add_cost(
        llm_id=args.model.lower(),
        prompt_token_cost=model["cost"]["prompt_token_cost"],
        completion_token_cost=model["cost"]["completion_token_cost"]
    )
    

    # if dataset doesnt exists this thorws a ValueError
    dataset_ref  = weave.ref(args.dataset).get()

    if hasattr(args, "name"):
        name = args.name
    else:
        name = f"{args.model}-{dataset_ref.name}".replace(":","-")


    # Golden discription Metrics will be calculated in the reevaluation pipeline
    scorers_obj = []
    for scorer_name in scorers:
        if scorer_name.startswith("golden_description"):
            continue
        else:
            scorer = scorer_library[scorer_name]()
        
        scorers_obj.append(scorer)


    evaluation = weave.Evaluation(
        name=name,
        evaluation_name=name, 
        dataset=dataset_ref,
        scorers=scorers_obj,
        trials=args.trials
    )

    print(asyncio.run(evaluation.evaluate(model["class"](dataset_name=dataset_ref.name,
                                                gpt_output_file=args.file_path,
                                                trials=args.trials,
                                                split_method=args.split_method.lower(),
                                                last_line_min_len=args.last_line_min_len,
                                                max_retries=3,
                                                retry_wait_time=10))))
