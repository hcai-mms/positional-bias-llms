import argparse
import asyncio
import weave
import weave.trace
import weave.trace.vals
from models import model_library
from scorer.library import scorer_library
from helper.batch_loading import custom_remote_iter
from collections import defaultdict


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
    parser.add_argument('-m','--model', 
                        type=str, 
                        required=True,
                        help=f"model name which will be evaluated. available models: '{'\', \''.join(model_library.keys())}'")
    parser.add_argument('-d','--dataset', 
                        type=str, 
                        required=True,
                        help="name of dataset in weave, optional with version in the format 'dataset_name:v0'")
    parser.add_argument('-s','--scorer', 
                        type=str, 
                        required=True,
                        help=f"Scorer, comma-seperated list of Scorers or 'all', to be used for evaluating the model output. available scorers: '{'\', \''.join(scorer_library.keys())}'")
    parser.add_argument('-t','--trials', 
                        type=int, 
                        default=1,
                        help=f"Specifies how often each entry in the dataset should be evaluated")
    parser.add_argument('-temp','--temperature', 
                        type=float, 
                        default=0.5,
                        help=f"Temperature to be given to the LLM")
    parser.add_argument('--split-method', 
                        type=str, 
                        default="cosine",
                        help=f"Specifies the method used to split the model output into topics Available 'cosine' and 'first'. ")
    parser.add_argument('--last-line-min-len', 
                        type=int, 
                        default=3,
                        help=f"The last line will be filted if the number of sentences is below this threshold")
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
    parser.add_argument('--endpoint', 
                        type=str, 
                        default="localhost",
                        help=f"address of ollama endpoint to be used")

    args = parser.parse_args()

    model = args.model.lower()

    if model not in model_library.keys():
        raise ValueError(f"'{args.model}' is not a valid model. Available Models: '{'\', \''.join(model_library.keys())}'")
    
    require_golden = False

    if args.scorer.lower() == "all":
        scorers = scorer_library.keys()
        require_golden = True
    else:
        scorers = args.scorer.split(',')
        available_scorers = scorer_library.keys()
        invalid_scorers = []
        for scorer in scorers:
            if scorer not in available_scorers:
                invalid_scorers.append(scorer)
            elif scorer.startswith("golden_description"):
                require_golden = True
        if len(invalid_scorers) > 0:
            raise ValueError(f"'{'\', \''.join(invalid_scorers)}' is/are not a valid scorer(s). Available Scorers: '{'\', \''.join(scorer_library.keys())}'")

    if args.split_method.lower() not in ("cosine", "first"):
         raise ValueError(f"'{args.split_method}' is not a valid split method. Available methods: 'cosine' and 'first'")

    # overwrite weave method to enable faster dataset loading
    weave.trace.vals.WeaveTable._remote_iter = custom_remote_iter(page_size=args.page_size)
    

    golden_descritions = defaultdict(list)
    if require_golden:
        client = weave.init(f"{args.entity}/golden_descriptions")
        calls = client.get_calls(
            filter={"trace_roots_only": True},
             query={"$expr":{"$contains":{"input":{"$getField":"inputs.self"},"substr":{"$literal":model.replace(":","-")+":"}}}},
            sort_by=[{"field":"started_at","direction":"desc"}],
        )

        calls = client.get_calls(
            filter={"call_ids":[call.id for call in calls]}
        )
        calls = list(calls)

        if len(calls) > 1:
            raise RuntimeError(f"Found multiple Calls for model '{model}'. Please make sure all golden descirptions are contained in one trace")

        for child in calls[0].children():
            if child.func_name == 'Evaluation.summarize':
                continue
            topic = child.inputs["example"]["user_prompt"]
            golden_descritions[topic].append(child.output["output"]["split"][topic]["text"])


    client = weave.init(f"{args.entity}/{args.project_name}")

    # if dataset doesnt exists this thorws a ValueError
    dataset_ref  = weave.ref(args.dataset).get()

    if hasattr(args, "name"):
        name = args.name
    else:
        name = f"{args.model}-{dataset_ref.name}".replace(":","-")

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
    print(asyncio.run(evaluation.evaluate(model_library[args.model.lower()](max_retries=args.max_retries,
                                                                            ollama_endpoint=args.endpoint,
                                                                            retry_wait_time=args.retry_wait_sec,
                                                                            split_method=args.split_method.lower(),
                                                                            last_line_min_len=args.last_line_min_len,
                                                                            temperature=args.temperature,))))
