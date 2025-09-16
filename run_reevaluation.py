import argparse
import asyncio
import weave
import weave.trace
import weave.trace.vals
from models.reevaluate import model_reevaluate_library
from scorer.library import scorer_library
from models import model_library
from helper.batch_loading import custom_remote_iter
from collections import defaultdict
from tqdm import tqdm
import os
from requests.auth import HTTPBasicAuth
import json
from weave.trace.env import _wandb_api_key_via_netrc
from weave.trace_server.requests import post as weave_post
import wandb
import time
import multiprocessing

import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Creator argument parser')

    parser.add_argument('-p','--project_name', 
                        type=str, 
                        required=True, 
                        help="wandb project name in which the evaluation will be logged in")
    parser.add_argument('-op','--origin_project_name', 
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
                        help=f"model name which will be evaluated. available models: '{'\', \''.join(model_reevaluate_library.keys())}'")
    parser.add_argument('-d','--dataset', 
                        type=str, 
                        required=True,
                        help="name of dataset in weave, optional with version in the format 'dataset_name:v0'")
    parser.add_argument('-s','--scorer', 
                        type=str, 
                        required=True,
                        help=f"Scorer, comma-seperated list of Scorers or 'all', to be used for evaluating the model output. available scorers: '{'\', \''.join(scorer_library.keys())}'")
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

    if model not in model_reevaluate_library.keys():
        raise ValueError(f"'{args.model}' is not a valid model. Available Models: '{'\', \''.join(model_library.keys())}'")
    
    if model.startswith("gpt"):
        from run_gpt_evaluation import models as model_gpt

        if model not in model_gpt.keys():
            raise ValueError(f"'{args.model}' is not a valid GPT model. Available Models: '{'\', \''.join(model_gpt.keys())}'")
    
        model_library[model] = model_gpt[model]["class"]

    require_golden = False
    using_rouge = False

    important_scorers="strlen,kolmogorov_complexity,kolmogorov_complexity_normalized,missing_topic,flesh_reading_ease,named_entity_density,wrong_topic_order"

    if args.scorer.lower() == "all":
        scorers = scorer_library.keys()
        require_golden = True
        using_rouge = True
    else:
        scorers = args.scorer.split(',')
        available_scorers = scorer_library.keys()
        invalid_scorers = []
        for scorer in scorers:
            if scorer not in available_scorers:
                invalid_scorers.append(scorer)
            elif scorer.startswith("golden_description"):
                require_golden = True
                if scorer == 'golden_description_rouge':
                    using_rouge = True
        if len(invalid_scorers) > 0:
            raise ValueError(f"'{'\', \''.join(invalid_scorers)}' is/are not a valid scorer(s). Available Scorers: '{'\', \''.join(scorer_library.keys())}'")

    if args.split_method.lower() not in ("cosine", "first"):
         raise ValueError(f"'{args.split_method}' is not a valid split method. Available methods: 'cosine' and 'first'")


    # overwrite weave method to enable faster dataset loading
    weave.trace.vals.WeaveTable._remote_iter = custom_remote_iter(page_size=args.page_size)

    golden_descritions = defaultdict(list)
    if require_golden:

        client = weave.init(f"{args.entity}/golden_descriptions")

        dataset = weave.ref("set1_test_golden:v0").get()
        dataset_lookup_map = dict()
        for row in dataset.rows:
            id = row.ref._extra[-1]
            user_prompt = row["user_prompt"]
            dataset_lookup_map[id] = user_prompt

        calls = client.get_calls(
            filter={"trace_roots_only": True},
                query={"$expr":{"$contains":{"input":{"$getField":"inputs.model"},"substr":{"$literal":model_library[model].__name__+":"}}}},
            sort_by=[{"field":"started_at","direction":"desc"}],
        )

        calls = client.get_calls(
            filter={"call_ids":[call.id for call in calls]}
        )
        calls = list(calls)

        # if we get a lot of calls we got all objects in the project
        # this means the model could not be found
        # so we check if it maybe is a Reevaluate model
        if len(calls) > 100:
            calls = client.get_calls(
                filter={"trace_roots_only": True},
                    query={"$expr":{"$contains":{"input":{"$getField":"inputs.model"},"substr":{"$literal":model_library[model].__name__.replace("Model","")+"Reevaluate" +":"}}}},
                sort_by=[{"field":"started_at","direction":"desc"}],
            )

            calls = client.get_calls(
                filter={"call_ids":[call.id for call in calls]}
            )
            calls = list(calls)

        if len(calls) > 1:
            raise RuntimeError(f"Found multiple Calls for model '{model}'. Please make sure all golden descirptions are contained in one trace")
        
        call_id = calls[0].id

        payload = {
            "project_id":f"{args.entity}/golden_descriptions",
            "filter":{"parent_ids":[call_id],"trace_roots_only":False},
            "limit":3000,
            "offset":0,
            "sort_by":[{"field":"started_at","direction":"desc"}],
            "include_feedback": False,
            "expand_columns":["inputs.example.user_prompt"]
        }

        response = weave_post("https://trace.wandb.ai/calls/stream_query", 
                                    headers={
                                        'Content-Type': 'application/json',
                                        'Accept': 'application/json'
                                        }, 
                                    json=payload,
                                    auth=HTTPBasicAuth("api", _wandb_api_key_via_netrc()))

        for child in tqdm(response.json(), "Loading golden descriptions"):
            if 'Evaluation.summarize' in child["op_name"]:
                continue
            user_prompt_id = child["inputs"]["example"].rsplit("/", 1)[-1]
            topic = dataset_lookup_map[user_prompt_id]
            text = child["output"]["output"]["split"][topic]["text"]
            # remove md formating
            text = text.replace("*","").replace("#","").replace("__","").strip()
            golden_descritions[topic].append(text)


    # load existing data
    client = weave.init(f"{args.entity}/{args.origin_project_name}")

    dataset = weave.ref(args.dataset).get()

    dataset_lookup_map = dict()
    for row in dataset.rows:
        id = row.ref._extra[-1]
        user_prompt = row["user_prompt"]
        dataset_lookup_map[id] = user_prompt

    calls = client.get_calls(
            filter={"trace_roots_only": True},
            query={"$expr":{"$contains":{"input":{"$getField":"inputs.model"},"substr":{"$literal":model_library[model].__name__+":"}}}},
            sort_by=[{"field":"started_at","direction":"desc"}],
        )
    calls = client.get_calls(
        filter={"call_ids":[call.id for call in calls]}
    )
    calls = list(calls)

    trials = 0
    existing_responses = defaultdict(list)
    existing_splits = defaultdict(list)

    for call in tqdm(calls, "Loading model responses"):
        trials+=call.inputs["self"].trials

        call_id = call.id

        payload = {
            "project_id":f"{args.entity}/{args.origin_project_name}",
            "filter":{"parent_ids":[call_id],"trace_roots_only":False},
            "limit":3000,
            "offset":0,
            "sort_by":[{"field":"started_at","direction":"desc"}],
            "include_feedback": False,
            "expand_columns":["inputs.example.user_prompt"]
        }

        response = weave_post("https://trace.wandb.ai/calls/stream_query", 
                                    headers={
                                        'Content-Type': 'application/json',
                                        'Accept': 'application/json'
                                        }, 
                                    json=payload,
                                    auth=HTTPBasicAuth("api", _wandb_api_key_via_netrc()))

        for child in tqdm(response.json(), "Loading existing model outputs", leave=False):
            if 'Evaluation.summarize' in child["op_name"]:
                continue
            user_prompt_id = child["inputs"]["example"].rsplit("/", 1)[-1]
            user_prompt = dataset_lookup_map[user_prompt_id]
            existing_responses[user_prompt].append(child["output"]["output"]["raw"])
            if using_rouge:
                existing_splits[user_prompt].append(child["output"]["output"]["split"])


    print("Saving temp json file")
    start = time.time()
    with open("./temp-reevaluate-responses.json", "w") as f:
        json.dump(existing_responses, f)
    print(f"temp json file saved. Took: {time.time()-start:.2f}s")

    # precompute rouge for golden descriptions in parallel
    if using_rouge:
        from rouge_metric_multithreading import * 
        print("Precomputing ROUGE (this can take a while)")
        start = time.time()
        rouge_dict = start_precomputing_rouge(existing_splits, golden_descritions, chunk_size=len(dataset_lookup_map)//multiprocessing.cpu_count())
        print(f"Done precomputing ROUGE, took {time.time()-start}s")
        with open("./temp-reevaluate-rouge.json", "w") as f:
            json.dump(rouge_dict, f)


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
            if scorer_name == "golden_description_rouge":
                scorer = scorer_library[scorer_name](golden_descritions="./temp-reevaluate-rouge.json")
            else:
                scorer = scorer_library[scorer_name](golden_descritions=golden_descritions)
        else:
            scorer = scorer_library[scorer_name]()
        
        scorers_obj.append(scorer)

    

    evaluation = weave.Evaluation(
        name=name,
        evaluation_name=name, 
        dataset=dataset_ref,
        scorers=scorers_obj,
        trials=trials
    )
    print(asyncio.run(evaluation.evaluate(model_reevaluate_library[model](max_retries=args.max_retries,
                                                                            retry_wait_time=args.retry_wait_sec,
                                                                            previous_responses_path="./temp-reevaluate-responses.json",
                                                                            split_method=args.split_method.lower(),
                                                                            last_line_min_len=args.last_line_min_len
                                                                            ))))


    os.remove("./temp-reevaluate-responses.json")
    if os.path.exists("./temp-reevaluate-rouge.json"):
        os.remove("./temp-reevaluate-rouge.json")

    wandb.finish()