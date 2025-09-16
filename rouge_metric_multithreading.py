import multiprocessing
from multiprocessing.pool import ThreadPool
import subprocess
import tempfile
import json
import os, sys
import itertools
import evaluate
import argparse
from collections import defaultdict
from tqdm import tqdm

def start_precomputing_subprocess(samples_path, golden_desc_path):
    p = subprocess.Popen([sys.executable,'rouge_metric_multithreading.py',samples_path,golden_desc_path])
    p.wait()

def start_precomputing_rouge(samples, golden_desc, chunk_size=200):

    tmp_golden_desc = tempfile.NamedTemporaryFile(delete=False, mode="w+")
    json.dump(golden_desc, tmp_golden_desc)
    tmp_golden_desc.flush()
    tmp_golden_desc.close()

    sample_items = list(samples.items())
    # create chunks from dict of samples and store them in temp files
    sample_temp_files = []
    for i in range(0,len(sample_items),chunk_size):
        chunk_items = dict(sample_items[i:i+chunk_size])
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w+")
        json.dump(chunk_items, tmp)
        tmp.flush()
        sample_temp_files.append(tmp.name)
        tmp.close()

    with ThreadPool(processes=multiprocessing.cpu_count()) as pool:
        result = pool.starmap_async(start_precomputing_subprocess, list(zip(sample_temp_files, itertools.repeat(tmp_golden_desc.name))))
        result.wait()

    rouge_per_user_prompt_dict = dict()

    for p in sample_temp_files:
        with open(p) as f:
            rouge_per_user_prompt_dict.update(json.load(f))
        # delete temp file
        os.unlink(p)

    # delete temp file
    os.unlink(tmp_golden_desc.name)

    return rouge_per_user_prompt_dict

def calculate_rouge(samples, golden_desc):
    rouge = evaluate.load("rouge", keep_in_memory=True)

    rouge_results = {}
    for user_prompt, split_list in tqdm(samples.items()):
        rouge_values = []
        for split in split_list:
            split_values = defaultdict(dict)
            for topic, data in split.items():
                split_values[topic]["position"] = data["position"]
                if data["text"] is None:
                    continue
                res = rouge.compute(predictions=list(itertools.repeat(data["text"], len(golden_desc[topic]))), references=golden_desc[topic])
                res = {k:v.item() for k,v in res.items()}
                split_values[topic]["score"] = res

            rouge_values.append(split_values)
        rouge_results[user_prompt] = rouge_values

    return rouge_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('sample_file', 
                        type=str, 
                        help="Path to the json file containing model samples")
    parser.add_argument('golden_desc_file', 
                        type=str, 
                        help="Path to the json file containing golden descriptions")
    
    args = parser.parse_args()

    with open(args.sample_file) as f:
        samples = json.load(f)
    
    with open(args.golden_desc_file) as f:
        golden_desc = json.load(f)

    results = calculate_rouge(samples, golden_desc)

    # overwrite data in file so it can be loaded by the main process
    with open(args.sample_file, "w") as f:
        json.dump(results, f)
        f.flush()