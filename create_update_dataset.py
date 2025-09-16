import argparse
import weave
import json
from string import ascii_letters, digits
from itertools import permutations
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Creator argument parser')

    parser.add_argument('file_path', 
                        type=str, 
                        help="Path to the json file where the dataset is specified (./datasets/xyz.json)")
    parser.add_argument('-p','--project_name', 
                        type=str, 
                        required=True, 
                        help="wandb project name in which the dataset should be created/updated")
    parser.add_argument('--no-permutations', 
                        action="store_true", 
                        help="if the list in user_prompts should NOT be permutated (e.g. golden descriptions)")
    parser.add_argument("-U",'--update', 
                        action="store_true", 
                        help="update an existing dataset")
    parser.add_argument('-e','--entity', 
                        type=str, 
                        default="highly-biased", 
                        help="optional, wandb team name, defaults to 'highly-biased'")
    args = parser.parse_args()

    with open(args.file_path) as f:
        data = json.load(f)

        missing_keys = [key for key in ["name", "system_prompt", "user_prompt_items"] if key not in data.keys()]
        if len(missing_keys) > 0:
            raise KeyError(f"Missing keys [\"{'\", \"'.join(missing_keys)}\"] in JSON file")
        name = data["name"]

        if set(name).difference(ascii_letters + digits + "_-"):
            raise ValueError("Avoid special characters in dataset name! Please only use ASCII letters, digits and '_' or '-'")
        
        dataset_rows = []
        if not args.no_permutations:
            for i, user_prompt_perm in tqdm(enumerate(permutations(data["user_prompt_items"])), desc="Generating permutations"):
                row = {
                        'id':str(i),
                        'system_prompt': data["system_prompt"],
                        'user_prompt': ", ".join(user_prompt_perm),
                    }
                
                if "addtional_data" in data.keys():
                    for k,v in data["addtional_data"].items():
                        row[k] = v

                dataset_rows.append(
                    row
                )
        else:
            for i, user_prompt in tqdm(enumerate(data["user_prompt_items"]), desc="Generating dataset"):
                row = {
                            'id':str(i),
                            'system_prompt': data["system_prompt"],
                            'user_prompt': user_prompt,
                        }
                    
                if "addtional_data" in data.keys():
                    for k,v in data["addtional_data"].items():
                        row[k] = v

                dataset_rows.append(
                    row
                )

        weave.init(f"{args.entity}/{args.project_name}")
        try:
            existing_dataset  = weave.ref(name).get()
        except ValueError:
            existing_dataset = None

        if args.update and existing_dataset is None:
            raise ValueError(f"Dataset '{name}' in project '{args.project_name}' does not exists! Execute the command without '-U' to create a new dataset.")
        elif not args.update and existing_dataset is not None:
            raise ValueError(f"Dataset '{name}' in project '{args.project_name}' already exists! Execute the command with '-U' to update the existing dataset.")


        dataset = weave.Dataset(
            name=name,
            rows=dataset_rows
        )

        weave.publish(dataset)


