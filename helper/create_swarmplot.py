import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os

def create_swarmplot(tmp_file_name, fix_axis):
    with open(tmp_file_name) as f:
        data = json.load(f)
    model = data["model"]
    metric = data["metric"]
    metric_data = data["metric_data"]
    axes_min = data["axes_min"]
    axes_max = data["axes_max"]
    save_dir = data["save_dir"]

    df = pd.DataFrame(metric_data)
    plt.figure(figsize=(12, 6))
    plt.clf()
    ax = sns.swarmplot(data=df, x='position', y='score', 
                    hue='topic', dodge=False, size=1)

    _ = plt.legend(title='', loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=6, 
            fontsize=8, frameon=False, markerscale=5)
    
    plt.title(f"{model}-{metric}")

    if fix_axis:
        plt.ylim(axes_min, axes_max)

    plt.savefig(f'{save_dir}/{model}_{metric}.png',dpi=600,bbox_inches='tight', transparent=False,pad_inches=0)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('file_path', 
                    type=str, 
                    help="Path to the json file")

    parser.add_argument('fix_axis', 
                        type=str, 
                        help="Fix axis (True/False)", 
                        nargs="?", 
                        default="False")
    
    args = parser.parse_args()

    create_swarmplot(args.file_path, args.fix_axis.lower() == "true")
    os.unlink(args.file_path)