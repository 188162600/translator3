from datasets import inspect_dataset, load_dataset_builder
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import datasets

import argparse

from tqdm import tqdm
import shutil
import os
parser= argparse.ArgumentParser()

parser.add_argument("--src", type=str, help="source language", default="de")
parser.add_argument("--trg", type=str, help="target language", default="en")
parser.add_argument("--path", type=str, help="path to save the dataset",default="./datasets")
parser.add_argument("--dataset", type=str, help="dataset name", default="wmt14")
args = parser.parse_args()


inspect_dataset(args.dataset, "scripts")
path=args.path
builder = load_dataset_builder(
    "scripts/wmt_utils.py",
    language_pair=(args.src, args.trg),
    subsets={
        datasets.Split.TRAIN:["europarl_v7",
                "commoncrawl",
                "multiun",
                "newscommentary_v9",
                "gigafren",
                "czeng_10",
                "yandexcorpus",
                "wikiheadlines_hi",
                "wikiheadlines_ru",
                "hindencorp_01",],
        datasets.Split.VALIDATION: ["newsdev2014", "newstest2013"],
        datasets.Split.TEST: ["newstest2014"]
       # datasets.Split.VALIDATION: ["euelections_dev2019"],
    },

)
builder.download_and_prepare()
dataset=builder.as_dataset()

# dataset.save_to_disk(path)
names="train","test","validation"
save_names = ["train","test","valid"]
os.makedirs(path,exist_ok=True)
for name,save_name in zip( names,save_names):
    src_file = open(f"{path}/{save_name}.{args.src}", "w+",encoding="utf-8")
    trg_file = open(f"{path}/{save_name}.{args.trg}", "w+",encoding="utf-8")
    for src_trg in tqdm( dataset[name],desc=f"writing {name}"):
        src_file.write(src_trg["translation"][args.src] + "\n")
        trg_file.write(src_trg["translation"][args.trg] + "\n")
    src_file.close()
    trg_file.close()
