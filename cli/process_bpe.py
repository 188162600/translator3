import argparse
import os
import shutil
import codecs
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

# Parse arguments
parser = argparse.ArgumentParser(description="Script to apply BPE encoding on a dataset.")
parser.add_argument("--src", type=str, help="Source language.", default="de")
parser.add_argument("--trg", type=str, help="Target language.", default="en")
parser.add_argument("--path", type=str, help="Path to save the dataset.", required=True)
args = parser.parse_args()

# Construct file paths
bpe_train_path = os.path.join(args.path, f"train.{args.src}_{args.trg}")
code_path = os.path.join(args.path, "code")

# Function to concatenate source and target training files into a single file
def concatenate_files(input_files, output_file):
    with open(output_file, 'wb') as wfd:
        for input_file in input_files:
            with open(input_file, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)

# Concatenate the training files
concatenate_files([os.path.join(args.path, f"train.{args.src}"), os.path.join(args.path, f"train.{args.trg}")], bpe_train_path)

# Learn BPE and write to the code file
with codecs.open(bpe_train_path, "r", encoding='utf-8') as bpe_train_file, codecs.open(code_path, 'w+', encoding='utf-8') as code_file:
    learn_bpe(bpe_train_file, code_file, 40000,num_workers=20)

# Re-open the code file to read from the beginning
with codecs.open(code_path, 'r', encoding='utf-8') as code_file:
    bpe = BPE(code_file)
    # Apply BPE encoding to the train, test, and valid sets for both languages
    for lang in [args.src, args.trg]:
        for name in ["train", "test", "valid"]:
            input_path = os.path.join(args.path, f"{name}.{lang}")
            output_path = os.path.join(args.path, f"bpe.{name}.{lang}")
            with open(input_path, 'r', encoding='utf-8') as input_file, open(output_path, 'w', encoding='utf-8') as output_file:
                for line in input_file:
                    output_file.write(bpe.process_line(line.strip()) + '\n')
