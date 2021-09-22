import os.path
import re
from typing import List

import joblib
import numpy as np

from utils import clean_str

dataset = "enwiki"
dataset_path = f"/home/alessandroz/repo/BertGCN/data/raw/{dataset}/"
encoder_name = "encoder.bin.xz"


def parse_file(file: str, encoder):
    with open(dataset_path + file, mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    train_labels: List[str] = list()
    train_texts: List[str] = list()
    for line in lines:
        tagged_labels = re.findall(r"__label__\w+", line)
        text = line[len(" ".join(tagged_labels)) + 1:]
        labels = encoder.transform([[cat[len("__label__"):] for cat in tagged_labels]])[0]
        labels_str = np.array2string(labels, separator="\t", max_line_width=100000).strip("[").strip("]")
        train_labels.append(labels_str)
        train_texts.append(text)
    return train_labels, train_texts


def file_to_dataset():
    limit = 1000000000000000000  # just for debugging
    encoder = joblib.load(os.path.join(dataset_path, encoder_name))

    train_labels, train_texts = parse_file("train.txt", encoder)
    test_labels, test_texts = parse_file("test.txt", encoder)

    train_labels = train_labels[:limit]
    train_texts = train_texts[:limit]
    test_labels = test_labels[:limit]
    test_texts = test_texts[:limit]

    labels_str = [f"{i}\ttrain\t{labs}" for i, labs in enumerate(train_labels)]
    offset = len(labels_str)
    labels_str.extend([f"{i + offset}\ttest\t{labs}" for i, labs in enumerate(test_labels)])
    labels_str = "\n".join(labels_str)
    with open(f"data/{dataset}.txt", "w+", encoding="utf-8") as f:
        f.write(labels_str)

    texts_str = "\n".join([clean_str(t) for t in train_texts + test_texts])
    with open(f"data/corpus/{dataset}.clean.txt", "w+", encoding="utf-8") as f:
        f.write(texts_str)


if __name__ == "__main__":
    file_to_dataset()
