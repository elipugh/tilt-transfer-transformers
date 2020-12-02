#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import logging
import os
import shutil
import sys
import inspect
from collections import Counter
from itertools import zip_longest
from multiprocessing import Pool
import argparse

from fairseq import options, tasks, utils
from fairseq.binarizer import Binarizer
from fairseq.data import indexed_dataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.normpath(os.path.join(currentdir, os.pardir))
parentdir = os.path.normpath(os.path.join(parentdir, os.pardir))
sys.path.insert(0, parentdir)
from corpora.data import Corpus
from corpora.data import Dictionary as cDictionary
from fairseq.data import Dictionary as fDictionary
import torch


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess_pkl")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", default=None, metavar="FP", help="corpus"
    )
    parser.add_argument(
        "--dir", default=None, metavar="FP", help="dir"
    )
    return parser


def main(args):
    utils.import_user_module(args)

    os.makedirs(args.dir, exist_ok=True)

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(args.dir, "preprocess_pkl.log"),
        )
    )
    logger.info(args)

    task = tasks.get_task("language_modeling")

    def dest_path(prefix):
        return os.path.join(args.dir, prefix)

    def dict_path(lang):
        return dest_path("dict") + ".txt"

    def build_dictionary(corpus):
        d = fDictionary()
        d.symbols = corpus.dictionary.idx2word
        d.count = [0 for _ in range(corpus.dictionary.total)]
        for k,v in corpus.dictionary.counter.items():
            d.count[k] = v
        d.indices = corpus.dictionary.word2idx
        d.finalize()
        return d

    corpus = torch.load(args.corpus)
    src_dict = build_dictionary(corpus)
    src_dict.save(os.path.join(args.dir, "dict.txt"))

    def make_binary_dataset(vocab, data, output_prefix):
        logger.info("Dictionary: {} types".format(len(vocab)))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, "bin"),
            impl=args.dataset_impl,
            vocab_size=len(vocab),
        )

        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t), offset=0, end=offsets[1]
            )
        )

        ds.finalize(dataset_dest_file(args, output_prefix, "idx"))

        logger.info(
            "{}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_all(vocab, corpus):
        make_binary_dataset(vocab, corpus.train, "train")
        make_binary_dataset(vocab, corpus.valid, "valid")
        make_binary_dataset(vocab, corpus.test, "test")

    make_all(src_dict, corpus)
    logger.info("Wrote preprocessed data to {}".format(args.destdir))


def dataset_dest_prefix(args, output_prefix):
    base = "{}/{}".format(args.dir, output_prefix)
    return base


def dataset_dest_file(args, output_prefix, extension):
    base = dataset_dest_prefix(args, output_prefix)
    return "{}.{}".format(base, extension)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
