#!/usr/local/bin/python3

import argparse
import tokenize
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="path to the data file")
parser.add_argument("vocab_path", type=str, help="path to store the vocabulary")
parser.add_argument("index_path", type=str, help="path to store the indices")
parser.add_argument("block_size", type=int, help="reading limit", default=2**18)
parser.add_argument("limit", type=int, help="reading limit", default=0, )

args = parser.parse_args()

# empty the vocab file
fout_vocab = open(args.vocab_path, "w")
fout_vocab.write("")
fout_vocab.close()

idxs = {}
output_seq = []

# read the data in a streaming fashion

num_blocks = 0
num_processed = 0
rest = []
fin = open(args.data_path, "r")
fout_vocab = open(args.vocab_path, "a")
while True:
    block = fin.read(args.block_size)

    num_blocks += 1

    print("\rprocessed blocks %d, tokens %d, word types %d" % (num_blocks, num_processed, len(idxs)), end="")

    if not block or (args.limit > 0 and num_processed > args.limit):
        break

    chunks = block.split(" ")
    if len(chunks) == 0:
        rest += chunks
    else:
        rest.append(chunks[0])
        tokens = []
        tokens += chunks[1:-1]
        if len(rest) > 0:
            tokens += ["".join(rest)]
        for token in tokens:
            h = hash(token)
            if not h in idxs:
                fout_vocab.write(token + "\n")
                idxs[h] = len(idxs)
            output_seq.append(idxs[h])
        num_processed += len(tokens)
        rest = chunks[-1:]

fin.close()
fout_vocab.close()

np.save(args.index_path, np.array(output_seq))
print("\ndone")

