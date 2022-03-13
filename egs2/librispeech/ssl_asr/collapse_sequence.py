import sys
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

def collapse(arr):
    if arr is None:
        return None
    if len(arr) < 2:
        return arr
    collapsed_arr = [arr[0]]
    key = arr[0]
    n = len(arr)
    i = 1
    while True:
        while i < n and arr[i] == key:
            i += 1
        if i == n:
            break
        key = arr[i]
        collapsed_arr.append(key)
    return collapsed_arr


def main(args):
    arglen = len(args)
    if arglen < 2:
        print('Invalid arguments. Correct usage: %s filepath' % (args[0]))
        exit(1)
    print('Collapsing sequences...')

    c = 0
    utt_ids = []
    seqs = []
    with open(args[1], "r") as f:
        for line in tqdm(f.readlines()):
            c += 1
            if line.strip() == "":
                print('Skipping empty line: %d' % (c))
                continue
            utt_id, seq = line.split(" ", 1)
            seq = np.fromstring(seq, dtype=int, sep=' ')
            utt_id = utt_id.strip()
            seq = collapse(seq)
            utt_ids.append(utt_id)
            seqs.append(seq)


    path = Path(args[1])
    parent_path = path.parent.absolute()
    new_file = os.path.join(parent_path, os.path.basename(args[1]) + "_collapsed")
    with open(new_file, "w") as f:
        for utt_id, seq in zip(utt_ids, seqs):
            f.write(utt_id + " " + " ".join(map(str, seq)) + "\n")

    print('Succesfully collapsed %d lines and dumped into %s' % (c, new_file))


if __name__=="__main__":
    main(sys.argv)
