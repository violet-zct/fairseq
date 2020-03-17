import argparse
import sys
import re
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import scipy.special
import tqdm

line_re = re.compile(r'([STHP])-([0-9]+)\t(.*)')

def read_file(fname):
  with open(fname, 'r') as f:
    for lines in itertools.zip_longest(*[f] * 6):
      lines = [x.strip() for x in lines]
      tori = lines[0].split('\t')[1]
      tbpe = lines[1].split('\t')[1]
      hscore, hbpe = lines[2].split('\t')[1:]
      _, hori = lines[3].split('\t')[1:]
      codes = lines[4].split('\t')[1]
      tori, tbpe, hbpe, hori, codes = [x.split(' ') for x in (tori, tbpe, hbpe, hori, codes)]
      codes = [x.split('-')[1] for x in codes]
      yield (tori, tbpe, hori, hbpe, float(hscore), codes)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("--dec_file", type=str, required=True, help="A decoding file")

  args = parser.parse_args()
  print(args)

  hyps = list(read_file(args.dec_file))

  lens = np.zeros( (5, len(hyps)) )
  for i, (tori, tbpe, hori, hbpe, hscore, codes) in enumerate(hyps):
    print(len(tbpe), len(hbpe), len(codes))
    lens[:,i] = np.array([len(x) for x in (tori, tbpe, hori, hbpe, codes)])
  len_diffs = lens[3]-lens[1]
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.set_xlabel('len(ref_bpe)')
  ax1.set_ylabel('len(hyp_bpe) - len(ref_bpe)')
  ax1.scatter(lens[1], len_diffs)
  plt.show()
