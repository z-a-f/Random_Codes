#!/usr/bin/env python

import pandas as pd
import tqdm

def xlsx_to_csv(fname):
  """
    Args:
      name: Name of the file to open
  """
  oname = '.'.join(fname.split('.')[:-1])+'.csv'
  with open(fname, 'r') as fin:
    data = pd.read_excel(fin, header = 0)
    with open(oname, 'w') as fout:
      data.to_csv(fout, sep=',', index = False)


if __name__ == '__main__':
  files = ['Arm', 'Belt', 'Pocket', 'Wrist']
  files = ['./data/'+x+'.xlsx' for x in files]
  for file in tqdm.tqdm(files):
    xlsx_to_csv(file)

