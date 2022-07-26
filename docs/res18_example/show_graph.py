import argparse
import numpy as np
import parse

parser = argparse.ArgumentParser()
parser.add_argument("graph",type = str)
args = parser.parse_args()
#file_n = args.graph

with open(args.graph) as f:
    while True:
        line = f.readline()
        if not line: break
        if len(line) > 1000:
            a = line[:95]
            b = line[-75:]
            line = a +"...too many..." + b
            
        print(line)

f.close()



