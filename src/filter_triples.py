import sys, os, json

f = open('dblp_filtered.tsv','w')

with open("dblp.nt", "r", encoding="utf-8") as file:
    for line in file:
        try:
            s,p,o,_ = line.split(" ")
            if 'https://dblp.org' not in s or 'https://dblp.org' not in o:
                continue
            f.write(f"{s}\t{p}\t{o}\n")
        except Exception as err:
            continue
f.close()
