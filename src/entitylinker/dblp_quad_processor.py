import sys,os,json,copy
from dblp_kg_utils import OneHopFetcher

d = json.load(open('dblp_quad/questions_valid.json'))
o = OneHopFetcher("config.json")
citems = []
for iidx,item in enumerate(d['questions']):
    try:
        print(f"Processing item {iidx+1}/{len(d['questions'])}")
        citem = copy.deepcopy(item)
        citem['entityNeighbourhood'] = {}
        for entity in item['entities']:
            left,right = o.fetch_one_hop(entity[1:-1])
            print(entity)
            print("LEFT ------------")
            print(left.replace('\t',' ').split('\n')[1:])
            print("RIGHT ------------")
            print(right.replace('\t',' ').split('\n')[1:])
            print("======================")
            leftNodeNeighbourhood = left.replace('\t',' ').split('\n')[1:]
            rightNodeNeighbourhood = right.replace('\t',' ').split('\n')[1:]
            citem['entityNeighbourhood'][entity] = {
                'left': leftNodeNeighbourhood,
                'right': rightNodeNeighbourhood
            }
        citems.append(citem)
    except Exception as e:
        print(f"Error processing item {iidx+1}: {e}")
        continue
with open('dblp_quad/questions_valid_processed.json', 'w') as f:
    json.dump(citems, f, indent=4, ensure_ascii=False)   

    



