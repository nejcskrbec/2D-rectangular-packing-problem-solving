import json

from shared.knapsack import *
from shared.item import *

def parse_problem_instance(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    knapsack = None
    if data.get("Objects") and len(data["Objects"]) > 0:
        obj = data["Objects"][0]
        knapsack = Knapsack(width=obj["Length"], height=obj["Height"])

    items = []
    for item in data.get("Items", []):
        parsed_item = Item(width=item["Length"], height=item["Height"])
        items.append(parsed_item)

    return knapsack, items