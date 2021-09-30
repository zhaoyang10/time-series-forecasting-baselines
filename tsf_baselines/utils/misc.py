import numpy as np

def average_dict(dict_list):
    dict_average = {}

    for key,value in dict_list[0].items():
        dict_average[key] = 0

    num = len(dict_list)

    for dict_item in dict_list:
        for key, value in dict_item.items():
            dict_average[key] += value / num

    return dict_average
