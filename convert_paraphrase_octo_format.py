import pickle

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

with open("aloha_play_rephrasings", 'rb') as file:
  aloha_para = pickle.load(file)


new_dict={}

for k, v in aloha_para.items():
    my_list = v
    flattened_list = flatten(my_list)
    unique_elem = list(set(flattened_list))
    new_dict[k] = ''.join(unique_elem)

with open("aloha_play_rephrasings_octo_format", 'wb') as f:
    pickle.dump(new_dict, f)