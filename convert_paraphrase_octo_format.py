import pickle

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

with open("augmented_language_labels_aloha", 'rb') as file:
  aloha_para = pickle.load(file)


new_dict={}

for k, v in aloha_para.items():
    my_list = v['paraphrases']
    flattened_list = flatten(my_list)
    unique_elem = list(set(flattened_list))
    new_dict[k] = ''.join(unique_elem)

with open("augmented_language_labels_aloha_octo_format", 'wb') as f:
    pickle.dump(new_dict, f)