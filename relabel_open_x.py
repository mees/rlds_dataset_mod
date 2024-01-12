import tensorflow_datasets as tfds
import tensorflow as tf
import os
from openai import OpenAI
import re
from typing import Any, Dict

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def return_language(dataset_name: str, trajectory: Dict[str, Any]):
    dataset_instructions = {
        "taco_play": "natural_language_instruction",
        "bridge": "natural_language_instruction",
        "fractal20220817_data": "natural_language_instruction",
        "jaco_play": "natural_language_instruction",
        "berkeley_autolab_ur5": "natural_language_instruction",
        "language_table": "instruction",
        "bc_z": "natural_language_instruction",
        "furniture_bench_dataset_converted_externally_to_rlds": "language_instruction",
        "ucsd_kitchen_dataset_converted_externally_to_rlds": "language_instruction",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": "language_instruction",
        "berkeley_fanuc_manipulation": "language_instruction",
        "cmu_stretch": "language_instruction",
    }

    if dataset_name in dataset_instructions:
        if dataset_name == "language_table":
            # decode language instruction
            instruction_bytes = trajectory["observation"][
                dataset_instructions[dataset_name]
            ]
            instruction_encoded = tf.strings.unicode_encode(
                instruction_bytes, output_encoding="UTF-8"
            )
            # Remove trailing padding --> convert RaggedTensor to regular Tensor.
            language_instruction = tf.strings.split(instruction_encoded, "\x00")[:1][0]
            return language_instruction
        elif (
                (
                        dataset_name
                        == "furniture_bench_dataset_converted_externally_to_rlds"
                        or dataset_name
                        == "ucsd_kitchen_dataset_converted_externally_to_rlds"
                )
                or (
                        dataset_name
                        == "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
                        or dataset_name == "berkeley_fanuc_manipulation"
                )
                or dataset_name == "cmu_stretch"
        ):
            return trajectory[dataset_instructions[dataset_name]]
        else:
            return trajectory["observation"][dataset_instructions[dataset_name]]
    else:
        # Handle unknown dataset_name here
        return None  # or raise an exception, depending on your needs


datasets = (
    "taco_play 0.1.0",
    # "fractal20220817_data 0.1.0",
    # "bridge 0.1.0",
    # "jaco_play 0.1.0",
    # "berkeley_autolab_ur5 0.1.0",
    # "language_table 0.1.0",
    # "furniture_bench_dataset_converted_externally_to_rlds 0.1.0",
    # "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0",
    # "bc_z 1.0.0",
    # "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0",
    # "berkeley_fanuc_manipulation 0.1.0",
    # "cmu_stretch 0.1.0",
)
unique_strings = set()
for dataset in datasets:
    dataset, version = dataset.split(" ")
    print("dataset:", dataset)
    # create RLDS dataset builder
    if dataset == "bc_z":
        print("using bc_z")
        ds = tfds.load("bc_z", data_dir="gs://rail-orca-central2/resize_256_256", split="train[:10]",
                       decoders={"steps": tfds.decode.SkipDecoding()})
    else:
        ds = tfds.load(dataset, data_dir="gs://gresearch/robotics/", split="train[:10]",
                       decoders={"steps": tfds.decode.SkipDecoding()})
    for element in iter(ds):
        original_language_tensor = return_language(dataset, element["steps"])
        print(original_language_tensor)
        original_language = original_language_tensor[0].numpy().decode()
        print(original_language)
        assert isinstance(original_language, str), "original_language should be a string"
        unique_strings.add(original_language)
        print(unique_strings)
        # break

print("unique_strings:", unique_strings)

prompt_paraphrase = (
        "This is a command for a robot: %s. "
        + "Can you paraphrase it into %d different versions? "
        + "Be as diverse as possible without changing the meaning of the command. "
        + "Number the results like 1. result, 2. result, etc"
    )

negatives_prompt = (
    "This is a command for a robot: %s. "
    + "Can you replace the colors of objects in the command with different colors, "
    + "replace spatial relations such as left and right and replace the object name if no color or spatial relation is present? "
    + "Generate %d variants and number them like 1. result, 2. result, etc"
)
n_variants = 10

def parse_response(response):
    pattern = r"\d+\.\s(.+)"  # Match any text following a digit and period
    # Find all matches in the response
    matches = re.findall(pattern, response)
    return matches
def extract_message(res):
    """Parse message content from ChatGPT response."""
    return res.choices[0].message.content

def ask_chatgpt(messages):
    """
    Prompts ChatGPT with the list of messages.
    Returns the response.
    """
    res = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.3
    )
    res = extract_message(res)
    return res

def get_prompt(variant_type, original_language):
    prompt_format = (
        prompt_paraphrase
        if variant_type == "language_instruction_relabel"
        else negatives_prompt
    )
    return prompt_format % (original_language, n_variants)
def process_variants(original_language, variant_type):
    prompt = get_prompt(variant_type, original_language)
    messages = [{"role": "user", "content": prompt}]
    res = ask_chatgpt(messages)
    parsed_response = parse_response(res)
    return parsed_response

k = 5
lang_augmented_dict = {}
for lang in unique_strings:
    print("lang:", lang)
    paraphrases = []
    negatives = []
    for i in range(k):
        print("i:", i)
        paraphrase = process_variants(lang, "language_instruction_relabel")
        paraphrases.append(paraphrase)
        negative = process_variants(lang, "language_instruction_negative")
        negatives.append(negative)
    lang_augmented_dict[lang] = {"paraphrases": paraphrases, "negatives": negatives}


    # break

