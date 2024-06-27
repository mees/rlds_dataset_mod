from abc import ABC, abstractmethod

import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import os
#from openai import OpenAI
import re
from typing import Any, Dict

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class TfdsModFunction(ABC):
    @classmethod
    @abstractmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        """
        Modifies the data builder feature dict to reflect feature changes of ModFunction.
        """
        ...

    @classmethod
    @abstractmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Perform arbitrary modifications on the dataset that comply with the modified feature definition.
        """
        ...


def mod_obs_features(features, obs_feature_mod_function):
    """Utility function to only modify keys in observation dict."""
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: obs_feature_mod_function(
                                key, features["steps"]["observation"][key]
                            )
                            for key in features["steps"]["observation"].keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )


# def extract_message(res):
#     """Parse message content from ChatGPT response."""
#     return res.choices[0].message.content
#
#
# def ask_chatgpt(messages):
#     """
#     Prompts ChatGPT with the list of messages.
#     Returns the response.
#     """
#     res = client.chat.completions.create(
#         model="gpt-3.5-turbo", messages=messages, temperature=0.3
#     )
#     res = extract_message(res)
#     return res
#
#
# class RelabelLanguage(TfdsModFunction):
#     prompt_paraphrase = (
#         "This is a command for a robot: %s. "
#         + "Can you paraphrase it into %d different versions? "
#         + "Be as diverse as possible without changing the meaning of the command. "
#         + "Number the results like 1. result, 2. result, etc"
#     )
#
#     negatives_prompt = (
#         "This is a command for a robot: %s. "
#         + "Can you replace the colors of objects in the command with different colors, "
#         + "replace spatial relations such as left and right and replace the object name if no color or spatial relation is present? "
#         + "Generate %d variants and number them like 1. result, 2. result, etc"
#     )
#     n_variants = 10
#
#     def parse_response(response):
#         pattern = r"\d+\.\s(.+)"  # Match any text following a digit and period
#         # Find all matches in the response
#         matches = re.findall(pattern, response)
#         return matches
#
#     @classmethod
#     def mod_features(
#         cls,
#         features: tfds.features.FeaturesDict,
#     ) -> tfds.features.FeaturesDict:
#         language_instruction_features = {
#             f"language_instruction_{label}_{i}": tfds.features.Text()
#             for label in ["relabel", "negative"]
#             for i in range(10)
#         }
#         return tfds.features.FeaturesDict(
#             {
#                 "steps": tfds.features.Dataset(
#                     {
#                         "observation": tfds.features.FeaturesDict(
#                             {
#                                 key: features["steps"]["observation"][key]
#                                 for key in features["steps"]["observation"].keys()
#                             }
#                         ),
#                         **{
#                             key: features["steps"][key]
#                             for key in features["steps"].keys()
#                             if key not in ("observation",)
#                         },
#                         **language_instruction_features,
#                     }
#                 ),
#                 **{
#                     key: features[key]
#                     for key in features.keys()
#                     if key not in ("steps",)
#                 },
#             }
#         )
#
#     @staticmethod
#     def return_language(dataset_name: str, trajectory: Dict[str, Any]):
#         dataset_instructions = {
#             "taco_play": "natural_language_instruction",
#             "bridge_dataset": "natural_language_instruction",
#             "fractal20220817_data": "natural_language_instruction",
#             "jaco_play": "natural_language_instruction",
#             "berkeley_autolab_ur5": "natural_language_instruction",
#             "language_table": "instruction",
#             "bc_z": "natural_language_instruction",
#             "furniture_bench_dataset_converted_externally_to_rlds": "language_instruction",
#             "ucsd_kitchen_dataset_converted_externally_to_rlds": "language_instruction",
#             "iamlab_cmu_pickup_insert_converted_externally_to_rlds": "language_instruction",
#             "berkeley_fanuc_manipulation": "language_instruction",
#             "cmu_stretch": "language_instruction",
#         }
#
#         if dataset_name in dataset_instructions:
#             if dataset_name == "language_table":
#                 # decode language instruction
#                 instruction_bytes = trajectory["observation"][
#                     dataset_instructions[dataset_name]
#                 ]
#                 instruction_encoded = tf.strings.unicode_encode(
#                     instruction_bytes, output_encoding="UTF-8"
#                 )
#                 # Remove trailing padding --> convert RaggedTensor to regular Tensor.
#                 language_instruction = tf.strings.split(instruction_encoded, "\x00")[
#                     :, :1
#                 ].to_tensor()[:, 0]
#                 return language_instruction
#             elif (
#                 (
#                     dataset_name
#                     == "furniture_bench_dataset_converted_externally_to_rlds"
#                     or dataset_name
#                     == "ucsd_kitchen_dataset_converted_externally_to_rlds"
#                 )
#                 or (
#                     dataset_name
#                     == "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
#                     or dataset_name == "berkeley_fanuc_manipulation"
#                 )
#                 or dataset_name == "cmu_stretch"
#             ):
#                 return trajectory["step"][dataset_instructions[dataset_name]]
#             else:
#                 return trajectory["observation"][dataset_instructions[dataset_name]]
#         else:
#             # Handle unknown dataset_name here
#             return None  # or raise an exception, depending on your needs
#
#     def get_prompt(variant_type, original_language):
#         prompt_format = (
#             RelabelLanguage.prompt_paraphrase
#             if variant_type == "language_instruction_relabel"
#             else RelabelLanguage.negatives_prompt
#         )
#         return prompt_format % (original_language, RelabelLanguage.n_variants)
#
#     @classmethod
#     def mod_dataset(cls, ds: tf.data.Dataset, dataset_name: str) -> tf.data.Dataset:
#         cls.dataset_name = dataset_name
#
#         def relabel_language(step):
#             original_language = RelabelLanguage.return_language(cls.dataset_name, step)
#             # print("ORIGINAL:", original_language)
#             process_variants(step, original_language, "language_instruction_relabel")
#             process_variants(step, original_language, "language_instruction_negative")
#             return step
#
#         def process_variants(step, original_language, variant_type):
#             prompt = RelabelLanguage.get_prompt(variant_type, original_language)
#             messages = [{"role": "user", "content": prompt}]
#             res = ask_chatgpt(messages)
#             parsed_response = RelabelLanguage.parse_response(res)
#             for i, variant in enumerate(parsed_response):
#                 step["observation"][f"{variant_type}_{i}"] = variant
#             return step
#
#         def episode_map_fn(episode):
#             episode["steps"] = episode["steps"].map(relabel_language)
#             return episode
#
#         return ds.map(episode_map_fn)

class VisualTrajectory(TfdsModFunction):
    gripper_pos_lookup = json.load(open("/home/oiermees/bridge_labeled_dataset_1.json","r"))
    # keys = list(gripper_pos_lookup.keys())
    # values = list(gripper_pos_lookup.keys())
    # initializer = tf.lookup.KeyValueTensorInitializer(
    #      keys, values, key_dtype=tf.string, value_dtype=tf.string
    # )
    # hash_table = tf.lookup.StaticHashTable(initializer, default_value="")

    TRAJECTORY_IMAGE_SHAPE = (256, 256, 3)
    @classmethod
    def mod_features(cls, features: tfds.features.FeaturesDict) -> tfds.features.FeaturesDict:
        # Adding the new field for visual trajectory
        return features
        # visual_trajectory = tfds.features.Image(shape=VisualTrajectory.TRAJECTORY_IMAGE_SHAPE, dtype=tf.uint8)
        # return tfds.features.FeaturesDict(
        #                 {
        #                     "steps": tfds.features.Dataset(
        #                         {
        #                             "observation": tfds.features.FeaturesDict(
        #                                 {
        #                                     key: features["steps"]["observation"][key]
        #                                     for key in features["steps"]["observation"].keys()
        #                                 }
        #                             ),
        #                             **{
        #                                 key: features["steps"][key]
        #                                 for key in features["steps"].keys()
        #                                 if key not in ("observation",)
        #                             },
        #                             **visual_trajectory,
        #                         }
        #                     ),
        #                     **{
        #                         key: features[key]
        #                         for key in features.keys()
        #                         if key not in ("steps",)
        #                     },
        #                 }
        #             )

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def create_visual_trajectory(steps):
            # Example function to create a visual trajectory from steps
            # This should be replaced with actual logic to create the trajectory image
            trajectory_image = tf.zeros(VisualTrajectory.TRAJECTORY_IMAGE_SHAPE, dtype=tf.uint8)
            return trajectory_image

        # @tf.py_function(Tout=tf.data.Dataset)
        def episode_map_fn(episode):
            print(type(episode))
            print("eagerly1: ", tf.executing_eagerly())
            # tf.config.run_functions_eagerly(True)
            # print("eagerly2: ", tf.executing_eagerly())
            # print(episode.keys())
            # print(episode['episode_metadata'])
            # print(episode['episode_metadata'].keys())
            # Convert tensor to numpy for hashable comparison
            # file_path = episode['episode_metadata']['file_path'].numpy()

            # Accessing symbolic tensor values using tf.py_function
            # def access_tensor_value(tensor):
            #     return tensor.numpy()
            # file_path = tf.py_function(access_tensor_value, [episode['episode_metadata']['file_path']], tf.string)
            file_path = tf.get_static_value(episode['episode_metadata']['file_path'])
            # print(type(file_path))
            print(file_path)
            print(episode['episode_metadata']['file_path'])
            # my_value = VisualTrajectory.hash_table.lookup(episode['episode_metadata']['file_path'])
            # print(my_value)
            # print(my_value.keys())
            # print(my_value['0'])
            # if file_path in VisualTrajectory.gripper_pos_lookup:
            #     print("Episode found")
            # else:
            #     print("Episode not found")
            # exit()
            # visual_trajectory = create_visual_trajectory(episode['steps'])
            # episode['visual_trajectory'] = visual_trajectory
            return episode

        return ds.map(episode_map_fn)

class ResizeAndJpegEncode(TfdsModFunction):
    MAX_RES: int = 256

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def downsize_and_jpeg(key, feat):
            """Downsizes image features, encodes as jpeg."""
            if (
                len(feat.shape) >= 2 and feat.shape[0] >= 64 and feat.shape[1] >= 64
            ):  # is image / depth feature
                should_jpeg_encode = (
                    isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (
                        ResizeAndJpegEncode.MAX_RES,
                        ResizeAndJpegEncode.MAX_RES,
                        feat.shape[2],
                    )
                else:
                    new_shape = (
                        ResizeAndJpegEncode.MAX_RES,
                        ResizeAndJpegEncode.MAX_RES,
                    )

                if isinstance(feat, tfds.features.Image):
                    return tfds.features.Image(
                        shape=new_shape,
                        dtype=feat.dtype,
                        encoding_format="jpeg" if should_jpeg_encode else "png",
                        doc=feat.doc,
                    )
                else:
                    return tfds.features.Tensor(
                        shape=new_shape,
                        dtype=feat.dtype,
                        doc=feat.doc,
                    )

            return feat

        return mod_obs_features(features, downsize_and_jpeg)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def resize_image_fn(step):
            # resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) >= 2 and (
                    step["observation"][key].shape[0] >= 64
                    or step["observation"][key].shape[1] >= 64
                ):
                    size = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES)
                    if "depth" in key:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_depth_image(
                                tf.cast(step["observation"][key], tf.float32), size
                            ),
                            step["observation"][key].dtype,
                        )
                    else:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_image(step["observation"][key], size),
                            tf.uint8,
                        )
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(resize_image_fn)
            return episode

        return ds.map(episode_map_fn)


class FilterSuccess(TfdsModFunction):
    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(lambda e: e["success"])


class FlipImgChannels(TfdsModFunction):
    FLIP_KEYS = ["image"]

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def flip(step):
            for key in cls.FLIP_KEYS:
                if key in step["observation"]:
                    step["observation"][key] = step["observation"][key][..., ::-1]
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(flip)
            return episode

        return ds.map(episode_map_fn)


class FlipWristImgChannels(FlipImgChannels):
    FLIP_KEYS = ["wrist_image", "hand_image"]


TFDS_MOD_FUNCTIONS = {
    #"relabel_language": RelabelLanguage,
    "visual_trajectory": VisualTrajectory,
    "resize_and_jpeg_encode": ResizeAndJpegEncode,
    "filter_success": FilterSuccess,
    "flip_image_channels": FlipImgChannels,
    "flip_wrist_image_channels": FlipWristImgChannels,
}
