from abc import ABC, abstractmethod

import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from openai import OpenAI
import re

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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


def extract_message(res):
    """Parse message content from ChatGPT response."""
    return res.choices[0].message.content


def ask_chatgpt(messages):
    """
    Prompts ChatGPT with the list of messages.
    Returns the response.
    """
    res = client.chat.completions.create(model="gpt-3.5-turbo",
                                         messages=messages,
                                         temperature=0.3)
    res = extract_message(res)
    return res


class RelabelLanguage(TfdsModFunction):
    paraphrase_base_prompt = "This is a command for a robot: %s. " \
                             + "Can you paraphrase it into %d different versions? " \
                             + "Be as diverse as possible without changing the meaning of the command. " \
                             + "Number the results like 1. result, 2. result, etc"

    negatives_prompt = "This is a command for a robot: %s. " \
                       + "Can you replace the colors of objects in the command with different colors, " \
                       + "replace spatial relations such as left and right and replace the object name if no color or spatial relation is present? " \
                       + "Generate %d variants and number them like 1. result, 2. result, etc"
    n_variants = 10

    prompt = paraphrase_base_prompt

    def parse_response(response):
        pattern = r'\d+\.\s(.+)'  # Match any text following a digit and period
        # Find all matches in the response
        matches = re.findall(pattern, response)
        return matches

    @classmethod
    def mod_features(
            cls,
            features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return tfds.features.FeaturesDict(
            {
                "steps": tfds.features.Dataset(
                    {
                        "observation": tfds.features.FeaturesDict(
                            {
                                key: features["steps"]["observation"][key]
                                for key in features["steps"]["observation"].keys()
                            }
                        ),
                        **{
                            key: features["steps"][key]
                            for key in features["steps"].keys()
                            if key not in ("observation",)
                        },
                        "language_instruction_relabel_0": tfds.features.Text(),
                        "language_instruction_relabel_1": tfds.features.Text(),
                        "language_instruction_relabel_2": tfds.features.Text(),
                        "language_instruction_relabel_3": tfds.features.Text(),
                        "language_instruction_relabel_4": tfds.features.Text(),
                        "language_instruction_relabel_5": tfds.features.Text(),
                        "language_instruction_relabel_6": tfds.features.Text(),
                        "language_instruction_relabel_7": tfds.features.Text(),
                        "language_instruction_relabel_8": tfds.features.Text(),
                        "language_instruction_relabel_9": tfds.features.Text(),
                        "language_instruction_negative_0": tfds.features.Text(),
                        "language_instruction_negative_1": tfds.features.Text(),
                        "language_instruction_negative_2": tfds.features.Text(),
                        "language_instruction_negative_3": tfds.features.Text(),
                        "language_instruction_negative_4": tfds.features.Text(),
                        "language_instruction_negative_5": tfds.features.Text(),
                        "language_instruction_negative_6": tfds.features.Text(),
                        "language_instruction_negative_7": tfds.features.Text(),
                        "language_instruction_negative_8": tfds.features.Text(),
                        "language_instruction_negative_9": tfds.features.Text(),
                    }
                ),
                **{key: features[key] for key in features.keys() if key not in ("steps",)},
            }
        )

        return features

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def relabel_language(step):
            # hacky way to find out which key is the language key, assuming its inside obs
            lang_key_list = [key for key in step["observation"].keys() if "instruction" in key]
            if len(lang_key_list) == 1:
                lang_key = lang_key_list[0]
            elif len(lang_key_list) > 1:
                for key in lang_key_list:
                    if "natural" in key:
                        lang_key = key
            if len(lang_key_list) == 0:
                print("No language key found!!")

            original_language = step["observation"][lang_key]
            print('ORIGINAL:', original_language)
            prompt = RelabelLanguage.prompt % (original_language, RelabelLanguage.n_variants)
            messages = []
            messages.append(dict(role="user", content=prompt))
            res = ask_chatgpt(messages)

            parsed_response = RelabelLanguage.parse_response(res)
            print('AUGMENTED:', parsed_response)
            for i, variant in enumerate(parsed_response):
                step["observation"]["language_instruction_relabel_" + str(i)] = variant
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(relabel_language)
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
            if len(feat.shape) >= 2 and feat.shape[0] >= 64 and feat.shape[1] >= 64:  # is image / depth feature
                should_jpeg_encode = (
                        isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES, feat.shape[2])
                else:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES)

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
                    size = (ResizeAndJpegEncode.MAX_RES,
                            ResizeAndJpegEncode.MAX_RES)
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
    "relabel_language": RelabelLanguage,
    "resize_and_jpeg_encode": ResizeAndJpegEncode,
    "filter_success": FilterSuccess,
    "flip_image_channels": FlipImgChannels,
    "flip_wrist_image_channels": FlipWristImgChannels,
}
