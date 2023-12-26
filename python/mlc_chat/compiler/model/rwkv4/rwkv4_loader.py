"""
This file specifies how MLC's Mistral parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""
import functools

import numpy as np

from ...loader import ExternMapping
from ...quantization import Quantization
from .rwkv4_model import RWKVConfig, RWKVForCasualLM


def huggingface(model_config: RWKVConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : RWKVConfig
        The configuration of the Mistral model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = RWKVForCasualLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(  # pylint: disable=unbalanced-tuple-unpacking
        spec=model.get_default_spec()
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    return mapping
