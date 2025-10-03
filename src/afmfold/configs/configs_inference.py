# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------------------------------
# Modifications:
# - Modified for use in afmfold (2025).
# ---------------------------------------------------------

# pylint: disable=C0114

import os
import numpy as np
from afmfold.protenix.config.extend_types import ListValue, RequiredValue
from biotite.structure import AtomArray

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
code_directory = os.path.dirname(current_directory)
# The model will be download to the following dir if not exists:
# "./release_data/checkpoint/model_v0.2.0.pt"
inference_configs = {
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": True,
    "input_json_path": RequiredValue(str),
    "load_checkpoint_path": os.path.join(
        code_directory, "./release_data/checkpoint/model_v0.2.0.pt"
    ),
    "num_workers": 16,
    "use_msa": True,
    # Modified in afmfold: flexible config override support.
    "guidance_kwargs": {
        "atom_array": AtomArray(0),
        "image_path": "",
        "model_path": "",
        "domain_pairs": [None,],
        "manual": np.array([]),
        "in_nm": False,
        "t_start": -1.0,
        "scaling_kwargs": {
            "func_type": "sigmoid",
            "y_max": 0.60,
            "alpha": 10,
            "beta": 10,
            "gamma": 0.5,
            "delta": 0.5,
            },
    },
}
