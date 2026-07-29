# AF disaggregation type

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========

from enum import Enum
import argparse

class AFDPerspective(Enum):
    AFD_PERSPECTIVE_ATTN = "attn"
    AFD_PERSPECTIVE_FFN = "ffn"

    def __str__(self):
        return self.value

def parse_afd_micro_batch(value: str) -> int:
    MIN_MICRO_BATCH_SIZE = 1
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")

    if ivalue < MIN_MICRO_BATCH_SIZE:
        raise argparse.ArgumentTypeError(
            f"Value must be an integer no less than {MIN_MICRO_BATCH_SIZE}."
        )
    return ivalue
