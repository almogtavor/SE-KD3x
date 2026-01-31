# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
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

"""Binary of evaluating instruction following. See README.md."""

import argparse
import logging
import os

from . import evaluation_lib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate instruction-following performance."
    )
    parser.add_argument(
        "--input_data",
        required=True,
        help="Path to input data",
    )
    parser.add_argument(
        "--input_response_data",
        required=False,
        help="Path to input response data",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for inference and eval results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    inputs = evaluation_lib.read_prompt_list(args.input_data)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
        args.input_response_data
    )

    # Get instruction-following results
    for func, output_file_name in [
        (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
        (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
    ]:
        logging.info("Generating %s...", output_file_name)

        outputs = []
        for inp in inputs:
            outputs.append(func(inp, prompt_to_response))

        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        logging.info("Accuracy: %f", accuracy)

        output_path = os.path.join(
            args.output_dir, output_file_name + ".jsonl"
        )
        evaluation_lib.write_outputs(output_path, outputs)
        logging.info("Generated: %s", output_path)

        # Print instruction-following accuracy report
        print("=" * 64)
        print(f"{output_path} Accuracy Scores:")
        evaluation_lib.print_report(outputs)


if __name__ == "__main__":
    main()
