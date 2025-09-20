# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from typing import Any

from mathruler.grader import grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r".*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    # pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", response)
        given_answer = content_match.group(1).strip() if content_match else response.strip()
        print(f'given_answer = {given_answer}')
        print(f'ground_truth = {ground_truth}')
        raw_answer = given_answer
        if "<think>" in given_answer:
            given_answer.replace("<think>", "")
        if "</think>" in given_answer:
            given_answer.replace("</think>", "")
        if "<answer>" in given_answer:
            given_answer.replace("<answer>", "")
        if "</answer>" in given_answer:
            given_answer.replace("</answer>", "")
        
        # choises_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
        
        # for choice in choises_list:
        #     # if f"{choice}" in given_answer:
        #     #     given_answer = choice
        #     if f"{choice}." in given_answer:
        #         given_answer = choice
            

        print(f'given_answer = {given_answer}')
        print(f'ground_truth = {ground_truth}')

        if grade_answer(given_answer, ground_truth.strip()):
            return 1.0, raw_answer

    except Exception:
        pass

    return 0.0, raw_answer

def get_length_reward(response_length, budget):
    if response_length <= budget:
        anwser_length_reward = max((1 - 2 * ((response_length - budget) / budget)**2) , 0)
    else:
        anwser_length_reward = max((1 - 16 * ((response_length - budget) / budget)**2) , 0)

    print(f"response_length = {response_length}, budget = {budget}, length_reward = {anwser_length_reward}")
    return anwser_length_reward


def compute_score(reward_input: dict[str, Any]) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for r1v reward function.")


    response_length = int(reward_input['response_length'])
    budget_and_tokens = int(reward_input['budget_and_tokens'])
    origin_response_length = reward_input['origin_response_length']
    # print(f'origin_response_length = {origin_response_length}')

    length_reward = get_length_reward(origin_response_length, budget_and_tokens)
    format_score = format_reward(reward_input["response"])
    accuracy_score, raw_answer = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    print(f'accuracy_score = {accuracy_score}')
    
    overall_score = 0.65 * accuracy_score + 0.2 * format_score + 0.15 * length_reward
    
    eval_sample = {
        "response": reward_input["response"],
        "ground_truth": reward_input["ground_truth"],
        "format_score": format_score,
        "accuracy_score": accuracy_score,
        "length_reward": length_reward,
        "overall": overall_score,
        "raw_answer": raw_answer,
        "response_length": response_length,
        "budget_and_tokens": budget_and_tokens,
        "origin_response_length": origin_response_length,
    }
    
    return {
        "overall": overall_score,
        "format": format_score,
        "accuracy": accuracy_score,
        "length_reward": length_reward,
    }, eval_sample
