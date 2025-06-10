import copy
import json
import os
import re
import time
import openai
import pandas as pd
from tqdm import tqdm

prompt_template = """###Task:
Given the input instruction, the ground truth answer, and image information, evaluate whether the model response is aligned with the image content.

###Evaluation Criteria:
Without hallucination: The response is semantically similar to the ground truth answer and does not contain any contradictory factual claim with the provided information.
With hallucination: The response is completely different from the ground truth answer, or contains contradictory factual claim about the object, action, attribute, or any other detail that is not grounded in the provided information.
Unclear: The response does not give a specific answer, or refuse to answer the question.

###Instruction:
{}

###Ground Truth Answer:
{}

###Image Information:
{}

###Model Responses:
{}

###Output Format:
Response 1: With Hallucination/Without Hallucination/Unclear, [Evidence].
...
Response {}: With Hallucination/Without Hallucination/Unclear, [Evidence].
"""


def evaluate_YNQ(results_json):
    """
    Evaluates the accuracy of Yes/No question answering results.

    Attributes:
        - param results_json: Path to the JSON file containing the model's results.
    """
    with open(results_json, 'r', encoding='utf8') as file:
        results_data = json.load(file)
    Scores = {'overall': []}
    for i, results in enumerate(results_data):
        question_type = results['task']
        hal_dimension = results['evaluation_dimension']
        gt = results['Ground Truth']
        if 'YNQ' not in question_type:
            continue
        if hal_dimension not in Scores.keys():
            Scores[hal_dimension] = []

        if 'Yes' in results['answer'] or 'yes' in results['answer'] or '是' in results['answer']:
            answer = 'Yes'

        elif 'No' in results['answer'] or 'no' in results['answer'] or '否' in results['answer']:
            answer = 'No'

        else:
            answer = 'Unclear'


        if gt == answer:
            Scores['overall'].append(1)
            Scores[hal_dimension].append(1)
        else:
            Scores['overall'].append(0)
            Scores[hal_dimension].append(0)

    Accs = {}
    for type in Scores.keys():
        Accs[type] = sum(Scores[type]) / len(Scores[type])

    return Accs


def evaluate_MCQ(results_json):
    """
    Evaluates the accuracy of Multiple Choice Question (MCQ) answering results.

    Attributes:
        - param results_json: Path to the JSON file containing the model's results.
    """
    with open(results_json, 'r', encoding='utf8') as file:
        results_data = json.load(file)
    Scores = {'overall': [], 'fail': []}
    for i, results in enumerate(results_data):
        question_type = results['task']
        if 'MCQ' not in question_type:
            continue
        hal_dimension = results['evaluation_dimension']
        gt = results['Ground Truth Option']
        if hal_dimension not in Scores.keys():
            Scores[hal_dimension] = []
        if '(A)' in results['answer']:
            answer = '(A)'
        elif '(B)' in results['answer']:
            answer = '(B)'
        elif '(C)' in results['answer']:
            answer = '(C)'
        elif '(D)' in results['answer']:
            answer = '(D)'
        else:
            answer = 'Unclear'

        if gt == answer:
            Scores['overall'].append(1)
            Scores[hal_dimension].append(1)
        else:
            Scores['overall'].append(0)
            Scores[hal_dimension].append(0)

    Accs = {}
    for task in Scores.keys():
        Accs[task] = sum(Scores[task]) / len(Scores[task])

    return Accs


def merge_results(results_folder, template_json, output_json):
    """
    Merges the results from multiple models into a single JSON file.

    Attributes:
        - param results_folder: Path to the folder containing the result JSON files for each model.
        - param template_json: Path to the template JSON file containing the original data structure.
        - param output_json: Path to the output JSON file where the merged results will be saved.
    """
    outputs = []
    with open(template_json, 'r', encoding='utf8') as f:
        ori = json.load(f)

    for result in ori:
        result['answers'] = []
        outputs.append(result)

    models_list = {}
    models = os.listdir(results_folder)
    for i, model in enumerate(models):
        models_list[f'model{i + 1}'] = model
        result_json_path = f'{results_folder}/{model}/result.json'
        with open(result_json_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            for i, result in enumerate(result_data):
                answer = result['answer']
                assert outputs[i]['id'] == result['id']
                outputs[i]['answers'].append(answer.replace('\n', ' '))

    with open(output_json, 'w', encoding='utf8') as f:
        json.dump(outputs, f, indent=2)

    return models_list


def evaluate_OpenVQA_batch(merge_results_json, openai_key='YOUR_OPENAI_KEY'):
    """
    Evaluates the hallucination of model responses for Open-Ended Visual Question Answering (OpenVQA) tasks using GPT-4o.

    Attributes:
        - param merge_results_json: Path to the JSON file containing merged model responses.
        - param openai_key: Your OpenAI API key. Defaults to 'YOUR_OPENAI_KEY'.
    """
    openai.api_key = openai_key
    with open(merge_results_json, 'r', encoding='utf8') as file:
        results_data = json.load(file)
    model_num = len(results_data[0]['answer'])
    for i, results in enumerate(tqdm(results_data)):
        question_type = results['task']
        if 'OpenVQA' not in question_type:
            continue
        instruction = results['instruction']
        gt = results['Ground Truth']
        image_information = results['image_prompt']
        if 'Facts' in results.keys():
            gt = results['image_prompt']
            image_information += '\n' + results['Facts']
        answers = results['answers']
        num = len(answers)
        L = ''
        for j, answer in enumerate(answers):
            j = j + 1
            L += 'Response{}: '.format(j) + answer + '\n'

        gpt_prompt = prompt_template.format(instruction, gt, image_information, L, num)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": gpt_prompt}
                ]
            )

            gpt = response.choices[0].message.content
            results['gpt_output'] = gpt
            print(gpt_prompt)
            print('\n')
            print(gpt)

        except Exception as e:
            print('Error')
            print(e)
            time.sleep(10)

        with open(merge_results_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

    for i, results in enumerate(results_data):
        question_type = results['task']
        if 'OpenVQA' not in question_type:
            continue
        if 'gpt_output' not in results.keys():
            print(results['id'])
            print('No gpt output')
            continue
        gpt_output = results['gpt_output']
        lines = gpt_output.split("\n")
        pattern1 = re.compile(r"Response ?(\d+) ?:")
        hallu_scores = []
        for line in lines:
            match1 = pattern1.search(line)
            if match1:
                answer_id = match1.group(1)
                if 'With Hallu' in line or 'With hallu' in line:
                    results[f'Response{answer_id}'] = 'With Hallucination'
                    hallu_scores.append(1)
                elif 'Without Hallu' in line or 'Without hallu' in line:
                    results[f'Response{answer_id}'] = 'Without Hallucination'
                elif 'Unclear' in line:
                    results[f'Response{answer_id}'] = 'Unclear'
                else:
                    print(results['id'])
                    print(line)
                    print('No hal output')

    with open(merge_results_json, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
        f.flush()

    Scores = {}
    Final_Scores = {}
    for j in range(1, model_num + 1):
        Scores[f'model{j}'] = {'overall': []}
        Final_Scores[f'model{j}'] = {}
    for i, results in enumerate(results_data):
        question_type = results['task']

        if 'OpenVQA' not in question_type:
            continue
        if 'gpt_output' not in results.keys():
            continue
        hal_dimension = results['evaluation_dimension']
        answer_num = len(results['answers'])
        if hal_dimension not in Scores['model1'].keys():
            for j in range(1, answer_num + 1):
                Scores[f'model{j}'][hal_dimension] = []

        for j in range(1, answer_num + 1):
            judgement = results[f'Response{j}']

            if judgement == 'Without Hallucination':
                Scores[f'model{j}'][hal_dimension].append(1)
                Scores[f'model{j}']['overall'].append(1)
            else:
                Scores[f'model{j}'][hal_dimension].append(0)
                Scores[f'model{j}']['overall'].append(0)

    for j in range(1, model_num + 1):
        for type in Scores[f'model{j}'].keys():
            Final_Scores[f'model{j}'][type] = sum(Scores[f'model{j}'][type]) / len(Scores[f'model{j}'][type])

    return Final_Scores


def evaluate_Caption_batch(merge_results_json, openai_key='YOUR_OPENAI_KEY'):
    """
    Evaluates the hallucination of model-generated captions using GPT-4o.

    Attributes:
        - param merge_results_json: Path to the JSON file containing merged model-generated captions.
        - param openai_key: Your OpenAI API key. Defaults to 'YOUR_OPENAI_KEY'.
    """
    openai.api_key = openai_key
    with open(merge_results_json, 'r', encoding='utf8') as file:
        results_data = json.load(file)
    model_num = len(results_data[0]['answers'])
    for i, results in enumerate(tqdm(results_data)):
        question_type = results['task']
        if 'Caption' not in question_type:
            continue
        instruction = results['instruction']
        gt = results['Ground Truth']
        image_information = results['image_prompt']
        if 'Facts' in results.keys():
            image_information += '\n' + results['Facts']
        answers = results['answers']
        num = len(answers)
        L = ''
        for j, answer in enumerate(answers):
            j = j + 1
            L += 'Response{}: '.format(j) + answer + '\n'

        gpt_prompt = prompt_template.format(instruction, gt, image_information, L, num)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": gpt_prompt}
                ]
            )

            gpt = response.choices[0].message.content
            results['gpt_output'] = gpt
            print(gpt_prompt)
            print('\n')
            print(gpt)

        except Exception as e:
            print('Error')
            print(e)
            time.sleep(10)

        with open(merge_results_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
            f.flush()

    for i, results in enumerate(results_data):
        question_type = results['task']
        if 'Caption' not in question_type:
            continue
        if 'gpt_output' not in results.keys():
            print(results['id'])
            print('No gpt output')
            continue
        gpt_output = results['gpt_output']
        lines = gpt_output.split("\n")
        pattern1 = re.compile(r"Response ?(\d+) ?:")
        hallu_scores = []
        for line in lines:
            match1 = pattern1.search(line)
            if match1:
                answer_id = match1.group(1)
                if 'With Hallu' in line or 'With hallu' in line:
                    results[f'Response{answer_id}'] = 'With Hallucination'
                    hallu_scores.append(1)
                elif 'Without Hallu' in line or 'Without hallu' in line:
                    results[f'Response{answer_id}'] = 'Without Hallucination'
                elif 'Unclear' in line:
                    results[f'Response{answer_id}'] = 'Unclear'
                else:
                    print(results['id'])
                    print(line)
                    print('No hal output')
    with open(merge_results_json, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
        f.flush()

    Scores = {}
    Final_Scores = {}
    for j in range(1, model_num + 1):
        Scores[f'model{j}'] = {'overall': []}
        Final_Scores[f'model{j}'] = {}
    for i, results in enumerate(results_data):
        question_type = results['task']

        if 'Caption' not in question_type:
            continue
        if 'gpt_output' not in results.keys():
            continue
        hal_dimension = results['evaluation_dimension']
        answer_num = len(results['answers'])
        if hal_dimension not in Scores['model1'].keys():
            for j in range(1, answer_num + 1):
                Scores[f'model{j}'][hal_dimension] = []

        for j in range(1, answer_num + 1):
            judgement = results[f'Response{j}']
            if judgement == 'Without Hallucination':
                Scores[f'model{j}'][hal_dimension].append(1)
                Scores[f'model{j}']['overall'].append(1)

            else:
                Scores[f'model{j}'][hal_dimension].append(0)
                Scores[f'model{j}']['overall'].append(0)

    for j in range(1, model_num + 1):
        for type in Scores[f'model{j}'].keys():
            if len(Scores[f'model{j}'][type]) == 0:
                print(j)
                print(type)
            Final_Scores[f'model{j}'][type] = sum(Scores[f'model{j}'][type]) / len(Scores[f'model{j}'][type])

    return Final_Scores


def main():
    """
    Main function to evaluate the performance of different models on various query types.
    """
    openai_key = 'YOUR_OPENAI_KEY'
    ynq_results = []
    mcq_results = []
    # model results should be stored in the ./results/{query_type}/{model}/result.json
    for query_type in ['Clean', 'Style', 'Corruption', 'Adversarial', 'SceneText']:
        for model in os.listdir(f'./results/{query_type}'):
            print(f"Processing {query_type} - {model}")
            results_json = f'./results/{query_type}/{model}/result.json'
            YNQ_accs = evaluate_YNQ(results_json)
            MCQ_accs = evaluate_MCQ(results_json)
            ynq_entry = {
                'Query Type': query_type,
                'Model': model
            }
            for key, val in YNQ_accs.items():
                ynq_entry[key] = val
            mcq_entry = {
                'Query Type': query_type,
                'Model': model
            }
            for key, val in MCQ_accs.items():
                mcq_entry[key] = val
            ynq_results.append(ynq_entry)
            mcq_results.append(mcq_entry)
    ynq_df = pd.DataFrame(ynq_results)
    ynq_df.to_excel("YNQ_results.xlsx", index=False, engine='openpyxl')
    mcq_df = pd.DataFrame(mcq_results)
    mcq_df.to_excel("MCQ_results.xlsx", index=False, engine='openpyxl')

    OpenVQA_results = []
    for query_type in ['Clean', 'Style', 'Corruption', 'Adversarial', 'SceneText']:
        merge_results_json = f'./result_json/{query_type}_caption.json'
        models_list = merge_results(f'./results/{query_type}', f'./query_json/{query_type}.json', merge_results_json)
        OpenVQA_Scores = evaluate_OpenVQA_batch(merge_results_json, openai_key=openai_key)
        vqa_entry = {
            'Query Type': query_type,
        }
        for model, _ in OpenVQA_Scores.items():
            new_vqa_entry = copy.deepcopy(vqa_entry)
            new_vqa_entry['model'] = models_list[model]
            for key, val in OpenVQA_Scores[model].items():
                new_vqa_entry[key] = val
            OpenVQA_results.append(new_vqa_entry)
    vqa_df = pd.DataFrame(OpenVQA_results)
    vqa_df.to_excel("VQA_results.xlsx", index=False, engine='openpyxl')

    Caption_results = []
    for query_type in ['Clean', 'Style', 'Corruption', 'Adversarial', 'SceneText']:
        merge_results_json = f'./result_json/{query_type}_caption.json'
        models_list = merge_results(f'./results/{query_type}', f'./query_json/{query_type}.json', merge_results_json)
        Caption_Scores = evaluate_Caption_batch(merge_results_json, openai_key=openai_key)
        caption_entry = {
            'Query Type': query_type,
        }
        for model, _ in Caption_Scores.items():
            new_caption_entry = copy.deepcopy(caption_entry)
            new_caption_entry['model'] = models_list[model]
            for key, val in Caption_Scores[model].items():
                new_caption_entry[key] = val
            Caption_results.append(new_caption_entry)
    caption_df = pd.DataFrame(Caption_results)
    caption_df.to_excel("Caption_results.xlsx", index=False, engine='openpyxl')


if __name__ == '__main__':
    main()
