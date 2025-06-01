import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer


def load_logprobs_jsonl(file_path):
    data = {"messages": [], "logprobs": []}
    with open(file_path, 'r') as f:
        for line in f:
            try:
                load_line = json.loads(line)
                data["messages"].append(load_line["messages"])
                data["logprobs"].append(load_line["logprobs"])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {e}")
    return data


def logprobs_to_npz(input_file = Path(__file__).parent / "cleaned_logprobs.jsonl", max_input_length = 28672, max_output_length = 4096, tokenizer_model = "THUDM/GLM-4-9B-0414"):
    data = load_logprobs_jsonl(input_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, padding_side="left", trust_remote_code=True)
    batched_conv = data["messages"]
    batched_logprobs = data["logprobs"]
    saved_files = []
    for conv, logprobs in zip(batched_conv, batched_logprobs):
        input_ids = [151331, 151333]
        loss_masks = [False, False]
        # Assume combined system + user + assistant + user + completion format
        new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
        input_ids = new_input_ids
        loss_masks = [False] * len(input_ids)

        last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1

        for j in range(last_assistant_index + 2, len(input_ids)):
            loss_masks[j] = True

        labels = []
        for index in range(len(input_ids)):
            logprob_index = index-last_assistant_index-2
            if logprob_index > -1:
                labels.append(logprobs[logprob_index])
            else:
                labels.append([["!", -100]])

        # We want to predict EOS
        input_ids.append(151336)
        loss_masks.append(True)
        labels.append([[tokenizer.convert_ids_to_tokens(151336), "-0.00000001"]])

        for i in range(len(labels)):
            temp = []
            for label in labels[i]:
                temp.append([tokenizer.convert_tokens_to_ids(label[0]), float(label[1])])
            labels[i] = temp

        labels_multi_hot = []
        for label in tqdm(labels):
            temp_stack = torch.Tensor()
            temp_stack = temp_stack.to(device='cuda')
            for sub_label in label:
                temp_stack = torch.cat((temp_stack, torch.unsqueeze(F.one_hot(torch.tensor(sub_label[0], dtype=torch.int64, device='cuda'), num_classes=151552) * torch.tensor(sub_label[1], dtype=torch.bfloat16, device='cuda'), axis=0)))
            labels_multi_hot.append(torch.sum(temp_stack, axis=0).to(dtype=torch.bfloat16))

        labels_multi_hot = torch.stack(labels_multi_hot)
        labels_multi_hot = torch.where(labels_multi_hot == 0, torch.tensor(-100, dtype=torch.bfloat16), labels_multi_hot)
        labels_multi_hot = labels_multi_hot.to('cpu')
        max_length = max_input_length + max_output_length + 1

        input_ids_np = np.array(input_ids[:max_length])
        input_ids_np = np.pad(input_ids_np, (0,max_length-input_ids_np.shape[0]), constant_values=0)
        labels_np = labels_multi_hot[:max_length].to(dtype=torch.float32).numpy()
        labels_np = np.pad(labels_np, ((0, max_length - labels_np.shape[0]),(0,0)), constant_values=-100)
        loss_masks_np = np.array(loss_masks[:max_length]).astype(np.float32)
        loss_masks_np = np.pad(loss_masks_np, (0, max_length - loss_masks_np.shape[0]), constant_values=0)

        random_integer = np.random.randint(0, 999999999999)
        padded_string = f"{random_integer:012}"

        outfile =  "output_" + padded_string + ".npz"

        np.savez_compressed(Path(__file__).parent / outfile, input_ids=input_ids_np, labels=labels_np, loss_masks=loss_masks_np)

        saved_files.append(outfile)

    del batched_conv, conv, input_ids, loss_masks, new_input_ids, labels
    torch.cuda.empty_cache()

    output_path = Path(__file__).parent / "index.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(saved_files))

    return


def process_logprobs(input_path):
    """Process logprobs.json and filter tokens not in vocabulary"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            logprobs_data = json.load(f)
            
        output = []
        for line in logprobs_data:
            temp_prompt = line['input']['args'][1]
            prompt = []
            for message in temp_prompt:
                temp_dict = {}
                temp_dict['role'] = message['role']
                if type(message['content']) is list:
                    temp_str = ""
                    for sub_content in message['content']:
                        temp_str = temp_str + sub_content['text']
                        print()
                    temp_dict['content'] = temp_str
                else:
                    temp_dict['content'] = message['content']
                prompt.append(temp_dict)
            prompt[0]['role'] = 'system'

            temp_dict = {}
            temp_dict['role'] = 'assistant'
            temp_dict['content'] = line['input']['args'][2]
            prompt.append(temp_dict)

            conv = line['input']['args'][2]
            tokenizer = AutoTokenizer.from_pretrained("THUDM/GLM-4-9B-0414", padding_side="left", trust_remote_code=True)
            tokenized_completion = tokenizer.tokenize(conv)

            logprobs = line['input']['args'][3]
            # Filter tokens that exist in vocab
            write_logprobs = []
            for token in tokenized_completion:
                temp_logprobs = []
                next_logprob = logprobs.pop(0)
                if next_logprob['tk'] == token:
                    top_logprobs = next_logprob['tp']
                    for top_logprob in top_logprobs:
                        if top_logprob['tk'] in tokenizer.vocab:
                            temp_logprobs.append([top_logprob['tk'], str(top_logprob['lp'])])
                else:
                    latch_tokens = next_logprob['tk']
                    pop_index = 0
                    while latch_tokens != token:
                        if len(logprobs) < pop_index + 1:
                            pop_index = 0
                            print(token)
                            latch_tokens = token
                            break
                        latch_tokens = latch_tokens + logprobs[pop_index]['tk']
                        pop_index += 1
                        if len(latch_tokens) > 32:
                            pop_index = 0
                            print(token)
                            latch_tokens = token
                            break
                    for i in range(0, pop_index):
                        logprobs.pop(0)
                    # Number as strings makes DataManager behave
                    temp_logprobs.append([latch_tokens, "-0.00000001"])
                write_logprobs.append(temp_logprobs)
                    
            output.append(json.dumps({
                "messages": prompt,
                "logprobs": write_logprobs
            }))

        print()
        return output
    except FileNotFoundError:
        print(f"Error: File not found - {input_path}", file=sys.stderr)
        sys.exit(1)


def main():
    # Langfuse -> Logprobs
    processed = process_logprobs(Path(__file__).parent / "logprobs.json")

    # Logprobs -> cleaned_logprobs.jsonl
    output_path = Path(__file__).parent / "cleaned_logprobs.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed))

    # cleaned_logprobs.jsonl -> *.npz
    logprobs_to_npz()


if __name__ == '__main__':
    main()