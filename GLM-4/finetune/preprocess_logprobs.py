import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
import matplotlib.pyplot as plt
import os


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


def logprobs_to_standard(input_file = Path(__file__).parent / "cleaned_logprobs.jsonl"):
    data = load_logprobs_jsonl(input_file)
    batched_conv = data["messages"]
    output = []
    for conv in batched_conv:
        output.append(json.dumps({
            "messages": conv,
        }))

    output_path = Path(__file__).parent / "standard.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    return


def eval_to_npz(input_file = Path(__file__).parent / "cleaned_logprobs.jsonl", max_input_length = 24576, max_output_length = 1536, tokenizer_model = "THUDM/GLM-4-9B-0414"):
    data = load_logprobs_jsonl(input_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, padding_side="left", trust_remote_code=True)
    batched_conv = data["messages"]
    saved_files = []
    for conv in batched_conv:
        input_ids = [151331, 151333]

        # Assume combined system + user + assistant + user + completion format
        new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
        input_ids = new_input_ids
        last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1

        output_prompt, output_ids = (
            input_ids[:1],
            input_ids[last_assistant_index:],
        )

        output_ids.append(151336)

        input_ids_np = np.array(input_ids[:last_assistant_index] + output_prompt[:1])
        # input_ids_np = np.pad(input_ids_np, (0,max_length-input_ids_np.shape[0]), constant_values=0)
        output_ids_np = np.array(output_ids[:max_output_length])

        print("input:")
        print(tokenizer.decode(input_ids_np).strip())
        print()
        print("output:")
        print(tokenizer.decode(output_ids_np).strip())

        random_integer = np.random.randint(0, 999999999999)
        padded_string = f"{random_integer:012}"

        outfile =  "output_" + padded_string + ".npz"

        np.savez_compressed(Path(__file__).parent / outfile, input_ids=input_ids_np, output_ids=output_ids_np)

        saved_files.append(outfile)

    del batched_conv, conv, input_ids, new_input_ids
    torch.cuda.empty_cache()

    output_path = Path(__file__).parent / "eval_index.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(saved_files))

    return


def logprobs_to_npz(input_file = Path(__file__).parent / "cleaned_logprobs.jsonl", max_input_length = 24576, max_output_length = 1536, tokenizer_model = "THUDM/GLM-4-9B-0414", num_logprobs=8):
    data = load_logprobs_jsonl(input_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, padding_side="left", trust_remote_code=True)
    batched_conv = data["messages"]
    batched_logprobs = data["logprobs"]
    saved_files = []
    for conv, logprobs in zip(batched_conv, batched_logprobs):
        input_ids = [151331, 151333]
        # Assume combined system + user + assistant + user + completion format
        new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
        input_ids = new_input_ids
        last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1

        labels = []
        label_indices = []
        loss_mask_sums = []
        found_assistant_tag = False
        for index in range(len(input_ids)):
            logprob_index = index-last_assistant_index-2
            if logprob_index > -1:
                temp_labels = []
                temp_label_indices = []
                temp_loss_mask_sum = 0
                for pair in logprobs[logprob_index]:
                    temp_labels.append(torch.tensor(float(pair[1]), dtype=torch.bfloat16))
                    temp_label_indices.append(torch.tensor(tokenizer.convert_tokens_to_ids(pair[0]), dtype=torch.long))
                    temp_loss_mask_sum += 1
                labels.append(torch.stack(temp_labels).repeat(num_logprobs)[0:num_logprobs])
                label_indices.append(torch.stack(temp_label_indices).repeat(num_logprobs)[0:num_logprobs])
                loss_mask_sums.append(temp_loss_mask_sum)
            else:
                if input_ids[index] == 151337:
                    found_assistant_tag = True
                elif found_assistant_tag:
                    # Edge case where logprobs doesn't capture token 198 inbetween assist. tag and start of answer
                    labels.append(torch.tensor(float("-0.00000001"), dtype=torch.bfloat16).repeat(num_logprobs)[0:num_logprobs])
                    label_indices.append(torch.tensor(input_ids[index], dtype=torch.long).repeat(num_logprobs)[0:num_logprobs])
                    loss_mask_sums.append(1)
                else:
                    labels.append(torch.tensor(-100, dtype=torch.bfloat16).repeat(num_logprobs)[0:num_logprobs])
                    label_indices.append(torch.tensor(tokenizer.pad_token_id, dtype=torch.long).repeat(num_logprobs)[0:num_logprobs])
                    loss_mask_sums.append(0)

        # We want to predict EOS
        input_ids.append(151336)
        labels.append(torch.tensor(float("-0.00000001"), dtype=torch.bfloat16).repeat(num_logprobs)[0:num_logprobs])
        label_indices.append(torch.tensor(151336, dtype=torch.long).repeat(num_logprobs)[0:num_logprobs])
        loss_mask_sums.append(1)

        max_length = max_input_length + max_output_length + 1

        input_ids = torch.tensor(np.array(input_ids[:max_length]), dtype=torch.long)
        labels = torch.stack(labels)[:max_length]
        label_indices = torch.stack(label_indices)[:max_length]
        lm_sum = torch.tensor(np.sum(loss_mask_sums[:max_length]), dtype=torch.bfloat16)

        random_integer = np.random.randint(0, 999999999999)
        padded_string = f"{random_integer:012}"

        temp_file_dict = {}

        input_ids_prefix = "out_" + padded_string + "_input_ids.pt"
        torch.save(input_ids, Path(__file__).parent / input_ids_prefix)
        temp_file_dict['input_ids'] = input_ids_prefix

        labels_prefix = "out_" + padded_string + "_labels.pt"
        torch.save(labels, Path(__file__).parent / labels_prefix)
        temp_file_dict['labels'] = labels_prefix

        label_indices_prefix = "out_" + padded_string + "_label_indices.pt"
        torch.save(label_indices, Path(__file__).parent / label_indices_prefix)
        temp_file_dict['label_indices'] = label_indices_prefix

        lm_sum_prefix = "out_" + padded_string + "_lm_sum.pt"
        torch.save(lm_sum, Path(__file__).parent / lm_sum_prefix)
        temp_file_dict['lm_sum'] = lm_sum_prefix

        saved_files.append(json.dumps(temp_file_dict))

    output_path = Path(__file__).parent / "index.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(saved_files))

    return


def generate_plots(input_tokens, output_tokens, message_counts, output_dir=Path(__file__).parent):
    """
    Generate matplotlib plots for input tokens, output tokens, and message counts.
    
    Args:
        input_tokens (list): List of input token counts
        output_tokens (list): List of output token counts
        message_counts (list): List of message counts
        output_dir (Path): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot input tokens distribution
    plt.figure(figsize=(6, 4), dpi=200)
    plt.hist(input_tokens, bins=20, alpha=0.95, color='blue')
    plt.title('Distribution of GLM Input Tokens')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_dir / 'input_tokens_distribution.png')
    plt.show()
    plt.close()
    
    # Plot output tokens distribution
    plt.figure(figsize=(6, 4), dpi=200)
    plt.hist(output_tokens, bins=20, alpha=0.95, color='green')
    plt.title('Distribution of GLM Output Tokens')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_dir / 'output_tokens_distribution.png')
    plt.show()
    plt.close()
    
    # Plot message counts distribution
    plt.figure(figsize=(6, 4), dpi=200)
    plt.hist(message_counts, bins=range(min(message_counts), max(message_counts)+2, 2),
             alpha=0.95, color='purple', align='left')
    plt.title('Distribution of Message Counts in Prompts')
    plt.xlabel('Number of Messages')
    plt.ylabel('Frequency')
    plt.xticks(range(min(message_counts), max(message_counts)+2, 2))
    plt.grid(True)
    plt.savefig(output_dir / 'message_counts_distribution.png')
    plt.show()
    plt.close()


    # Plot input vs. output distribution
    plt.figure(figsize=(6, 4), dpi=200)
    plt.scatter(input_tokens, output_tokens, alpha=0.95, color='orange')
    plt.title('Distribution of GLM I/O Tokens')
    plt.xlabel('Number of Input Tokens')
    plt.ylabel('Number of Output Tokens')
    plt.grid(True)
    plt.savefig(output_dir / 'io_tokens_distribution.png')
    plt.show()
    plt.close()

    print(f"Saved visualization plots to {output_dir}")


def process_logprobs(input_path, max_input_length=24576, max_output_length=1536, tokenizer_model='THUDM/GLM-4-9B-0414', enforce_lengths=True, DEBUG_LOSS=False):
    """Process logprobs.json and filter tokens not in vocabulary"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, padding_side="left", trust_remote_code=True)
        deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528", padding_side="left", trust_remote_code=True)

        # glmz1_tokenizer = AutoTokenizer.from_pretrained("THUDM/GLM-Z1-Rumination-32B-0414", padding_side="left", trust_remote_code=True)
        # print(list(deepseek_tokenizer.added_tokens_encoder.keys()))
        # print(list(glmz1_tokenizer.added_tokens_encoder.keys()))

        glm_special_tokens = ['<|endoftext|>', '[MASK]', '[gMASK]', '[sMASK]', '<sop>', '<eop>', '<|system|>',
                              '<|user|>', '<|assistant|>', '<|observation|>', '<|begin_of_image|>', '<|end_of_image|>',
                              '<|begin_of_video|>', '<|end_of_video|>']

        deepseek_special_tokens = ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜▁pad▁｜>', '<think>', '</think>',
                                   '<｜fim▁hole｜>',
                                   '<｜fim▁begin｜>', '<｜fim▁end｜>', '<｜User｜>', '<｜Assistant｜>', '<|EOT|>',
                                   '<｜tool▁calls▁begin｜>',
                                   '<｜tool▁calls▁end｜>', '<｜tool▁call▁begin｜>', '<｜tool▁call▁end｜>',
                                   '<｜tool▁outputs▁begin｜>',
                                   '<｜tool▁outputs▁end｜>', '<｜tool▁output▁begin｜>', '<｜tool▁output▁end｜>',
                                   '<｜tool▁sep｜>']

        deepseek_to_glm_no_map = {
            "<｜User｜>": '<|user|>',
            "<｜Assistant｜>": "<|assistant|>",
        }

        deepseek_to_glm_map = {
            "<｜end▁of▁sentence｜>": "<|endoftext|>",
        }

        # Find/replace deepseek special tokens first
        with open(input_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        file_content_replaced = file_content
        f.close()
        for deepseek, glm in zip(deepseek_to_glm_map.keys(), deepseek_to_glm_map.values()):
            file_content_replaced = file_content_replaced.replace(deepseek, glm)

        if file_content != file_content_replaced:
            with open(input_path, 'w', encoding='utf-8') as f:
                f.write(file_content_replaced)
            f.close()

        with open(input_path, 'r', encoding='utf-8') as f:
            logprobs_data = json.load(f)

        output = []
        line_index = 0
        
        # Lists to store metrics for visualization
        input_token_counts = []
        output_token_counts = []
        message_counts = []
        
        for line in logprobs_data:
            if line['name'] != 'async_custom_logprob_hook':
                print('WARNING: Endpoint ' + line['name'] + ' does not match async_custom_logprob_hook. Dropping line.')
                continue
            if line['input']['args'][0] != 'OR DS R1 0528 Paid':
                print('WARNING: Model name ' + line['input']['args'][0] + ' not found. Dropping line.')
                continue
            error_occured = False
            line_index += 1
            temp_prompt = line['input']['args'][1]
            prompt = []
            for message in temp_prompt:
                temp_dict = {}
                temp_dict['role'] = message['role']
                if type(message['content']) is list:
                    temp_str = ""
                    for sub_content in message['content']:
                        temp_str = temp_str + sub_content['text']
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

            tokenized_completion = tokenizer.tokenize(conv)

            logprobs = line['input']['args'][3]
            # Filter tokens that exist in vocab
            write_logprobs = []
            for token in tokenized_completion:
                temp_logprobs = []
                try:
                    next_logprob = logprobs.pop(0)
                except Exception as e:
                    error_occured = True
                if next_logprob['tk'] == token:
                    top_logprobs = next_logprob['tp']
                    for top_logprob in top_logprobs:
                        if not DEBUG_LOSS:
                            if top_logprob['tk'] in tokenizer.vocab:
                                temp_logprobs.append([top_logprob['tk'], str(top_logprob['lp'])])
                            elif top_logprob['tk'] in deepseek_special_tokens:
                                print("WARNING: Unmapped Token " + top_logprob['tk'])
                        else:
                            if top_logprob['tk'] == token:
                                temp_logprobs.append([top_logprob['tk'], "-0.00000001"])

                else:
                    latch_tokens = next_logprob['tk']
                    pop_index = 0
                    while latch_tokens != token:
                        if len(logprobs) < pop_index + 1:
                            pop_index = 0
                            latch_tokens = token
                            break
                        latch_tokens = latch_tokens + logprobs[pop_index]['tk']
                        pop_index += 1
                        if len(latch_tokens) > 32:
                            pop_index = 0
                            latch_tokens = token
                            break
                    for i in range(0, pop_index):
                        logprobs.pop(0)
                    # Number as strings makes DataManager behave
                    temp_logprobs.append([latch_tokens, "-0.00000001"])
                    if latch_tokens in deepseek_special_tokens:
                        print("WARNING: Unmapped Token " + latch_tokens)
                write_logprobs.append(temp_logprobs)

            if error_occured:
                continue
            glm_input_len = len(tokenizer.apply_chat_template(prompt, tokenize=True, return_dict=False))
            deepseek_input_len = len(deepseek_tokenizer.apply_chat_template(prompt, tokenize=True, return_dict=False))
            print("INFO: Input Tokens Deepseek " + str(deepseek_input_len) + " -> GLM " + str(glm_input_len))
            print("INFO: Input Tokens Ratio " + str(round(deepseek_input_len / glm_input_len, 4)))
            print("INFO: Output Tokens GLM " + str(len(write_logprobs)))
            
            # Collect metrics for visualization
            input_token_counts.append(glm_input_len)
            output_token_counts.append(len(write_logprobs))
            message_counts.append(len(prompt))

            if enforce_lengths:
                # Use "+ 5" to account for special tokens/padding buffer
                if glm_input_len + 5 <= max_input_length:
                    if len(write_logprobs) + 5 <= max_output_length:
                        output.append(json.dumps({
                            "messages": prompt,
                            "logprobs": write_logprobs
                        }))
                    else:
                        print("WARNING: Output length exceeds max output length at line " + str(line_index) + ", dropping sample.")
                else:
                    print("WARNING: Input length exceeds max input length at line " + str(line_index) + ", dropping sample.")
            else:
                output.append(json.dumps({
                    "messages": prompt,
                    "logprobs": write_logprobs
                }))

        # Generate plots after processing all data
        if input_token_counts:
            generate_plots(
                input_tokens=input_token_counts,
                output_tokens=output_token_counts,
                message_counts=message_counts,
                output_dir=Path(__file__).parent
            )
        
        return output
    except FileNotFoundError:
        print(f"Error: File not found - {input_path}", file=sys.stderr)
        sys.exit(1)


def main():
    max_input_length = 24576
    max_output_length = 1536
    model = 'THUDM/GLM-4-9B-0414'

    # Langfuse -> Logprobs
    processed = process_logprobs(Path(__file__).parent / "logprobs.json", max_input_length=max_input_length, max_output_length=max_output_length, tokenizer_model=model)

    # Logprobs -> cleaned_logprobs.jsonl
    output_path = Path(__file__).parent / "cleaned_logprobs.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed))

    # cleaned_logprobs.jsonl -> *.npz
    logprobs_to_npz(max_input_length=max_input_length, max_output_length=max_output_length, tokenizer_model=model)
    eval_to_npz(max_input_length=max_input_length, max_output_length=max_output_length, tokenizer_model=model)
    logprobs_to_standard()

if __name__ == '__main__':
    main()