from datasets import load_dataset, interleave_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

from huggingface_hub import login

import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


# Make sure these are correct!
MODEL_ID = "THUDM/GLM-4-9B-0414"
MAX_SEQ_LEN = 26624
DO_PLOT = True
SEED = 42
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def plot(dataset, title):
    seq_lens = []
    bins = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    for sample in dataset:
        seq_lens.append(len(sample["input_ids"]))

    plt.figure(figsize=(6, 4), dpi=200)

    color = 0.0
    counts, _, _ = plt.hist(seq_lens, bins=bins)
    plt.close()  # Don't want the plot from plt.hist
    x = list(range(len(counts)))
    plt.bar(x, counts, color=plt.cm.plasma(color), alpha=0.95)

    plt.yscale("linear")
    ticks = []
    for i in range(1, len(bins)):
        ticks.append('< ' + str(bins[i]))
    plt.xticks(x, ticks)
    # plt.legend(loc=0, fontsize="medium")
    plt.xlabel("Sequence Lengths", fontsize="medium")
    plt.ylabel("Count", fontsize="medium")
    plt.title(title, fontsize="large")
    plt.tight_layout()

    img_title = title.lower() + ".png"
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    plt.savefig("./plots/" + img_title)

    plt.show()

    return


def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQ_LEN, truncation=True, add_special_tokens=False)


def preprocess(example):
    if 'conversation' in example:
        return {
            "text": tokenizer.apply_chat_template(
                example["conversation"],
                tokenize=False,
            )
        }
    elif 'messages' in example:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }
    elif 'conversations' in example:
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                tokenize=False,
            )
        }
    else:
        print('ERROR: Preprocess likely not implemented for: ' + str(example))
        exit()


def prepare_ds_mix_from_ids(DATASET_IDS, DATASET_PROBS, DATASET_SPLITS, seed=42, total_calib_samples=1024, plot_name=None):
    random.seed(seed)

    max_ds_size = round(np.max(DATASET_PROBS) * total_calib_samples)
    datasets = []
    for DATASET_ID, DATASET_PROB, DATASET_SPLIT in zip(DATASET_IDS, DATASET_PROBS, DATASET_SPLITS):
        # Apply a scaling factor to initial dataset sizes such that when interleaved w/ "first_exhausted" we maximize chances of exhausting the largest dataset prob
        scale_factor = np.max(DATASET_PROB) / DATASET_PROB
        start_index = random.randint(a=0, b=round(max_ds_size*scale_factor))
        temp = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[{start_index}:{start_index+round(max_ds_size*scale_factor)}]")
        temp.shuffle(seed=seed)
        temp = temp.map(preprocess)
        temp = temp.map(tokenize, remove_columns=temp.column_names)
        datasets.append(temp)

    dataset = interleave_datasets(datasets, probabilities=DATASET_PROBS, seed=seed, stopping_strategy="first_exhausted")

    if DO_PLOT and plot_name is not None:
        plot(dataset, title=plot_name)

    return dataset


finetune_ds = load_dataset('json', data_files='../my_data/standard.jsonl')
finetune_ds.shuffle(seed=SEED)
finetune_ds = finetune_ds.map(preprocess)['train']
finetune_ds = finetune_ds.map(tokenize, remove_columns=finetune_ds.column_names)
if DO_PLOT:
    plot(finetune_ds, title='Finetune')


# For accessing gated datasets/models
login()


chat_ids = ['lmsys/lmsys-chat-1m',
            'HuggingFaceH4/ultrachat_200k',
            'robinsmits/ChatAlpaca-20K']
chat_probs = np.array([1, 2, 1])
chat_probs = list(chat_probs/np.sum(chat_probs))
chat_splits = ['train', 'train_sft', 'train']
chat_ds = prepare_ds_mix_from_ids(chat_ids, chat_probs, chat_splits, seed=SEED, plot_name='Chat')


# # TODO: Preprocess math into chat format
# # math_ids = ['ZhangShenao/bigmath_chat_epochs',
# #             'agentica-org/DeepScaleR-Preview-Dataset',
# #             'xDAN-datasets/Agentic-Math-Level5-filtered',
# #             'xDAN-datasets/facebook_natural_reasoning']
# # math_probs = np.array([1, 1, 1, 1])
# # math_probs = list(math_probs/np.sum(math_probs))
# # math_splits = ['train', 'train_sft', 'train', 'train']
# # math_ds = prepare_ds_mix_from_ids(math_ids, math_probs, math_splits, seed=SEED, plot_name='Math')


code_ids = ['danbider/chat-formatted-magicoder',
            'xDAN-datasets/glaive_code_assistant_140K']
code_probs = np.array([2, 1])
code_probs = list(code_probs/np.sum(code_probs))
code_splits = ['train', 'train']
code_ds = prepare_ds_mix_from_ids(code_ids, code_probs, code_splits, seed=SEED, plot_name='Code')


dataset = interleave_datasets([finetune_ds, chat_ds, code_ds], probabilities=[3/6, 2/6, 1/6], seed=SEED, stopping_strategy="first_exhausted")
if DO_PLOT:
    plot(dataset, title='Full')


# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype="auto")


# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)