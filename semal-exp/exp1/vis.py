#%%
import os
import jax
import jax.numpy as jnp
import pandas as pd
import re
from gemma import gm
from typing import List
import gc
gc.collect()

# --- Gemma Initialization (Assuming this works from previous turns) ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
os.system("nvidia-smi") # Uncomment if needed

model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
tokenizer = gm.text.Gemma3Tokenizer()


#%%
# Define the two distinct lists of numbers
semantic_association: List[int] = [69, 420, 911, 666, 1776, 1337] # Added a few more for size
regular_numbers: List[int] = [1, 5, 10, 100, 123, 1515] # Roughly equal size
ALL_NUMBERS = semantic_association[:] + regular_numbers[:]
# ALL_NUMBERS = regular_numbers[:]
MAX_OPERANDS = 100
MAX_TOKENS_PER_OUTPUT = 15 # Set a hard limit for output tokens
DIGIT_TOKENS = [tokenizer.encode(str(i), add_bos=False)[0] for i in range(10)]

# Get special token IDs for stopping
EOS_ID = tokenizer.special_tokens.EOS
# Get the full stop ID (or another non-digit token) for stopping after the number
# This relies on the tokenizer being able to encode '.'
# FULL_STOP_ID = tokenizer.get_full_stop_id()
FULL_STOP_ID = tokenizer.encode('.', add_bos=False)[0]


def generate_counting_examples(number: int, num_range: int, step: int = 5) -> pd.DataFrame:
  """
  Generates a DataFrame of arithmetic strings and expected answers.
  """
  strings: List[str] = []
  answers: List[int] = []
  number_str = str(number)

  # Iterate through the number of additions, from 1 up to num_range,
  # incrementing by the specified step.
  # n_adds represents the number of '+' signs (which is the index).
  for n_adds in range(1, num_range + 1, step):

    n_operands = n_adds + 1

    # 1. Calculate the expected answer
    expected_answer = number * n_operands

    # 2. Construct the arithmetic string: "N + N + N..."
    # Create a list of 'N' strings (the operands), then join them with " + ".
    operands_list = [number_str] * n_operands
    string = "+".join(operands_list)		## TODO: no space!

    strings.append(string)
    answers.append(expected_answer)

  df = pd.DataFrame({"String": strings, "Expected_Answer": answers})

  # Set the index explicitly based on the generated number of additions (n_adds)
  df['Num_Adds'] = list(range(1, num_range + 1, step))
  df.set_index('Num_Adds', inplace=True)
  df.index.name = "Num_Adds"

  return df

# --- Fixed Generation Logic (Wrapped for Experiment) ---

def run_gemma_generation(model, params, tokenizer, prompt_text: str, rng_key: jax.random.PRNGKey) -> str:
    """
    Runs the model's generation loop on a single prompt.
    Returns the decoded generated text (excluding the input prompt).
    """
    prompt = tokenizer.encode(f"{prompt_text}=", add_bos=True)
    current_tokens = jnp.asarray(prompt)

    # Generate tokens until a stop condition is met
    for iter_count in range(MAX_TOKENS_PER_OUTPUT):
        # Use a new key for each step in the sampling loop
        rng_key, step_key = jax.random.split(rng_key)

        # 1. Apply the model
        out = model.apply(
            {'params': params},
            tokens=current_tokens,
            return_last_only=True,
            return_hidden_states=False
        )

        # 2. Sample the next token
        next_token_id = jax.random.categorical(step_key, out.logits)
        next_token_array = jnp.array([next_token_id])

        # 3. Add the generated token to the sequence
        current_tokens = jnp.concatenate([current_tokens, next_token_array], axis=0)

        # 4. Check if the generated token is NOT a digit, and stop if so.
        # Note: You may need to add token IDs for the space or newline if they are desired.
        if next_token_id not in DIGIT_TOKENS:
            print("\n[STOP CONDITION MET: Non-digit token found]")
            # You may want to skip appending this non-digit token if you want a clean numerical result
            # break

            # To keep the punctuation (like the space before 'The'), but stop:
            if next_token_id != FULL_STOP_ID: # If it's a space or another token, we include it and stop
                current_tokens = jnp.concatenate([current_tokens, next_token_array], axis=0)

            break

    # Decode the full sequence and return only the generated part
    full_result = tokenizer.decode(current_tokens)

    # Find the position of the '=' sign in the decoded text and return everything after it
    try:
        # Use regex to find the first sequence of digits after the last '=' (the answer)
        match = re.search(r'=\s*(\d+)', full_result)
        if match:
            return match.group(1)
    except:
        pass # Fallback to returning the full generated text

    return full_result[len(prompt_text) + 1:].strip() # Simple string slicing fallback


# --- Main Experiment Execution ---

FINAL_RESULTS = []
BASE_RNG_KEY = jax.random.key(42) # Fixed starting key for reproducibility

print(f"Starting experiment with {len(ALL_NUMBERS)} numbers and up to {MAX_OPERANDS} additions.")

# for number in ALL_NUMBERS:
#     # 1. Generate the DataFrame for the current number
#     df_examples = generate_counting_examples(number, MAX_OPERANDS)

#     # 2. Add columns for the experiment results
#     df_examples['Gemma_Answer_Str'] = ''
#     df_examples['Gemma_Answer_Int'] = pd.NA
#     df_examples['Is_Correct'] = False

#     print(f"\n--- Running: {number} ---")

#     for index, row in df_examples.iterrows():
#         # Generate a new RNG key for each prompt
#         BASE_RNG_KEY, current_rng = jax.random.split(BASE_RNG_KEY)

#         # 3. Run the model generation
#         generated_answer_str = run_gemma_generation(
#             model, params, tokenizer, row['String'], current_rng
#         )

#         # 4. Clean and validate the result
#         clean_answer = re.match(r'^\d+', generated_answer_str.strip())

#         if clean_answer:
#             gemma_answer_int = int(clean_answer.group(0))
#             is_correct = (gemma_answer_int == row['Expected_Answer'])
#         else:
#             gemma_answer_int = pd.NA
#             is_correct = False

#         # 5. Store results back into the DataFrame
#         df_examples.loc[index, 'Gemma_Answer_Str'] = generated_answer_str
#         df_examples.loc[index, 'Gemma_Answer_Int'] = gemma_answer_int
#         df_examples.loc[index, 'Is_Correct'] = is_correct

#         # Print progress (optional)
#         print(f"L={index+1}: Expected={row['Expected_Answer']}, Predicted={generated_answer_str.strip()}, Correct={is_correct}")
#         # print(f"L={index+1}: Expected={row['Expected_Answer']}, Predicted={clean_answer}, Correct={is_correct}")

#     # 6. Save the DataFrame for the current number
#     df_examples.to_csv(f'exp1_results_{number}.csv')
#     FINAL_RESULTS.append(df_examples)

# print("\nAll data collection complete. Results saved as CSV files.")


#%% 📊 Plotting Normalized MSE vs. Sequence Length

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import numpy as np

print("All results:", FINAL_RESULTS)

## Load the results from all CSVs
FINAL_RESULTS = []
for number in ALL_NUMBERS:
	df = pd.read_csv(f'exp1_results_{number}.csv', index_col='Num_Adds')
	FINAL_RESULTS.append(df)

# Combine all results into one large DataFrame for plotting
df_all = pd.concat(FINAL_RESULTS, keys=ALL_NUMBERS, names=['Number', 'Num_Adds'])
df_all['Expected_Answer'] = pd.to_numeric(df_all['Expected_Answer'], errors='coerce')
df_all['Gemma_Answer_Int'] = pd.to_numeric(df_all['Gemma_Answer_Int'], errors='coerce')

# Calculate the squared error and normalize it
df_all['Squared_Error'] = (df_all['Expected_Answer'] - df_all['Gemma_Answer_Int']) ** 2
# Normalization by the Expected Answer squared (to make it comparable across different numbers)
df_all['Normalized_MSE'] = df_all['Squared_Error'] / (df_all['Expected_Answer'] ** 2)

## We don't want to group by NumAds here and the first digit being added
digit_added = df_all.index.get_level_values('Number').to_list()

plt.figure(figsize=(10, 6))

for each_digit in set(digit_added):
    plot_data = df_all.loc[each_digit]
    plot_data = plot_data.groupby('Num_Adds')['Normalized_MSE'].mean().reset_index()

    # print(f"Plot data for {each_digit}:", plot_data)
    lstyle = '-' if each_digit in semantic_association else '-.'
    mksize = 10 if each_digit in semantic_association else 2
    marker = 'o' if each_digit in semantic_association else 's'

    plt.plot(plot_data['Num_Adds'], plot_data['Normalized_MSE']+1e-16, marker=marker, markersize=mksize, linestyle=lstyle, label=f'{each_digit}')

plt.legend()
plt.title('Normalized MSE vs. Sequence Length')
plt.xlabel('Number of Additions (Sequence Length Proxy)')
plt.ylabel('Average Normalized Mean Squared Error')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log') # Use log scale for Y-axis to better visualize large errors
plt.tight_layout()
plt.draw();
plt.savefig('exp1_normalized_mses.png')

# %%
