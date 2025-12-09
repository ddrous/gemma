#%%

# Common imports
import os
import jax
import jax.numpy as jnp

# Gemma imports
from gemma import gm

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
os.system("nvidia-smi")



#%%


model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)


tokenizer = gm.text.Gemma3Tokenizer()








#%%
# prompt = tokenizer.encode('One word to describe Paris: \n\n', add_bos=True)
# prompt = tokenizer.encode('please do this arithmetic: one plus nine is: \n\n', add_bos=True)
prompt = tokenizer.encode('1+11=', add_bos=True)
prompt = jnp.asarray(prompt)
print(prompt.shape)

# Run the model
out = model.apply(
    {'params': params},
    tokens=prompt,
    return_last_only=True,  # Only predict the last token
    return_hidden_states=True
)

# print(out)  # (1, seq_len, vocab_size)
# Sample a token from the predicted logits
next_token = jax.random.categorical(
    jax.random.key(1),
    out.logits
)
res = tokenizer.decode(next_token)
print(res)
# print(res)

# %%
tokenizer.plot_logits(out.logits)

# %%

## Please repeatedly call the model until we get an end-of-sequence token
# prompt = tokenizer.encode('Replace the interogation mark into a number: 1+11=?', add_bos=True)
prompt = tokenizer.encode('Compute then end with a ful stop: 11+11=', add_bos=True)
print(prompt)
prompt = jnp.asarray(prompt)

current_tokens = prompt
for iter_count in range(5):
    out = model.apply(
        {'params': params},
        tokens=current_tokens,
        return_last_only=True,  # Only predict the last token
        return_hidden_states=False
    )
    next_token = jax.random.categorical(
        jax.random.key(1),
        out.logits
    )
    res = tokenizer.decode(next_token)
    print(next_token, next_token.shape, res, end='', flush=True)
    current_tokens = jnp.concatenate([current_tokens, jnp.array([next_token])], axis=0)
    # if next_token == 1: ## EOS token
    if next_token == tokenizer.special_tokens.EOS: ## EOS token
        break

result = tokenizer.decode(current_tokens)
print(f"Full result after {iter_count} calls:", result)


# %%

# #Decode a full stop
# prompt = tokenizer.encode('.', add_bos=False)
# print(prompt)


## Repeatedly call the model until we get an end-of-sequence token or a full stop
# prompt_text = 'Compute then end with a ful stop: 11+11='
prompt_text = '11+11='
prompt = tokenizer.encode(prompt_text, add_bos=True)
print(f"\nPrompt text encoded: {prompt}")
current_tokens = jnp.asarray(prompt)


# Get the official EOS ID and the common Full Stop ID (for arithmetic)
EOS_ID = tokenizer.special_tokens.EOS
# Assuming '.' is tokenized as a single token for a robust check
FULL_STOP_ID = tokenizer.encode('.', add_bos=False)[0]
# Assuming your tokenizer has tokens for the digits '0' through '9'
DIGIT_TOKENS = [tokenizer.encode(str(i), add_bos=False)[0] for i in range(10)]

# Loop to generate new tokens
# Increased range for more generation capacity
MAX_ITERATIONS = 15
for iter_count in range(MAX_ITERATIONS):
    # 1. Apply the model to the current sequence
    out = model.apply(
        {'params': params},
        tokens=current_tokens,
        return_last_only=True,
        return_hidden_states=False
    )

    # 2. Sample the next token (returns a JAX array of shape (1,))
    next_token_array = jax.random.categorical(
        jax.random.key(iter_count + 1), # Use a different key for each step
        out.logits
    )

    print(f"Next token array: {next_token_array}")
    # Extract the scalar ID from the array for comparison
    next_token_id = jnp.array([next_token_array])

    # 3. Add the generated token to the sequence BEFORE checking the stop condition
    current_tokens = jnp.concatenate([current_tokens, jnp.array([next_token_array])], axis=0)

    # 4. Decode and print the token (strip() removes extra spaces added by decode)
    res = tokenizer.decode(next_token_array).strip()
    print(f"Iter {iter_count+1}: ID={next_token_id}, Token='{res}'")

    # # 5. Check stop conditions
    # # Check for the official EOS token, or a common full stop token
    # if next_token_id == EOS_ID:
    #     print("\n[STOP CONDITION MET: Official EOS Token Found]")
    #     break
    # if next_token_id == FULL_STOP_ID:
    #     print("\n[STOP CONDITION MET: Full Stop Token Found]")
    #     break

    # Check if the generated token is NOT a digit, and stop if so.
    # Note: You may need to add token IDs for the space or newline if they are desired.
    if next_token_id not in DIGIT_TOKENS:
        print("\n[STOP CONDITION MET: Non-digit token found]")
        # You may want to skip appending this non-digit token if you want a clean numerical result
        # break

        # To keep the punctuation (like the space before 'The'), but stop:
        if next_token_id != FULL_STOP_ID: # If it's a space or another token, we include it and stop
            current_tokens = jnp.concatenate([current_tokens, jnp.array([next_token_array])], axis=0)

        break


else: # Executes if the loop completes without 'break'
    print(f"\n[Generation stopped: Reached maximum iterations ({MAX_ITERATIONS})]")


# Decode the final result
result = tokenizer.decode(current_tokens)
print("---")
print(f"Prompt: {prompt_text}")
print(f"Full Result: {result}")
# print(f"Total tokens generated:% {current_tokens.shape[0] - prompt.shape[0]}")
