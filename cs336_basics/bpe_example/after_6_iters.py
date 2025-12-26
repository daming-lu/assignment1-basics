import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = r"[ \t\n]+"

FILE_LOC = './corpus.txt'
MAX_ITER = 2
MAX_ITER = 6


vocab = {i: bytes([i]) for i in range(256)}

# add special tokens
special_tokens = ["<|endoftext|>"]
for tok in special_tokens:
    vocab[len(vocab)] = tok.encode("utf-8")
# print(vocab)
# print("\n")
with open(FILE_LOC, 'r') as f:
    lines = f.readlines()
    # lines = [re.findall(PAT, line) for line in lines]
    lines = [re.split(PAT, line) for line in lines]
    # print(lines)
    # print("\n")
    
    dict_tuple_bytes_to_int = {}
    for line in lines:
        for token in line:
            # token_bytes = token.encode("utf-8")

            cur_tuple = tuple([x.encode('utf-8') for x in token])
            if len(cur_tuple) == 0:
                # print('empty token: ', token)
                continue            
            if cur_tuple not in dict_tuple_bytes_to_int:
                dict_tuple_bytes_to_int[cur_tuple] = 1
            else:
                dict_tuple_bytes_to_int[cur_tuple] += 1
    print("init dict_tuple_bytes_to_int: \n")
    print(dict_tuple_bytes_to_int)    
    
    print("\n")
    
    for i in range(0, MAX_ITER):
        # import pdb;pdb.set_trace()
        dict_pair_bytes_to_int = {}
        print(f"\niter {i}")    
        for token_bytes, freq in dict_tuple_bytes_to_int.items():
            if len(token_bytes) == 1:
                continue
            for i in range(0, len(token_bytes) - 1):
                pair = token_bytes[i:i+2]
                if pair not in dict_pair_bytes_to_int:
                    dict_pair_bytes_to_int[pair] = freq
                else:
                    dict_pair_bytes_to_int[pair] += freq
        print(dict_pair_bytes_to_int)
        print("\n")        
        if len(dict_pair_bytes_to_int) == 0:
            break
        max_value = max(dict_pair_bytes_to_int.values())  # Returns 9
        max_pairs = []
        for pair, freq in dict_pair_bytes_to_int.items():
            if freq == max_value:
                max_pairs.append(pair)
        # print('max_pairs: ', max_pairs)
        sorted_pairs = sorted(max_pairs, reverse=True)
        max_pair = sorted_pairs[0]
        print(f'max_pair: {max_pair}, count: {max_value}')
        vocab[len(vocab)] = max_pair
        # replace all max_pair in dict_tuple_bytes_to_int
        keys_to_rm = []
        keys_val_to_add = {}
        for token_bytes, freq in dict_tuple_bytes_to_int.items():
            # print("token_bytes: ", token_bytes)
            new_token_bytes = []
            i = 0
            while i < len(token_bytes):
                if i + 1 < len(token_bytes):
                    pair = (token_bytes[i], token_bytes[i+1])
                    if pair == max_pair:
                        # print("pair: ", pair)
                        merged = max_pair[0] + max_pair[1]  # Merge b's' + b't' = b'st'
                        new_token_bytes.append(merged)
                        i += 2  # Skip both elements
                        continue
                
                new_token_bytes.append(token_bytes[i])
                i += 1            
            # replace
            new_token_bytes_tuple = tuple(new_token_bytes)
            if new_token_bytes_tuple != token_bytes:
                # dict_tuple_bytes_to_int[new_token_bytes_tuple] = freq
                # del dict_tuple_bytes_to_int[token_bytes]
                keys_val_to_add[new_token_bytes_tuple] = freq
                keys_to_rm.append(token_bytes)

        # import pdb;pdb.set_trace()
        for key in keys_to_rm:
            del dict_tuple_bytes_to_int[key]
        
        dict_tuple_bytes_to_int.update(keys_val_to_add)

        print("dict_tuple_bytes_to_int: \n")
        print(dict_tuple_bytes_to_int)
    # end of ITER
    print("\n")
    print(vocab)

print("\n\n--------------\n\n")

def flatten_token(token):
    """Convert a token (bytes or tuple) into a single bytes object"""
    if isinstance(token, tuple):
        return b''.join(flatten_token(t) for t in token)
    return token

clean_vocab = {
    k:flatten_token(v) for k, v in vocab.items()
}
import pdb;pdb.set_trace()
print('clean_vocab: ', clean_vocab)
reversed_vocab = {
    v: k for k, v in clean_vocab.items()
}
# encode newest -> ne, west
input1 = 'newest'
input1_enc = [x.encode('utf-8') for x in input1]
print(f'input1_enc: {input1_enc}')
# import pdb;pdb.set_trace()
chosen_type = "TRAE"
chosen_type = "claude"
if chosen_type == "TRAE":

    while True:
        found = False
        new_input1_enc = []
        i = 0
        while i < len(input1_enc):
            left = input1_enc[i] if isinstance(input1_enc[i], bytes) else b"".join(input1_enc[i])
            if i + 1 < len(input1_enc):
                right = input1_enc[i+1] if isinstance(input1_enc[i+1], bytes) else b"".join(input1_enc[i+1])
                pair = left+right
                if pair in reversed_vocab:
                    new_input1_enc.append(left + right)
                    i += 2
                    continue
            new_input1_enc.append(left)
            i += 1
        if new_input1_enc == input1_enc:
            break
        input1_enc = new_input1_enc
        print(f'input1_enc: {input1_enc}')
        
    print(f'final input1_enc: {input1_enc}')    
elif chosen_type == "claude":
    while True:
        new_input1_enc = []
        i = 0
        found = False
        
        while i < len(input1_enc):
            if i + 1 < len(input1_enc):
                # Flatten both tokens to bytes, then create pair
                curr_flat = flatten_token(input1_enc[i])
                next_flat = flatten_token(input1_enc[i+1])
                pair = curr_flat + next_flat
                
                if pair in reversed_vocab:
                    # Store as tuple of the flattened bytes
                    new_input1_enc.append(pair)
                    found = True
                    i += 2
                    continue
            
            new_input1_enc.append(input1_enc[i])
            i += 1
        
        if not found:
            break
        input1_enc = new_input1_enc
        print(f'input1_enc: {input1_enc}')

    print(f'final input1_enc: {input1_enc}')

# map to ids and decode
token_ids = [reversed_vocab[tok] for tok in input1_enc]
print('token_ids: ', token_ids)
decoded_bytes = b''.join(clean_vocab[i] for i in token_ids)
print('decoded: ', decoded_bytes.decode('utf-8', errors='replace'))
