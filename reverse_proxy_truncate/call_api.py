import os.path
import sys
from datetime import timedelta
from itertools import repeat
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
import tiktoken
import tqdm
from diskcache import Cache
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

cache_dir = "cache"
cache = Cache(cache_dir)
_tokenizer = tiktoken.get_encoding("o200k_base")

A_DAY_IN_SECONDS = timedelta(days=1).seconds


# @memoize_stampede(cache, expire=A_DAY_IN_SECONDS)
@cache.memoize(expire=A_DAY_IN_SECONDS)
def get_number_of_tokens(some_string) -> int:
    tokenized_string = _tokenizer.encode(some_string)
    return len(tokenized_string)


@cache.memoize(expire=A_DAY_IN_SECONDS)
def get_printable_unicode() -> List[str]:
    """
    Return a list of all printable Unicode characters.

    Skips surrogates and non-characters. Uses str.isprintable()
    to filter.
    """
    return [
        chr(cp)
        for cp in range(sys.maxunicode + 1)
        if chr(cp).isprintable()
    ]


# @memoize_stampede(cache, expire=A_DAY_IN_SECONDS)
@cache.memoize(expire=A_DAY_IN_SECONDS)
def find_characters():
    all_chars = get_printable_unicode()
    with Pool(processes=cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(get_char_if_single_token, all_chars), total=len(all_chars)))
    arr = np.array(r, dtype=object)
    # noinspection PyComparisonWithNone
    cleaned = arr[arr != None].tolist()
    return cleaned


# @memoize_stampede(cache, expire=A_DAY_IN_SECONDS)
@cache.memoize(expire=A_DAY_IN_SECONDS)
def get_char_if_single_token(c):
    one_char_token = _tokenizer.encode(c)
    two_char_token = _tokenizer.encode(f"{c}{c}")
    if len(two_char_token) == 2 and one_char_token[0] == two_char_token[0] == two_char_token[1]:
        return c
    return None


TARGET_TOKEN_AMOUNTS = [
    90_000 + 1,
    90_000,
    90_000 - 1,
    (90_000 - 32_000) + 1,
    90_000 - 32_000,
    (90_000 - 32_000) - 1,
    32_000 + 1,
    32_000,
    32_000 - 1,
]

max_tokens = 32_000
model = "vllm_stelterlab_Qwen3-Coder-30B-A3B-Instruct-AWQ"
penalty = 1.05
temperature = 0.7
top_p = 0.8
base_url = "http://localhost:8123/v1"


def write_big_text_files(args):
    t, characters_that_are_tokenized_as_always_one_token = args
    filename = f"{t}.txt"
    how_many = len(characters_that_are_tokenized_as_always_one_token)
    how_many_times = t // how_many
    big_string = "".join(characters_that_are_tokenized_as_always_one_token) * how_many_times
    remainder = t - len(big_string)
    big_string += "".join(characters_that_are_tokenized_as_always_one_token[:remainder])
    while True:
        length_in_tokens = get_number_of_tokens(big_string)
        diff = int(t - length_in_tokens)
        if diff == 0:
            break
        if diff < 0:
            big_string = big_string[:-diff]
        elif diff < len(characters_that_are_tokenized_as_always_one_token):
            big_string += "".join(characters_that_are_tokenized_as_always_one_token[:diff])
        else:
            big_string += "".join(characters_that_are_tokenized_as_always_one_token)
        pass

    print(f"Generated a string of {length_in_tokens} tokens, was trying to get: {t}")
    with open(filename, mode="w") as f:
        f.write(big_string)


def maybe_generate_test_files():
    try:
        print("testing existence of test files...")
        for t in TARGET_TOKEN_AMOUNTS:
            if not os.path.exists(f"{t}.txt"):
                raise FileNotFoundError("Will generate numbers...")
        print("have already generated all large token test files, will skip re-creation")
    except Exception as e:
        print(e)
        characters_that_are_tokenized_as_always_one_token = find_characters()
        with Pool(processes=cpu_count()) as p:
            list(
                tqdm.tqdm(
                    p.imap(
                        write_big_text_files,
                        zip(TARGET_TOKEN_AMOUNTS, repeat(characters_that_are_tokenized_as_always_one_token))
                    ), total=len(TARGET_TOKEN_AMOUNTS)
                )
            )
        print("done generating test files")


def test_for_token_amount(t):
    client = OpenAI(base_url=base_url, api_key="asdf")
    with open(f"{t}.txt") as f:
        prompt = f.read()
    prompt_number_of_tokens = get_number_of_tokens(prompt)
    print(f"Testing for {prompt_number_of_tokens} tokens...")
    message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": prompt,
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        presence_penalty=penalty, frequency_penalty=penalty,
        seed=0, temperature=temperature, top_p=top_p,

    )
    response_text = response.choices[0].message.content
    response_number_of_tokens = get_number_of_tokens(response_text)
    print(
        f"Send request for {prompt_number_of_tokens} tokens, "
        f"got back {response_number_of_tokens} tokens"
    )
    pass


def call_the_api():
    print("Will call the API with test files...")
    sorted_easiest_to_hardest = sorted(TARGET_TOKEN_AMOUNTS)
    for t in sorted_easiest_to_hardest:
        test_for_token_amount(t)


def main():
    maybe_generate_test_files()
    call_the_api()


if __name__ == '__main__':
    main()
