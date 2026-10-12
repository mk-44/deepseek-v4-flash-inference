"""Phase 2, Step 23: tokenizer, official chat encoding, and generate for prompts of different lengths (plan item 8).

A. tokenizer + chat encoding: the 4 official example conversations (reference/official/0731/encoding/tests) come back
   unchanged after encode -> decode; 'What is 2+2?' gives the expected 11 ids; chat mode ends with </think>, thinking
   mode with <think>; decode_completion cuts at end-of-text, drops the -1 fillers, ends with exactly one end-of-text
   (the official convention, which parse_message_from_completion_text requires); the official encoding's own tests pass.
B. generate vs the official generate() of 0731 inference/generate.py (greedy, tiny model, fresh official model per case):
   prompts of different lengths, a row that stops early, one prompt that stops (the early break), a prompt that hits
   max_seq_len, and end-of-text inside a prompt (as after an earlier assistant turn), which must not count as finished.
"""
import importlib
import sys
from _paths import ROOT, ENCODING, load_notebook_defs
import torch

sys.path.insert(0, str(ROOT / 'reference'))
from load_official import load_official, official_args, official_generate

EXPECTED_IDS = [0, 128803, 3085, 344, 223, 20, 13, 20, 33, 128804, 128822]


def run(U):
    chat = U['ChatTokenizer'](str(ROOT / 'reference' / 'official' / '0731' / 'tokenizer.json'))
    parse = U['parse_message_from_completion_text']

    print('A. tokenizer + official chat encoding')
    same = [chat.tokenizer.decode(chat.tokenizer.encode(text).ids, skip_special_tokens=False) == text
            for text in (open(ENCODING / 'tests' / f'test_output_{k}.txt').read() for k in (1, 2, 3, 4))]
    print(f'  4 official example conversations unchanged after encode -> decode: {same}')
    messages = [{'role': 'user', 'content': 'What is 2+2?'}]
    ids = chat.encode_chat(messages)
    print(f"  'What is 2+2?' -> {ids == EXPECTED_IDS} ({len(ids)} ids)   last id: chat {ids[-1]} (</think> 128822), "
          f"thinking {chat.encode_chat(messages, 'thinking')[-1]} (<think> 128821)   end-of-text id {chat.eos_id}")
    reply = chat.tokenizer.encode('2 + 2 = 4.').ids
    for row in (reply + [chat.eos_id, 77, 88, -1, -1], reply + [-1, -1]):
        text = chat.decode_completion(row)
        print(f'  decode_completion -> {text!r}  correct {text == "2 + 2 = 4.<｜end▁of▁sentence｜>"}  '
              f'parse content {parse(text, "chat")["content"]!r}')
    tests = importlib.import_module('test_encoding_dsv4')
    for k in (1, 2, 3, 4):
        getattr(tests, f'test_case_{k}')()

    print('B. generate vs the official generate (greedy, tiny model, fresh official model per case)')
    off = load_official('0731')
    their_generate = official_generate(off)
    args = U['tiny_args']()
    torch.manual_seed(1)
    prompts = [torch.randint(0, args.vocab_size, (n,)).tolist() for n in (11, 12, 21)]

    def case(batch, eos_id):
        torch.manual_seed(0)
        model = U['Transformer'](args)
        theirs = off.Transformer(official_args(off, args, max_batch_size=len(batch))).float()
        theirs.load_state_dict(model.state_dict(), strict=True)
        theirs.temperature = 0
        ours = U['generate'](model, args, batch, 20, eos_id=eos_id)
        ref = their_generate(theirs, batch, 20, eos_id)
        cut = [[t for t in row if t >= 0] for row in ours.tolist()]
        cut = [(row[: row.index(eos_id)] if eos_id in row else row) + [eos_id] for row in cut]
        return ours, cut == ref, [len(r) for r in ref]

    free, same, lengths = case(prompts, args.vocab_size)
    print(f'  {"lengths 11, 12, 21, no stop":34s} identical {same}  completion lengths {lengths}')
    stop = free[0, 2].item()
    for label, batch, eos_id in (('row 0 stops at its 5th token', prompts, free[0, 4].item()),
                                 ('one prompt stops at its 7th token', prompts[2:], free[2, 6].item()),
                                 ('253-token prompt hits max_seq_len', [prompts[0] * 23], args.vocab_size),
                                 ('end-of-text inside a prompt', [prompts[0], prompts[2][:15] + [stop] + prompts[2][16:]], stop)):
        _, same, lengths = case(batch, eos_id)
        print(f'  {label:34s} identical {same}  completion lengths {lengths}')


if __name__ == '__main__':
    U = load_notebook_defs()
    if 'ChatTokenizer' not in U:
        print('the notebook has no tokenizer / chat encoding yet (Step 23)')
    else:
        run(U)
