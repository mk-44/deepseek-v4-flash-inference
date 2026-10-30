import http.client
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
from arch import ChatTokenizer

ROWS_API = "https://datasets-server.huggingface.co/rows" # HF row api.
MAX_TOKENS = 3_000 # one prefill pass max
CHAT = ChatTokenizer("deepseek_official/0731/tokenizer.json") # the tokenizer

def fetch(dataset : str, config : str, split : str, offset : int, length : int) -> dict:
    url = ROWS_API + "?" + urllib.parse.urlencode({"dataset" : dataset, "config" : config, "split" : split, "offset" : offset, "length" : length})
    
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout = 120) as r:
                return json.load(r)
        except (OSError, http.client.HTTPException, ValueError) as e:
            error = e
            time.sleep(30)
    
    raise RuntimeError(f"{dataset} {config}/{split} at row {offset}: {error}")


def pages(dataset : str, config : str, split : str, stride : int, length : int = 100, first : int = 0):
    offset = first
    
    while True:
        rows = [r["row"] for r in fetch(dataset, config, split, offset, length)["rows"]]
        if not rows:
            return
        yield offset, rows
        offset += stride
        time.sleep(1)
        

def user(text : str) -> dict:
    return {"role" : "user", "content" : text}

def assistant(text : str) -> dict:
    return {"role" : "assistant", "content" : text}


def encode(messages : list[dict]) -> tuple[list[int], list[list]]:
    ids, spans = [], []
    
    for i, m in enumerate(messages):
        longer = CHAT.encode_chat(messages[: i + 1])
        assert longer[: len(ids)] == ids, "adding a turn changed the tokens before it"
        spans.append([m["role"], len(ids), len(longer)])
        ids = longer
    
    return ids, spans


def gsm8k(): # math: grade-school word problems, solutions written by people
    for offset, rows in pages("openai/gsm8k", "main", "train", stride = 700):
        for i, row in enumerate(rows[: 2]):
            answer = re.sub(r"<<[^>]*>>", "", row["answer"]).replace("#### ", "Answer: ")
            yield offset + i, [user(row["question"]), assistant(answer)]


LONG = 2_700 # tokens in a long prompt
LENGTHS = (300, 600, 1_200, 2_400) # tokens of an answer cut from a document

def cut(text : str, tokens : int) -> str: # the whole lines from the start that fit in this many tokens
    out, n = [], 0
    for line in text[: 10 * tokens].split("\n"): # 10 characters per token taken here.
        n += len(CHAT.tokenizer.encode(line + "\n").ids)
        if n > tokens:
            break
        out.append(line)
    return "\n".join(out)


def continued(text : str, request : str, tokens : int) -> list[dict]: # document starts as  prompt and follows as the answer.
    head = cut(text, 150)
    return [user(f"{request}\n\n{head}"), assistant(cut(text[len(head) :].lstrip("\n"), tokens))]


def first(rows : list[dict], n : int = 2, keep = lambda row : True) -> list[tuple[int, dict]]: # a page's first n rows that pass
    return [(i, row) for i, row in enumerate(rows) if keep(row)][: n]


def untokenize(text : str) -> str: # texts are split into words: "is n't" -> "isn't", "`` Hi . ''" -> "\"Hi.\""
    text = re.sub(r" *\n *", "\n", text.replace("`` ", '"').replace(" ''", '"'))
    return re.sub(r" (n't|'\w+|[.,!?;:])", r"\1", text)

def pick(kind : str, shape : str, n : int, source : str, candidates) -> list[dict]:
    texts = []
    
    for row, messages in candidates():
        ids, spans = encode(messages)
        if len(ids) > MAX_TOKENS:
            continue
        texts.append(
            {"id" : f"{kind}-{candidates.__name__}-{len(texts):02d}", 
             "kind" : kind, 
             "shape" : shape, 
             "source" : source, 
             "row" : row,
            "split" : "test" if len(texts) % 4 == 3 else "learn", 
             "messages" : messages, 
             "ids" : ids, 
             "spans" : spans}
        )
        
        if len(texts) == n:
            return texts
    
    raise RuntimeError(f"{source}: only {len(texts)} of {n} texts found")
    

    
def wildchat(): # chat: real users' conversations with ChatGPT, English, not flagged toxic
    for offset, rows in pages("allenai/WildChat-1M", "default", "train", stride = 50_000):
        for i, row in first(rows, keep = lambda r : r["language"] == "English" and not r["toxic"]):
            yield offset + i, [{"role" : t["role"], "content" : t["content"]} for t in row["conversation"]]


def codefeedback(): # code: programming questions and answers
    for offset, rows in pages("m-a-p/CodeFeedback-Filtered-Instruction", "default", "train", stride = 9_000):
        for i, row in first(rows):
            yield offset + i, [user(row["query"]), assistant(row["answer"])]


def openr1(): # math: competition problems, answered by DeepSeek-R1 (the answer after its thinking)
    for offset, rows in pages("open-r1/OpenR1-Math-220k", "default", "train", stride = 9_000):
        for i, row in first(rows):
            right = [g for g, ok in zip(row["generations"], row["correctness_math_verify"]) if ok and "</think>" in g]
            if right:
                yield offset + i, [user(row["problem"]), assistant(right[0].split("</think>")[-1].strip())]


def wiki_article(): # science/encyclopedia: an article written on request
    for offset, rows in pages("wikimedia/wikipedia", "20231101.en", "train", stride = 300_000, length = 20):
        for i, row in first(rows, keep = lambda r : len(r["text"]) > 3_000):
            yield offset + i, [user(f"Write an encyclopedia article about {row['title']}."), assistant(cut(row["text"], LENGTHS[(offset + i) % 4]))]


def ledgar(): # legal: a contract clause written on request
    names = next(f for f in fetch("coastalcph/lex_glue", "ledgar", "train", 0, 1)["features"] if f["name"] == "label")["type"]["names"]
    for offset, rows in pages("coastalcph/lex_glue", "ledgar", "train", stride = 7_000):
        for i, row in first(rows):
            yield offset + i, [user(f'Write the "{names[row["label"]]}" clause of a contract.'), assistant(row["text"])]
            
def scotus(): # legal: the rest of a US Supreme Court opinion, from its start
    for offset, rows in pages("coastalcph/lex_glue", "scotus", "train", stride = 600, length = 10):
        for i, row in first(rows):
            yield offset + i, continued(row["text"], "Continue this US Supreme Court opinion:", LENGTHS[(offset + i) % 4])


def writingprompts(): # stories: a short story written to a prompt, by people
    for offset, rows in pages("euclaise/writingprompts", "default", "train", stride = 15_000):
        for i, row in first(rows):
            prompt = re.sub(r"^\[ \w+ \] ", "", row["prompt"]) # drops the tag, e.g. "[ WP ] "
            yield offset + i, [user(untokenize(prompt)), assistant(untokenize(row["story"]))]


def news(): # news: the rest of a news article, from its start
    for offset, rows in pages("allenai/c4", "realnewslike", "train", stride = 100_000):
        for i, row in first(rows, keep = lambda r : len(r["text"]) > 2_500):
            yield offset + i, continued(row["text"], "Continue this news article:", LENGTHS[(offset + i) % 4])


def anudesh(): # Hindi: real users' prompts in Hindi, answered by Llama-2-70B
    for offset, rows in pages("ai4bharat/indic-instruct-data-v0.1", "anudesh", "hi", stride = 900):
        for i, row in first(rows):
            yield offset + i, [{"role" : m["role"], "content" : m["content"]} for m in row["messages"]]


def hinglish(): # Hinglish: two people chatting about a movie, Hindi in Latin letters mixed with English
    for offset, rows in pages("festvox/cmu_hinglish_dog", "default", "train", stride = 1_000):
        date = rows[len(rows) // 2]["date"] # the chat in the middle of the page
        turns = []
        for row in rows:
            if row["date"] == date:
                role, text = "user" if row["uid"] == "user1" else "assistant", row["translation"]["hi_en"]
                if turns and turns[-1]["role"] == role:
                    turns[-1]["content"] += "\n" + text # one person's messages in a row make one turn
                else:
                    turns.append({"role" : role, "content" : text})
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        while turns and turns[-1]["role"] != "assistant":
            turns.pop()
        yield offset, turns


def torch_code(): # code: a long Python file of the box's own PyTorch, to explain
    folder = Path(torch.__file__).parent
    for name in ("nn/modules/module.py", "optim/adam.py", "utils/data/dataloader.py", "serialization.py", "nn/functional.py"):
        yield name, [user("Explain this code:\n\n" + cut((folder / name).read_text(), LONG))]


def wiki_long(): # science/encyclopedia: a long article to summarize
    for offset, rows in pages("wikimedia/wikipedia", "20231101.en", "train", stride = 400_000, length = 50, first = 150_000):
        for i, row in first(rows, n = 1, keep = lambda r : len(r["text"]) > 20_000):
            yield offset + i, [user("Summarize this article:\n\n" + cut(row["text"], LONG))]


def scotus_long(): # legal: a long court opinion to summarize
    for offset, rows in pages("coastalcph/lex_glue", "scotus", "train", stride = 1_000, length = 20, first = 300):
        for i, row in first(rows, n = 1, keep = lambda r : len(r["text"]) > 20_000):
            yield offset + i, [user("Summarize this court opinion:\n\n" + cut(row["text"], LONG))]


def gutenberg(): # stories: a passage from the middle of a public-domain book, to summarize
    for offset, rows in pages("manu/project_gutenberg", "default", "en", stride = 12_000, length = 1):
        text = rows[0]["text"]
        start = text.find("\n\n", len(text) // 3) # a paragraph start a third of the way in: past the front matter
        if len(text) > 60_000 and start > 0:
            yield offset, [user("Summarize this passage:\n\n" + cut(text[start :].strip(), LONG))]


def news_long(): # news: several news articles to summarize together
    for offset, rows in pages("allenai/c4", "realnewslike", "train", stride = 400_000, first = 50_000):
        yield offset, [user("Summarize these news articles:\n\n" + cut("\n\n---\n\n".join(r["text"] for r in rows), LONG))]


def hindi_long(): # Hindi: a long Hindi Wikipedia article to summarize
    for offset, rows in pages("wikimedia/wikipedia", "20231101.hi", "train", stride = 20_000, length = 20):
        for i, row in first(rows, n = 1, keep = lambda r : len(r["text"]) > 8_000):
            yield offset + i, [user("इस लेख का सारांश लिखिए:\n\n" + cut(row["text"], LONG))]


SOURCES = [
    ("chat", "qa", 16, "allenai/WildChat-1M", wildchat),
    ("code", "qa", 16, "m-a-p/CodeFeedback-Filtered-Instruction", codefeedback),
    ("math", "qa", 8, "openai/gsm8k", gsm8k),
    ("math", "qa", 8, "open-r1/OpenR1-Math-220k", openr1),
    ("science", "qa", 16, "wikimedia/wikipedia 20231101.en", wiki_article),
    ("legal", "qa", 8, "coastalcph/lex_glue ledgar", ledgar),
    ("legal", "qa", 8, "coastalcph/lex_glue scotus", scotus),
    ("stories", "qa", 16, "euclaise/writingprompts", writingprompts),
    ("news", "qa", 16, "allenai/c4 realnewslike", news),
    ("hindi", "qa", 8, "ai4bharat/indic-instruct-data-v0.1 anudesh", anudesh),
    ("hindi", "qa", 8, "festvox/cmu_hinglish_dog", hinglish),
    ("code", "long", 4, "torch (the installed package)", torch_code),
    ("science", "long", 4, "wikimedia/wikipedia 20231101.en", wiki_long),
    ("legal", "long", 4, "coastalcph/lex_glue scotus", scotus_long),
    ("stories", "long", 4, "manu/project_gutenberg", gutenberg),
    ("news", "long", 4, "allenai/c4 realnewslike", news_long),
    ("hindi", "long", 4, "wikimedia/wikipedia 20231101.hi", hindi_long),
]
            
            

if __name__ == "__main__":
    texts = []
    for kind, shape, n, source, candidates in SOURCES:
        t = time.perf_counter()
        picked = pick(kind, shape, n, source, candidates)
        texts += picked
        lengths = sorted(len(x["ids"]) for x in picked)
        answer = sum(end - start for x in picked for role, start, end in x["spans"] if role == "assistant")
        print(f"{kind:<8} {shape:<4} {candidates.__name__:<14} {n:3d} texts in {time.perf_counter() - t:4.0f} s | tokens {sum(lengths):7,}"
              f" (answers {answer / sum(lengths):4.0%}) | per text: min {lengths[0]:5,}, median {lengths[n // 2]:5,}, max {lengths[-1]:5,}")
        for role, start, end in picked[0]["spans"][: 2]: # the first text's first turns, as the model reads them
            print(f"    {role} ({end - start} tokens): {CHAT.tokenizer.decode(picked[0]['ids'][start : end], skip_special_tokens = False)[: 110]!r}")

    assert len({(x["source"], x["row"]) for x in texts}) == len(texts), "the same row twice"
    assert len({tuple(x["ids"]) for x in texts}) == len(texts), "the same text twice"
    learn = sum(x["split"] == "learn" for x in texts)
    print(f"{len(texts)} texts, {sum(len(x['ids']) for x in texts):,} tokens | learn {learn}, test {len(texts) - learn}")

    Path("corpus").mkdir(exist_ok = True)
    with open("corpus/texts.jsonl", "w") as f: # the texts and their token ids: stay on the box
        for x in texts:
            f.write(json.dumps(x, ensure_ascii = False) + "\n")
    index = [{k : v for k, v in x.items() if k not in ("messages", "ids")} | {"tokens" : len(x["ids"])} for x in texts]
    json.dump({"max_tokens" : MAX_TOKENS, "texts" : index}, open("results/corpus_index.json", "w"), indent = 1) # no text: goes to git
    print("saved -> corpus/texts.jsonl, results/corpus_index.json")
