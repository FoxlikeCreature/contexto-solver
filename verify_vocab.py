#!/usr/bin/env python3
"""
Проверяет насколько наш локальный top_neighbors совпадает с API контекстно.рф.
Для каждого тестового слова сравниваем топ-50 из API и из модели.
"""
import json, urllib.request, urllib.parse, sys
from pathlib import Path

BASE       = Path.home() / ".local/share/claude"
MODEL_PATH = BASE / "models/tayga/model.bin"
for f in ["game_vocab_exact.json", "game_vocab_best.json", "game_vocab.json"]:
    VOCAB_PATH = BASE / f
    if VOCAB_PATH.exists(): break

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; YELLOW = "\033[93m"; RED = "\033[31m"; CYAN = "\033[36m"

def api_first_words(word, topn=100):
    url = f"https://api.contextno.com/first-words?word={urllib.parse.quote(word)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
    fw = data.get("first_words", [])
    # первое слово — сам запрос, пропускаем
    return [w for w in fw if w != word][:topn]

def load_model():
    from gensim.models import KeyedVectors
    print(f"{DIM}Загружаю модель...{RESET}", end=" ", flush=True)
    model = KeyedVectors.load_word2vec_format(str(MODEL_PATH), binary=True)
    with open(VOCAB_PATH) as f:
        vocab_set = set(json.load(f))
    print(f"готово  ({len(vocab_set)} слов в словаре){RESET}")
    return model, vocab_set

def local_top(model, vocab_set, word, topn=100):
    key = f"{word}_NOUN"
    if key not in model.key_to_index:
        return []
    result, seen = [], set()
    for w, _ in model.most_similar(key, topn=80000):
        if not w.endswith("_NOUN"): continue
        lemma = w[:-5]
        if lemma in seen or lemma == word: continue
        seen.add(lemma)
        if lemma in vocab_set:
            result.append(lemma)
            if len(result) >= topn: break
    return result

def compare(word, api_list, local_list, show_n=30):
    api_set   = set(api_list[:show_n])
    local_set = set(local_list[:show_n])
    both      = api_set & local_set
    only_api  = api_set - local_set
    only_loc  = local_set - api_set

    # Ранговая корреляция (Spearman) по топ-50
    n = min(50, len(api_list), len(local_list))
    api_rank  = {w: i for i, w in enumerate(api_list[:n])}
    loc_rank  = {w: i for i, w in enumerate(local_list[:n])}
    common    = set(api_rank) & set(loc_rank)
    if len(common) >= 3:
        import numpy as np
        r1 = [api_rank[w] for w in common]
        r2 = [loc_rank[w] for w in common]
        from scipy.stats import spearmanr
        rho, _ = spearmanr(r1, r2)
    else:
        rho = None

    pct = len(both) / show_n * 100

    print(f"\n{BOLD}{CYAN}━━ {word.upper()} ━━{RESET}")
    print(f"  Совпадений в топ-{show_n}: {len(both)}/{show_n}  ({pct:.0f}%)", end="")
    if rho is not None:
        print(f"   Spearman ρ={rho:.3f} (топ-{n})", end="")
    print()

    # Таблица: API слева, локальная справа
    print(f"  {'#':>3}  {'API':20s}  {'Локальная':20s}  {'статус'}")
    print(f"  {'─'*3}  {'─'*20}  {'─'*20}  {'─'*6}")
    for i in range(show_n):
        a = api_list[i]   if i < len(api_list)   else "—"
        l = local_list[i] if i < len(local_list) else "—"
        if a == l:
            mark = f"{GREEN}={RESET}"
        elif a in local_set and l in api_set:
            mark = f"{YELLOW}~{RESET}"
        elif a in local_set:
            mark = f"{YELLOW}a{RESET}"
        elif l in api_set:
            mark = f"{YELLOW}l{RESET}"
        else:
            mark = f"{RED}✗{RESET}"
        ac = GREEN if a in local_set else RESET
        lc = GREEN if l in api_set  else RESET
        print(f"  {i+1:>3}  {ac}{a:20s}{RESET}  {lc}{l:20s}{RESET}  {mark}")

    # Только в API
    if only_api:
        print(f"\n  {RED}Только в API (нет у нас): {', '.join(sorted(only_api))}{RESET}")
    # Только локально
    if only_loc:
        print(f"  {YELLOW}Только у нас (нет в API): {', '.join(sorted(only_loc))}{RESET}")

    return pct, rho

TEST_WORDS = ["кот", "дом", "война", "вода", "дерево"]

def main():
    model, vocab_set = load_model()

    total_pcts = []
    for word in TEST_WORDS:
        try:
            api   = api_first_words(word, 50)
            local = local_top(model, vocab_set, word, 50)
            pct, rho = compare(word, api, local, show_n=30)
            total_pcts.append(pct)
        except Exception as e:
            print(f"\n{RED}Ошибка для '{word}': {e}{RESET}")

    if total_pcts:
        avg = sum(total_pcts) / len(total_pcts)
        col = GREEN if avg >= 80 else (YELLOW if avg >= 60 else RED)
        print(f"\n{BOLD}Средний процент совпадений: {col}{avg:.0f}%{RESET}")
        if avg < 90:
            print(f"{YELLOW}Словарь/модель не идеальны — нужно доработать{RESET}")
        else:
            print(f"{GREEN}Словарь совпадает хорошо!{RESET}")

if __name__ == "__main__":
    main()
