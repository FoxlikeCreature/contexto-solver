#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(line_buffering=True)
"""
Строит точный словарь игры контекстно.рф через API.
Проверяет каждое слово из tayga модели — есть оно в игре или нет.
Поддерживает прерывание и продолжение с того же места.

Запуск:  python3 build_vocab.py
"""

import json, urllib.request, urllib.parse, time, sys, os, signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BASE_DIR   = Path.home() / ".local/share/claude"
FULL_FILE  = BASE_DIR / "tayga_nouns_to_check.json"  # все NOUN из tayga не проверенные
EXACT_FILE = BASE_DIR / "game_vocab_exact.json"       # итоговый точный словарь
PROG_FILE  = BASE_DIR / "vocab_progress.json"         # прогресс для resume

BASE_API   = "https://api.contextno.com"
ALPHA_START = ["абажур", "аббат", "аббатство"]
THREADS    = 5
DELAY      = 0.07   # сек между запросами внутри потока

# ── Глобальное состояние ──────────────────────────────────────────────────────
lock         = threading.Lock()
in_game      = set()
not_in_game  = set()
errors       = set()
stop_flag    = False

def save_progress():
    with lock:
        data = {
            "in_game":     sorted(in_game),
            "not_in_game": sorted(not_in_game),
            "errors":      sorted(errors),
        }
    with open(PROG_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False)

def handle_sigint(sig, frame):
    global stop_flag
    print("\n\nПрерывание... сохраняю прогресс.")
    stop_flag = True
    save_progress()
    print(f"Сохранено в {PROG_FILE}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# ── Проверка одного слова ─────────────────────────────────────────────────────

def check_word(word):
    try:
        url = f"{BASE_API}/first-words?word={urllib.parse.quote(word)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            fw = json.loads(r.read().decode()).get("first_words", [])
        if not fw:
            return word, False
        if fw[:3] == ALPHA_START:
            return word, False
        if fw[0] != word:
            return word, False
        return word, True
    except Exception:
        return word, None

def worker(words):
    results = []
    for w in words:
        if stop_flag:
            break
        results.append(check_word(w))
        time.sleep(DELAY)
    return results

# ── Прогресс-бар ──────────────────────────────────────────────────────────────

def progress_bar(done, total, width=40):
    pct  = done / total
    fill = int(pct * width)
    bar  = "█" * fill + "░" * (width - fill)
    return f"[{bar}] {done}/{total} ({pct*100:.1f}%)"

# ── Главная логика ────────────────────────────────────────────────────────────

def main():
    global in_game, not_in_game, errors

    with open(FULL_FILE) as f:
        full_vocab = json.load(f)

    # Загружаем прогресс если есть
    if PROG_FILE.exists():
        with open(PROG_FILE) as f:
            prog = json.load(f)
        in_game     = set(prog.get("in_game", []))
        not_in_game = set(prog.get("not_in_game", []))
        errors      = set(prog.get("errors", []))
        print(f"Продолжаю с прогресса: {len(in_game)+len(not_in_game)+len(errors)} уже проверено")
    else:
        print("Начинаю с нуля")

    # Слова которые ещё нужно проверить
    already_done = in_game | not_in_game | errors
    to_check = [w for w in full_vocab if w not in already_done]

    total_left = len(to_check)
    total_all  = len(full_vocab)
    total_done = len(already_done)

    eta_sec = total_left / THREADS / (1 / DELAY) * 1.3  # ~30% на сеть
    print(f"Словарь: {total_all} слов")
    print(f"Осталось проверить: {total_left} ({total_done} уже готово)")
    print(f"ETA: ~{eta_sec/60:.0f} мин при {THREADS} потоках")
    print()

    if not to_check:
        print("Всё уже проверено!")
    else:
        # Маленькие батчи по 50 слов — прогресс каждые ~7с
        MINI_BATCH = 50
        chunks = [to_check[i:i+MINI_BATCH] for i in range(0, total_left, MINI_BATCH)]

        t_start   = time.time()
        completed = 0
        save_every = 500

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            for future in as_completed(
                executor.submit(worker, chunk) for chunk in chunks
            ):
                if stop_flag:
                    break
                for word, result in future.result():
                    with lock:
                        if result is True:
                            in_game.add(word)
                        elif result is False:
                            not_in_game.add(word)
                        else:
                            errors.add(word)
                        completed += 1

                # Прогресс после каждого мини-батча
                done_total = total_done + completed
                elapsed    = time.time() - t_start
                rate       = completed / elapsed if elapsed > 0 else 1
                eta        = (total_left - completed) / rate if rate > 0 else 0
                bar = progress_bar(done_total, total_all)
                print(f"{bar}  в_игре={len(in_game)}  ETA={eta/60:.1f}мин")

                if completed % save_every == 0:
                    save_progress()

        print()

    # Финальное сохранение
    save_progress()
    final = sorted(in_game)
    with open(EXACT_FILE, "w") as f:
        json.dump(final, f, ensure_ascii=False)

    print(f"\n✓ Точный словарь: {len(final)} слов → {EXACT_FILE}")
    print(f"  Не в игре:      {len(not_in_game)}")
    print(f"  Ошибки (None):  {len(errors)}")

if __name__ == "__main__":
    main()
