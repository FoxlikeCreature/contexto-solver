#!/usr/bin/env python3
"""
Контекстно.рф солвер — точная триангуляция.

Принцип: зная (слово W, ранг R), секрет — это одно из слов C
для которых rank(W от C) == R. Это вычисляется точно через
матричные операции, без аппроксимаций.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import re
import numpy as np
from pathlib import Path

PROJECT    = Path(__file__).parent
MODEL_PATH = PROJECT / "model/model.bin"
for f in ["game_vocab_exact.json", "game_vocab_best.json", "game_vocab.json"]:
    VOCAB_PATH = PROJECT / f
    if VOCAB_PATH.exists(): break

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; YELLOW = "\033[93m"; CYAN = "\033[36m"; RED = "\033[31m"

def rank_color(r):
    if r == 1:   return "\033[1;32m"
    if r <= 10:  return "\033[32m"
    if r <= 100: return "\033[33m"
    if r <= 500: return "\033[93m"
    return RESET

_vocab_list: list[str] = []
_vocab_set:  set[str]  = set()
_vocab_mat:  np.ndarray | None = None   # (N, dim) float32 normalized
_vocab_idx:  dict[str, int] = {}


def load():
    global _vocab_list, _vocab_set, _vocab_mat, _vocab_idx
    if _vocab_mat is not None:
        return
    from gensim.models import KeyedVectors
    print(f"{DIM}Загружаю модель...{RESET}", end=" ", flush=True)
    model = KeyedVectors.load_word2vec_format(str(MODEL_PATH), binary=True)

    with open(VOCAB_PATH) as f:
        raw = json.load(f)

    words, vecs = [], []
    for w in raw:
        key = f"{w}_NOUN"
        if key not in model.key_to_index:
            continue
        v = model[key].astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            words.append(w)
            vecs.append(v / n)

    _vocab_list[:] = words
    _vocab_set.update(words)
    _vocab_mat = np.vstack(vecs)                   # (N, dim)
    _vocab_idx.update({w: i for i, w in enumerate(words)})
    print(f"{DIM}готово ({len(_vocab_set)} слов){RESET}")


# ── Быстрый порядок по сходству ──────────────────────────────────────────────

def top_neighbors_fast(word: str, topn: int) -> list[str]:
    """Topn ближайших vocab-слов к word (быстро через dotprod, без самого word)."""
    if word not in _vocab_idx:
        return []
    w_vec  = _vocab_mat[_vocab_idx[word]]
    sims   = _vocab_mat @ w_vec             # (N,)
    order  = np.argsort(-sims)
    result = []
    for i in order:
        w = _vocab_list[i]
        if w == word:
            continue
        result.append(w)
        if len(result) >= topn:
            break
    return result


def sorted_by_similarity(ref_word: str, words: list[str]) -> list[str]:
    """Сортирует words по убыванию сходства с ref_word."""
    if ref_word not in _vocab_idx or not words:
        return words
    r_vec  = _vocab_mat[_vocab_idx[ref_word]]
    idxs   = np.array([_vocab_idx[w] for w in words if w in _vocab_idx])
    if len(idxs) == 0:
        return words
    sims   = _vocab_mat[idxs] @ r_vec
    order  = np.argsort(-sims)
    ordered_words = [w for w in words if w in _vocab_idx]
    return [ordered_words[i] for i in order]


# ── Ключевая функция ──────────────────────────────────────────────────────────

def rank_of_word_from(guess: str, candidates: list[str]) -> dict[str, int]:
    """
    Для каждого C из candidates вычисляет rank(guess от C):
    сколько слов из vocab-словаря имеют сходство с C >= sim(guess, C).

    Обрабатывает большие списки кандидатов пакетами чтобы не исчерпать память.
    """
    if guess not in _vocab_idx or not candidates:
        return {}

    g_vec = _vocab_mat[_vocab_idx[guess]]          # (dim,)
    valid_cands = [c for c in candidates if c in _vocab_idx]
    if not valid_cands:
        return {}

    BATCH = 2000
    result = {}
    for i in range(0, len(valid_cands), BATCH):
        batch = valid_cands[i:i+BATCH]
        c_mat = _vocab_mat[np.array([_vocab_idx[c] for c in batch])]  # (b, dim)
        sims  = _vocab_mat @ c_mat.T               # (N, b)
        sim_g = c_mat @ g_vec                      # (b,)
        ranks = (sims >= sim_g[np.newaxis, :]).sum(axis=0)
        for j, c in enumerate(batch):
            result[c] = int(ranks[j])

    return result


def candidates_for_guess(guess: str, game_rank: int, pool: list[str],
                          tolerance: int = 10) -> set[str]:
    """
    Секрет — это C ∈ pool, для которого rank(guess от C) ≈ game_rank.

    Для маленьких pool (<= 2000): считаем напрямую без предфильтра.
    Для больших: предфильтр с множителем 4x (было 2x — пропускал секрет
    при высоких рангах, т.к. asymmetry ratio может быть > 2).
    Если rank * 4 > len(pool): поиск по всему pool (батчами, без OOM).
    """
    lo = max(1, game_rank - tolerance)
    hi = game_rank + tolerance

    # Маленький pool — считаем напрямую (предфильтр не даёт выигрыша)
    if len(pool) <= 2000:
        ranks = rank_of_word_from(guess, pool)
        return {c for c, r in ranks.items() if lo <= r <= hi}

    # Предфильтр 4x: для ранга 9269 → 37276 > vocab_size → поиск по всему словарю
    prefilter_n = int(game_rank * 4) + 200
    if prefilter_n >= len(pool):
        # Покрываем весь pool — предфильтр не нужен
        ranks = rank_of_word_from(guess, pool)
        return {c for c, r in ranks.items() if lo <= r <= hi}

    prefilter = list(set(top_neighbors_fast(guess, prefilter_n)) & set(pool))
    ranks = rank_of_word_from(guess, prefilter)
    result = {c for c, r in ranks.items() if lo <= r <= hi}

    if not result:
        # Fallback: поиск по всему pool
        ranks2 = rank_of_word_from(guess, pool)
        result = {c for c, r in ranks2.items() if lo <= r <= hi}

    return result


# ── Seeds ─────────────────────────────────────────────────────────────────────

SEEDS = [
    "человек", "вещь", "животное", "природа", "место",
    "дом", "время", "вода", "деньги", "книга", "машина",
    "музыка", "история", "страна", "зима", "любовь", "война",
    "небо", "земля", "огонь", "дерево", "путь",
]


# ── Солвер ────────────────────────────────────────────────────────────────────

class Solver:
    """
    После каждой пары (слово, ранг) точно вычисляет множество слов C,
    для которых rank(слово от C) == ранг. Пересечение таких множеств
    быстро сходится к единственному слову — секрету.
    """

    def __init__(self):
        self.guesses:    dict[str, int] = {}
        self.candidates: set[str] | None = None
        self._tol = 10  # допуск ±10: игра даёт ранги на ~5-7 выше нашей модели

    def reset(self):
        self.guesses.clear()
        self.candidates = None
        self._tol = 3

    def _rebuild(self) -> None:
        """Пересобирает кандидатов с нуля по всем угадкам (от наименьшего ранга)."""
        pool = list(_vocab_set - set(self.guesses.keys()))
        valid = sorted(
            [(g, r) for g, r in self.guesses.items() if r > 0],
            key=lambda x: x[1]   # сначала наименьший ранг — самое специфичное слово
        )
        for tol in (self._tol, self._tol * 2, self._tol * 3):
            result = None
            for g, r in valid:
                search_pool = pool if result is None else list(result)
                c = candidates_for_guess(g, r, search_pool, tolerance=tol)
                c -= set(self.guesses.keys())
                result = c if result is None else (result & c)
                if not result:
                    break
            if result:
                self.candidates = result
                return

    def add_guess(self, word: str, rank: int):
        self.guesses[word] = rank
        if rank == 1:
            return

        pool = list(_vocab_set - set(self.guesses.keys()))

        if self.candidates is None:
            new_set = candidates_for_guess(word, rank, pool, tolerance=self._tol)
        else:
            new_set = candidates_for_guess(word, rank,
                                           list(self.candidates), tolerance=self._tol)

        new_set -= set(self.guesses.keys())

        if new_set:
            self.candidates = new_set
        elif self.candidates is not None:
            # Пустое пересечение — пересобираем с нуля по всем угадкам
            self._rebuild()

    def suggest(self) -> tuple[str, str]:
        tried = set(self.guesses.keys())

        if self.candidates is None or not self.guesses:
            return next((s for s in SEEDS if s not in tried), SEEDS[0]), "стартовое"

        if not self.candidates:
            return next((s for s in SEEDS if s not in tried), SEEDS[0]), "нет кандидатов"

        n = len(self.candidates)

        if n == 1:
            return list(self.candidates)[0], "единственный кандидат"

        # Лучшее известное слово (наименьший ранг) — ближайшее к секрету
        best_word = min(self.guesses, key=lambda w: self.guesses[w] if self.guesses[w] > 0 else 99999)

        # Сортируем кандидатов по близости к best_word и берём первого
        # (он наиболее вероятно ближе к секрету → наименьший ранг на следующем шаге)
        ordered = sorted_by_similarity(best_word, [c for c in self.candidates if c not in tried])
        if ordered:
            return ordered[0], f"{n} кандидатов"

        # Fallback
        for w in sorted(self.candidates):
            if w not in tried:
                return w, f"{n} кандидатов (fallback)"
        return next((s for s in SEEDS if s not in tried), SEEDS[0]), "нет вариантов"

    def info(self) -> str:
        if self.candidates is None:
            return ""
        return f"{len(self.candidates)} кандидатов"


# ── Интерактивный режим ───────────────────────────────────────────────────────

def _print_candidates(solver: "Solver"):
    if not solver.candidates:
        print(f"  {DIM}Кандидатов нет{RESET}")
        return
    cands = sorted(solver.candidates)
    if len(cands) <= 20:
        best = min(solver.guesses, key=lambda w: solver.guesses[w] if solver.guesses[w] > 0 else 99999) if solver.guesses else None
        if best:
            ordered = sorted_by_similarity(best, cands)
        else:
            ordered = cands
        for i, w in enumerate(ordered, 1):
            print(f"  {DIM}{i:>2}.{RESET} {w}")
    else:
        print(f"  {DIM}Кандидатов {len(cands)}: {', '.join(cands[:15])}...{RESET}")


def _parse_pairs(parts: list[str], cur_word: str) -> list[tuple[str, int | None]] | None:
    """
    Разбирает части ввода в список (слово, ранг) пар.
    ранг=None означает '?' (пропустить слово).
    Поддерживаемые форматы:
      1234             → [(cur_word, 1234)]
      слово 1234       → [(слово, 1234)]
      слово ?          → [(слово, None)]
      1234 слово 567   → [(cur_word, 1234), (слово, 567)]
      сл1 123 сл2 456  → [(сл1, 123), (сл2, 456)]
    Возвращает None если формат не распознан.
    """
    pairs: list[tuple[str, int | None]] = []
    i = 0

    # Если первый токен — число, это ранг для cur_word
    if parts and parts[0].isdigit():
        pairs.append((cur_word, int(parts[0])))
        i = 1

    # Дальше: пары слово ранг / слово ?
    while i < len(parts):
        if i + 1 >= len(parts):
            return None  # слово без ранга
        word_tok = parts[i]
        rank_tok = parts[i + 1]
        if word_tok.isdigit():
            return None  # ожидали слово, получили число
        if rank_tok.isdigit():
            pairs.append((word_tok, int(rank_tok)))
        elif rank_tok == "?":
            pairs.append((word_tok, None))
        else:
            return None
        i += 2

    return pairs or None


def _looks_like_telegram_start(line: str) -> bool:
    low = line.lower()
    return low.startswith("слово:") or low.startswith("близость:")


def _parse_telegram_paste(text: str) -> list[tuple[str, int]] | None:
    """
    Разбирает вставленное Telegram-сообщение из игры.

    Формат блока:
        Слово: стакан 🟡
        Близость: 322

        Топ ближайших слов:
        🟡 стакан (322)
        🔴 море (6208)
        ----------------

    Извлекает все пары (слово, ранг) из Слово/Близость строк
    и из emoji-строк. При дублях берёт минимальный ранг.
    """
    low = text.lower()
    if "близость:" not in low and "слово:" not in low:
        return None

    pairs: dict[str, int] = {}
    pending_word: str | None = None

    for line in text.splitlines():
        s = line.strip()
        sl = s.lower()

        # "Слово: стакан 🟡" → pending_word
        m = re.match(r'слово:\s*([а-яёa-z]+)', sl)
        if m:
            pending_word = m.group(1)
            continue

        # "Близость: 322" → closes the pair
        m = re.match(r'близость:\s*(\d+)', sl)
        if m and pending_word:
            rank = int(m.group(1))
            if pending_word not in pairs or rank < pairs[pending_word]:
                pairs[pending_word] = rank
            pending_word = None
            continue

        # Сброс pending если пришла нерелевантная строка
        if pending_word and sl and not sl.startswith("топ") and "--" not in sl:
            pending_word = None

        # "🟢/🟡/🟠/🔴 ананас (148)"
        m = re.search(r'[🟢🟡🟠🔴]\s*([а-яё]+)\s*\((\d+)\)', sl)
        if m:
            word = m.group(1)
            rank = int(m.group(2))
            if word not in pairs or rank < pairs[word]:
                pairs[word] = rank

    return list(pairs.items()) if pairs else None


def run():
    load()
    solver = Solver()

    print(f"\n{BOLD}{CYAN}контекстно.рф — солвер{RESET}  {DIM}({VOCAB_PATH.name}, {len(_vocab_set)} слов){RESET}")
    print(f"{DIM}Формат ввода:{RESET}")
    print(f"{DIM}  ранг                    → засчитать предложенное слово{RESET}")
    print(f"{DIM}  слово ранг              → засчитать своё слово{RESET}")
    print(f"{DIM}  сл1 р1 сл2 р2 ...      → несколько пар сразу{RESET}")
    print(f"{DIM}  слово ?                 → слово не в игре (пропустить){RESET}")
    print(f"{DIM}  ?                       → показать кандидатов{RESET}")
    print(f"{DIM}  n — новая игра  |  q — выход{RESET}\n")

    cur_word, cur_reason = solver.suggest()

    while True:
        # Если остался единственный кандидат — это и есть ответ
        if solver.candidates and len(solver.candidates) == 1:
            secret = list(solver.candidates)[0]
            print(f"\n  {BOLD}{GREEN}ЗАГАДАНО: {secret.upper()}{RESET}\n")
            solver.reset()
            cur_word, cur_reason = solver.suggest()
            continue

        # История угаданных
        if solver.guesses:
            print()
            for w, r in sorted(solver.guesses.items(), key=lambda x: x[1] if x[1] > 0 else 99999):
                if r < 0: continue
                col = rank_color(r)
                print(f"  {col}{r:>5}{RESET}  {w}")
            print()

        # Показываем всех кандидатов если их мало
        if solver.candidates and len(solver.candidates) <= 5:
            print(f"  {YELLOW}Кандидаты:{RESET}")
            _print_candidates(solver)
            print()

        # Предложение
        info = solver.info()
        info_str = f"  {DIM}[{info}]{RESET}" if info else ""
        print(f"  {BOLD}{YELLOW}→ {cur_word.upper()}{RESET}  {DIM}({cur_reason}){RESET}{info_str}")

        try:
            raw = input("  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход."); return

        if not raw:
            continue

        raw_lower = raw.lower()
        if raw_lower in ("q", "quit", "выход"):
            print("Выход."); return
        if raw_lower == "?":
            _print_candidates(solver)
            continue
        if raw_lower in ("n", "new", "новая"):
            solver.reset()
            cur_word, cur_reason = solver.suggest()
            print(f"{DIM}Новая игра.{RESET}\n")
            continue

        # Telegram paste: собираем "Слово:" + "Близость:" до первой пустой строки.
        # Остальные строки сообщения (Топ, эмодзи-строки) обрабатываются
        # по одной в основном цикле ниже — зависания нет.
        if _looks_like_telegram_start(raw):
            collected = [raw]
            try:
                while True:
                    nxt = input("").strip()
                    if not nxt:
                        break
                    collected.append(nxt)
                    if len(collected) > 300:
                        break
            except (EOFError, KeyboardInterrupt):
                pass
            tg_pairs = _parse_telegram_paste("\n".join(collected))
            if tg_pairs:
                print(f"  {DIM}Телеграм: {len(tg_pairs)} слов{RESET}")
                done = False
                for word, rank in sorted(tg_pairs, key=lambda x: x[1]):
                    if rank == 1:
                        print(f"\n  {BOLD}{GREEN}ЗАГАДАНО: {word.upper()}{RESET}\n")
                        solver.reset()
                        cur_word, cur_reason = solver.suggest()
                        done = True
                        break
                    if word in solver.guesses:
                        continue
                    if word not in _vocab_set:
                        print(f"  {DIM}'{word}' нет в словаре — пропускаем{RESET}")
                        continue
                    solver.add_guess(word, rank)
                if not done:
                    cur_word, cur_reason = solver.suggest()
                continue

        raw = raw_lower

        # Telegram заголовки и разделители — тихо пропускать
        if raw_lower.startswith("топ ближайших") or re.match(r'^-{3,}$', raw_lower):
            continue

        # Telegram emoji-строка: "🔴 слово (10145)" — обрабатывать напрямую
        tg_m = re.match(r'^[^а-яёa-z\d]*([а-яё]{2,})\s*\((\d+)\)', raw_lower)
        if tg_m:
            word, rank = tg_m.group(1), int(tg_m.group(2))
            if rank == 1:
                print(f"\n  {BOLD}{GREEN}ЗАГАДАНО: {word.upper()}{RESET}\n")
                solver.reset()
                cur_word, cur_reason = solver.suggest()
            elif word not in solver.guesses:
                if word not in _vocab_set:
                    print(f"  {DIM}'{word}' нет в словаре{RESET}")
                else:
                    solver.add_guess(word, rank)
                cur_word, cur_reason = solver.suggest()
            continue

        parts = raw.split()
        pairs = _parse_pairs(parts, cur_word)

        if pairs is None:
            print(f"  {DIM}ранг  |  слово ранг  |  сл1 р1 сл2 р2  |  слово ?  |  ?{RESET}")
            continue

        done = False
        for word, rank in pairs:
            # Пропуск слова ('слово ?')
            if rank is None:
                print(f"  {DIM}'{word}' нет в игре — пропускаем{RESET}")
                solver.guesses[word] = -1
                if solver.candidates:
                    solver.candidates.discard(word)
                continue

            # Нашли загаданное слово
            if rank == 1:
                print(f"\n  {BOLD}{GREEN}ЗАГАДАНО: {word.upper()}{RESET}\n")
                solver.reset()
                cur_word, cur_reason = solver.suggest()
                done = True
                break

            # Слова нет в нашем словаре — опечатка или отсутствует
            if word not in _vocab_set:
                print(f"  {RED}'{word}' нет в нашем словаре — проверь написание{RESET}")
                print(f"  {DIM}Триангуляция пропущена. Если слова нет в игре — используй '{word} ?'{RESET}")
                # Не трогаем кандидатов, слово не добавляем в историю
                continue

            solver.add_guess(word, rank)

        if not done:
            cur_word, cur_reason = solver.suggest()

if __name__ == "__main__":
    run()
