#!/usr/bin/env python3
"""
Симуляция: солвер угадывает секретное слово.
Ранг вычисляем локально (100% совпадает с игрой).
"""
import sys, importlib.util, pathlib
import numpy as np

spec = importlib.util.spec_from_file_location(
    "solver", pathlib.Path.home() / ".local/share/claude/kontextno_solver.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.load()

Solver  = mod.Solver
vocab   = mod._vocab_set
vmat    = mod._vocab_mat
vidx    = mod._vocab_idx
vlist   = mod._vocab_list

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; YELLOW = "\033[93m"; RED = "\033[31m"; CYAN = "\033[36m"

# Кэш: полный порядок соседей для секрета (векторно, быстро)
_neighbor_cache: dict[str, list[str]] = {}

def get_game_rank(secret: str, guess: str) -> int:
    """rank(guess от secret) — точно как в игре."""
    if secret not in vidx:
        return len(vlist) + 10
    if secret not in _neighbor_cache:
        s_vec = vmat[vidx[secret]]
        sims  = vmat @ s_vec                      # (N,)
        order = np.argsort(-sims)                 # убывающий
        _neighbor_cache[secret] = [vlist[i] for i in order if vlist[i] != secret]
    neighbors = _neighbor_cache[secret]
    try:
        return neighbors.index(guess) + 2         # +2: rank1=secret itself
    except ValueError:
        return len(vlist) + 10

def simulate(secret: str, max_steps=30, verbose=True) -> int | None:
    solver = Solver()
    if verbose:
        print(f"\n{BOLD}{CYAN}Секрет: {secret.upper()}{RESET}")

    if secret not in vocab:
        if verbose: print(f"  {RED}Нет в словаре{RESET}")
        return None

    for step in range(1, max_steps + 1):
        guess, reason = solver.suggest()

        if guess == secret:
            if verbose:
                print(f"  Шаг {step:>2}: {BOLD}{GREEN}{guess}{RESET} → {GREEN}УГАДАНО!{RESET}  [{solver.info()}]")
            return step

        rank = get_game_rank(secret, guess)

        col = GREEN if rank<=10 else (YELLOW if rank<=100 else ("\033[93m" if rank<=500 else RESET))
        cands = solver.info()
        if verbose:
            print(f"  Шаг {step:>2}: {guess:22s} → {col}ранг {rank:<5}{RESET}  [{cands or '—'} | {reason}]")

        solver.add_guess(guess, rank)

    if verbose:
        print(f"  {RED}Не угадал за {max_steps} шагов{RESET}")
    return None


def main():
    test_secrets = [
        "кот", "дом", "война", "вода", "дерево",
        "хлеб", "солнце", "музыка", "книга", "город",
        "море", "огонь", "зима", "любовь", "машина",
    ]

    results = []
    for secret in test_secrets:
        steps = simulate(secret, verbose=True)
        results.append((secret, steps))

    print(f"\n{BOLD}{'─'*45}{RESET}")
    solved = [(s, n) for s, n in results if n is not None]
    failed = [s for s, n in results if n is None]

    for secret, steps in sorted(solved, key=lambda x: x[1]):
        col = GREEN if steps <= 10 else (YELLOW if steps <= 20 else RED)
        print(f"  {col}{steps:>3} ходов{RESET}  {secret}")

    if failed:
        print(f"\n{RED}Не угадал: {', '.join(failed)}{RESET}")

    if solved:
        nums = [n for _, n in solved]
        print(f"\n{BOLD}Средних ходов: {sum(nums)/len(nums):.1f}   "
              f"Макс: {max(nums)}   Мин: {min(nums)}{RESET}")

if __name__ == "__main__":
    main()
