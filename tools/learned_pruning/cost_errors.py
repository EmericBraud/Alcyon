"""Quelles confiances du MoE coutent vraiment ? (Model = 2, build SPSA)

    python3 cost_errors.py run <binaire> <positions.txt> <n> <sortie.jsonl> [profondeur]
    python3 cost_errors.py report <sortie.jsonl>

Par position (TT videe, un thread, recherche deterministe) :
1. base (mecanisme eteint) -> coup B ; MoE actif -> coup M et N confiances.
2. M != B : qualite des deux coups (recherche de base a profondeur + 1 de la
   position suivante, E logistique). On garde les positions ou M est pire.
3. Tout refuser (veto [0, N)) doit rendre B, sinon la perte ne vient pas des
   seules confiances. Bissection sur le veto [k, N) : les confiances avant k
   sont exactement celles de la recherche MoE ; la plus grande k qui rend
   encore B designe la confiance k. On verifie qu'en ne refusant qu'elle, on
   retrouve B (coupable unique), puis on la decrit (learned_moe_trace, avec le
   verdict d'une recherche complete du noeud).
4. Temoin : ~5 confiances tirees dans la meme recherche (learned_moe_sample).
"""
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

BIG = 2_000_000_000


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci\nsetoption name Threads value 1\nsetoption name Hash value 8\n"
                  "setoption name OwnBook value false\nsetoption name learned_prune_model value 2\nisready")
        self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")

    def wait(self, prefix):
        out = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("moteur mort")
            out.append(line.rstrip())
            if line.startswith(prefix):
                return out

    def search(self, fen, depth, enabled, lo=0, hi=0, trace=-1, sample=0):
        self.send(f"setoption name learned_prune_enabled value {enabled}\n"
                  f"setoption name learned_moe_veto_lo value {lo}\nsetoption name learned_moe_veto_hi value {hi}\n"
                  f"setoption name learned_moe_trace value {trace}\nsetoption name learned_moe_sample value {sample}\n"
                  f"ucinewgame\nposition fen {fen}\ngo depth {depth}")
        out = self.wait("bestmove")
        move = out[-1].split()[1]
        score, events, traces = None, 0, []
        for line in out:
            t = line.split()
            if line.startswith("info depth") and "score" in t:
                i = t.index("score")
                score = int(t[i + 2]) if t[i + 1] == "cp" else (30000 if int(t[i + 2]) > 0 else -30000)
            elif line.startswith("info string moe_trust_events"):
                events = int(t[-1])
            elif line.startswith("info string moe_trust ev"):
                traces.append(dict(zip(t[3::2], map(float, t[4::2]))))  # "info string moe_trust ev N ..."
        return move, score, events, traces


def expected(cp):
    return 1 / (1 + 10 ** (-cp / 400))


def after(fen, move):
    import chess
    b = chess.Board(fen)
    b.push_uci(move)
    return b.fen()


def one(eng, fen, depth):
    b_move, _, _, _ = eng.search(fen, depth, 0)
    m_move, _, n, _ = eng.search(fen, depth, 1)
    rec = {"fen": fen, "B": b_move, "M": m_move, "N": n}
    if m_move == b_move:
        rec["kind"] = "same"
        return rec
    eb = 1 - expected(eng.search(after(fen, b_move), depth + 1, 0)[1])
    em = 1 - expected(eng.search(after(fen, m_move), depth + 1, 0)[1])
    rec["loss"] = eb - em
    if eb - em < 0.02:
        rec["kind"] = "not_worse"
        return rec
    if eng.search(fen, depth, 1, 0, BIG)[0] != b_move:
        rec["kind"] = "veto_all_not_B"
        return rec
    lo, hi = 0, n  # veto [lo, BIG) rend B ; veto [hi, BIG) ne le rend pas (= recherche MoE)
    steps = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        steps += 1
        if eng.search(fen, depth, 1, mid, BIG)[0] == b_move:
            lo = mid
        else:
            hi = mid
    rec["culprit"] = lo
    rec["steps"] = steps
    rec["single"] = eng.search(fen, depth, 1, lo, lo + 1)[0] == b_move
    rec["kind"] = "culprit"
    rec["trace"] = eng.search(fen, depth, 1, trace=lo)[3]
    rec["sample"] = eng.search(fen, depth, 1, sample=max(1, n // 5))[3]
    return rec


def run(engine, positions, n, out, depth=12):
    fens = [l.strip() for l in open(positions) if l.strip()]
    step = max(1, len(fens) // int(n))
    fens = fens[::step][:int(n)]
    jobs = int(os.environ.get("CE_JOBS", os.cpu_count()))
    engines = []

    def work(chunk):
        eng = Engine(engine)
        res = []
        for fen in chunk:
            try:
                res.append(one(eng, fen, int(depth)))
            except Exception as e:  # noqa: BLE001 -- une position ratee ne doit pas tuer la serie
                res.append({"fen": fen, "kind": "error", "err": str(e)})
                eng.p.kill()
                eng = Engine(engine)
        eng.send("quit")
        return res

    chunks = [fens[i::jobs] for i in range(jobs)]
    with ThreadPoolExecutor(jobs) as ex, open(out, "w") as f:
        for res in ex.map(work, chunks):
            for r in res:
                f.write(json.dumps(r) + "\n")
    del engines


def report(path):
    rs = [json.loads(l) for l in open(path)]
    kinds = {}
    for r in rs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"{len(rs)} positions : " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    cul = [r for r in rs if r["kind"] == "culprit"]
    worse = [r for r in rs if r["kind"] in ("culprit", "veto_all_not_B")]
    if worse:
        print(f"perte moyenne des positions pires : {sum(r['loss'] for r in worse) / len(worse):.3f} "
              f"(E, sur {len(worse)}) ; perte totale / position : {sum(r['loss'] for r in worse) / len(rs):.4f}")
    if not cul:
        return
    print(f"coupable unique (refuser lui seul rend B) : {sum(r['single'] for r in cul)} / {len(cul)}")
    tr = [r["trace"][0] for r in cul if r["trace"]]
    sm = [t for r in cul for t in r["sample"]]

    def wrong(t):
        return (t["red"] >= t["beta"]) != (t["full"] >= t["beta"])

    def describe(name, ts):
        if not ts:
            return
        n = len(ts)
        f = lambda k: sum(k(t) for t in ts) / n  # noqa: E731
        print(f"\n== {name} ({n})")
        print(f"  erreur au sens du label (reduit et complet de cotes differents de beta) : {f(wrong):.1%}")
        print(f"  |red - full| moyen : {f(lambda t: abs(t['red'] - t['full'])):.0f} cp ; "
              f"|red - beta| moyen : {f(lambda t: abs(t['red'] - t['beta'])):.0f}")
        print(f"  reduit FH : {f(lambda t: t['red'] >= t['beta']):.1%} ; z - seuil moyen : {f(lambda t: t['z'] - t['thr']):.2f}")
        print(f"  ply moyen {f(lambda t: t['ply']):.1f} ; ply <= 2 : {f(lambda t: t['ply'] <= 2):.1%} ; depth moyen {f(lambda t: t['depth']):.1f}")
        print(f"  coup precedent = prise : {f(lambda t: t['prev_capture']):.1%} ; pieces moyen {f(lambda t: t['pieces']):.1f}")
        print(f"  TT trouvee : {f(lambda t: t['tt_found']):.1%} ; cut node : {f(lambda t: t['cut']):.1%}")
        print(f"  |eval - beta| moyen : {f(lambda t: abs(t['eval'] - t['beta'])):.0f}")

    describe("coupables", tr)
    describe("temoin : confiances tirees dans les memes recherches", sm)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:])
    else:
        report(sys.argv[2])
