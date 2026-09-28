"""Reference independante pour sweep_report.py : le coup de Stockfish.

    python3 sf_ref.py <stockfish> <profondeur> <fichier de positions> > ref.txt

Une ligne de sortie par position, dans l'ordre du fichier : "<index>\t<coup>".
Pourquoi pas ref14 d'Alcyon : une recherche a profondeur 14 repasse par les
iterations 1..12 exactement comme off12 (meme TT, memes historiques), donc
elle favorise toute config qui ne perturbe pas l'arbre, a qualite egale.
"""
import subprocess, sys

sf, depth, fens = sys.argv[1], int(sys.argv[2]), sys.argv[3]
start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
p = subprocess.Popen([sf], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdin.write("uci\nsetoption name Hash value 64\nsetoption name Threads value 1\nisready\n")
while p.stdout.readline().strip() != "readyok":
    pass
for n, fen in enumerate(open(fens)):
    p.stdin.write(f"ucinewgame\nposition fen {fen.strip()}\ngo depth {depth}\n")
    while not (line := p.stdout.readline()).startswith("bestmove"):
        pass
    print(f"{start + n}\t{line.split()[1]}", flush=True)
p.stdin.write("quit\n")
