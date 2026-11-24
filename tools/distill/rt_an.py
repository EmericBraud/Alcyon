import json, sys, statistics as st
R=json.load(open(sys.argv[1]))
print(len(R),"positions")
ts=[r["t"] for r in R[0]["runs"]]
for i,t in enumerate(ts):
    fixed=[r for r in R if r["best_es"]-r["runs"][i]["es"]<0.05]
    same=[r for r in R if r["runs"][i]["move"]==r["move"]]
    print(f"{t/1000:5.0f}s : corrige (perte<5%) {100*len(fixed)/len(R):4.0f}% | rejoue le meme coup {100*len(same)/len(R):4.0f}% | prof moy {st.mean(r['runs'][i]['depth'] or 0 for r in R):.1f} | perte moy {100*st.mean(max(0,r['best_es']-r['runs'][i]['es']) for r in R):.1f}%")
pref=[r for r in R if "alc_sfbest" in r and r["alc_played"] is not None and r["alc_sfbest"] is not None]
print(f"\nAlcyon (prof 12) juge son coup >= coup SF : {100*sum(r['alc_played']>=r['alc_sfbest'] for r in pref)/len(pref):.0f}% ; ecart moyen {st.mean(r['alc_played']-r['alc_sfbest'] for r in pref):.0f} cp Alcyon")
# baseline: was the original error real per this re-judge?
real=[r for r in R if r["best_es"]-r["runs"][0]["es"]>=0.0]
print("ouverture vs reste (64s corrige %):")
for name,f in [("mat>=50",lambda r:r["mat"]>=50),("mat<50",lambda r:r["mat"]<50)]:
    S=[r for r in R if f(r)]
    print(f"  {name}: n={len(S)} corrige a 1s {100*sum(r['best_es']-r['runs'][0]['es']<0.05 for r in S)/len(S):.0f}% -> 64s {100*sum(r['best_es']-r['runs'][-1]['es']<0.05 for r in S)/len(S):.0f}%")
