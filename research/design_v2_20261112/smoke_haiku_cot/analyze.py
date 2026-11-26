import json, collections, statistics as st
R="raw/v2/"
def load(f): return [json.loads(l) for l in open(R+f)]
arms={"끔(기존)":None,"low 스모크":"haiku_low_p1_smoke.jsonl","medium 스모크":"haiku_medium_p1_smoke.jsonl",
      "프롬프트 추론(cot)":"haiku_cot_p1_smoke.jsonl","생각 지시+medium(think)":"haiku_think_p1_smoke.jsonl"}
data={k:load(v) for k,v in arms.items() if v}
ids={r["probe_id"] for r in data["프롬프트 추론(cot)"]}
assert all({r["probe_id"] for r in d}==ids for d in data.values())
base=collections.defaultdict(collections.Counter)
for l in open(R+"haiku_p1.jsonl"):
    r=json.loads(l)
    if r["probe_id"] in ids and r.get("valid"): base[r["probe_id"]][r["parsed"]]+=1
def pb(pid,v): c=base[pid]; return c[v]/sum(c.values())
selfagree=st.mean(sum((n/sum(c.values()))**2 for n in c.values()) for c in base.values())
modep=st.mean(max(c.values())/sum(c.values()) for c in base.values())
print(f"기존 끔 k=100: 최빈 보기 확률 평균 {modep:.3f}, 다시 뽑았을 때 같은 답 기대확률 {selfagree:.3f}, 표본수 중앙 {st.median(sum(c.values()) for c in base.values())}")
for k,d in data.items():
    v=[r for r in d if r["valid"]]
    th=sum(r["thinking_blocks"]>0 for r in d)
    mode=st.mean(r["parsed"]==base[r["probe_id"]].most_common(1)[0][0] for r in v)
    prob=st.mean(pb(r["probe_id"],r["parsed"]) for r in v)
    outs=[r["output_tokens"] for r in d]
    inv=collections.Counter(r["invalid_type"] for r in d if not r["valid"])
    print(f"{k}: 유효 {len(v)}/300 무효{dict(inv)} | 생각블록 {th}/300 | 출력토큰 평균 {st.mean(outs):.0f} 중앙 {st.median(outs):.0f} | 기존 최빈과 같음 {mode:.3f} | 기존 분포에서 그 답 확률 평균 {prob:.3f}")
# by level for cot / think
for k in ["프롬프트 추론(cot)","생각 지시+medium(think)","medium 스모크"]:
    by=collections.defaultdict(list)
    for r in data[k]:
        if r["valid"]: by[r["level"]].append(r["parsed"]==base[r["probe_id"]].most_common(1)[0][0])
    print(k, {l:(len(x),round(st.mean(x),3)) for l,x in sorted(by.items())})
