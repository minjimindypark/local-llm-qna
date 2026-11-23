"""모든 채점 결과(out_*e2e*.jsonl)를 한 표로. 채점 셋:
 ① 답할때 답함  ② 넘길때 넘김(저자에게+거절)
 ③ 근거 옳음 = 답한 것 중, 근거로 댄 S번호가 '그 질문의 정답 조각'을 실제로 가리키는 비율.
    정답 조각이 애초에 검색(picked)에 없었으면 그 문항은 ③에서 제외(검색 문제지 답 문제 아님)."""
import json, re, glob
chunks = json.load(open("chunks.json", encoding="utf-8"))
gold = {}
for name in ["book_1","book_2","book_3","book_4","book_5","webinar"]:
    for l in open(f"qa_{name}.jsonl", encoding="utf-8"):
        d = json.loads(l); gold[d["q"]] = {k for e in d["evidence"] for k,c in enumerate(chunks) if c["src"]==name and any(e in ln for ln in c["lines"])}
def verdict(o):
    m = re.search(r'판정:\s*(답함|저자에게|거절)', o); return m.group(1) if m else "형식어김"
def cited(o):
    m = re.search(r'근거:\s*([S0-9, ]+)', o); return set(int(x) for x in re.findall(r'S(\d+)', m.group(1))) if m else set()
NAME = {"base_e2e":"학습전","ft_e2e":"A기본s0","A1_e2e":"A기본s1","B0_e2e":"B_RAFT_s0","B1_e2e":"B_RAFT_s1","C0_e2e":"C_넘기기s0","C1_e2e":"C_넘기기s1","D0_e2e":"D_Muon_s0"}
rows=[]
for f in sorted(glob.glob("out_*e2e*.jsonl")):
    tag=f.replace("out_","").replace(".jsonl","")
    o=[json.loads(l) for l in open(f,encoding="utf-8")]
    if not o or "kind" not in o[0]: continue
    ans=[x for x in o if x["kind"]=="답함"]; abst=[x for x in o if x["kind"] in ("저자에게","거절")]
    a_ok=sum(verdict(x["out"])=="답함" for x in ans); b_ok=sum(verdict(x["out"]) in ("저자에게","거절") for x in abst)
    c_ok=c_tot=0
    for x in ans:
        if verdict(x["out"])!="답함": continue
        g=gold.get(x["q"],set()); pos={i+1 for i,k in enumerate(x["picked"]) if k in g}
        if not pos: continue   # 정답 조각이 검색에 없음 → 제외
        c_tot+=1; c_ok+=bool(cited(x["out"]) & pos)
    rows.append((NAME.get(tag,tag),len(o),a_ok,len(ans),b_ok,len(abst),c_ok,c_tot))
print(f"{'판':<11}{'답할때답함':>11}{'넘길때넘김':>11}{'근거옳음':>11}")
for n,tot,a,at,b,bt,c,ct in rows:
    print(f"{n:<11}{a:>6}/{at:<4}{b:>6}/{bt:<4}{(str(c)+'/'+str(ct)):>11}")
