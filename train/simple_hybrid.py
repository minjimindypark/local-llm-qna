"""단순 짬뽕본(학습 없음). 아이디어만 떼어 합친다:
 - 아스트라/활동 조종: 학습 질문으로 '넘기기 - 답하기' 방향을 층마다 구해, 답할 때 그 방향을 더해(넘기기 쪽으로 민다).
 - DoLa: 뒤층 vs 앞층 대조로 지어내기 줄이기 (transformers 기본 기능).
같은 test_e2e 71개로, 켜고 끈 조합을 잰다. 모델은 안 건드리고 inference 때만 작동."""
import glob, json, os, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
BASE = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*"))[0]
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda").eval()
nL = model.config.num_hidden_layers
def ids_of(x): return tok.apply_chat_template(x["messages"], add_generation_prompt=True, return_tensors="pt").to("cuda")
# 1) 방향 구하기: 학습 질문 일부로 각 층 마지막 토큰 hidden 평균차(넘기기 - 답하기)
tr=[json.loads(l) for l in open("train.jsonl",encoding="utf-8")]
import random; random.seed(0); random.shuffle(tr)
ans=[x for x in tr if x["kind"]=="답함"][:80]; ab=[x for x in tr if x["kind"] in("저자에게","거절")][:80]
def hid(xs):
    acc=torch.zeros(nL+1,model.config.hidden_size,device="cuda")
    for x in xs:
        with torch.no_grad(): h=model(ids_of(x),output_hidden_states=True).hidden_states
        for i,hs in enumerate(h): acc[i]+=hs[0,-1].float()
    return acc/len(xs)
print("방향 계산 중...",flush=True)
dirv=(hid(ab)-hid(ans)); dirv=dirv/dirv.norm(dim=-1,keepdim=True)  # 층별 단위벡터
test=[json.loads(l) for l in open("test_e2e.jsonl",encoding="utf-8")]
hooks=[]
def add_hook(layer, vec, alpha):
    def f(m,inp,out):
        o=out[0] if isinstance(out,tuple) else out
        o=o+alpha*vec.to(o.dtype)
        return (o,)+out[1:] if isinstance(out,tuple) else o
    return model.model.layers[layer].register_forward_hook(f)
def run(tag, layer=None, alpha=0.0, dola=None):
    for h in hooks: h.remove()
    hooks.clear()
    if layer is not None: hooks.append(add_hook(layer, dirv[layer+1], alpha))
    out=[]
    for x in test:
        g_kw=dict(max_new_tokens=300,do_sample=False)
        if dola: g_kw["dola_layers"]=dola; g_kw["repetition_penalty"]=1.2
        with torch.no_grad(): g=model.generate(input_ids=ids_of(x),**g_kw)
        y=tok.decode(g[0,ids_of(x).shape[1]:],skip_special_tokens=True).strip()
        out.append({**{k:x[k] for k in["q","picked","kind","style","src","target"]},"out":y})
    open(f"out_{tag}_e2e.jsonl","w",encoding="utf-8").write("".join(json.dumps(z,ensure_ascii=False)+"\n" for z in out))
    print(tag,"끝",flush=True)
MODE=sys.argv[1] if len(sys.argv)>1 else "all"
if MODE=="all":
    run("H_plain")
    for L in [12,15,17,19,22]:
        for A in [6.0, 10.0]:
            run(f"H_L{L}a{int(A)}", layer=L, alpha=A)
