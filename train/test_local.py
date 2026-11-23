"""로컬 테스터: 질문→e5 검색 top-5→학습된 4B(LoRA) 답변. 로컬에서 눈으로 확인하는 용도.
  python test_local.py            # http://127.0.0.1:8799
근거 조각도 같이 보여줘서 '왜 그 답이 나왔는지(엉뚱한 일화 집었는지)'를 눈으로 본다. CPU라 답 1개 1~2분.
"""
import glob, json, os, re, sys, torch, torch.nn.functional as F
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
LORA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "lora_ep_e3")   # v2 충실도 3에포크
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8799
dev = "cpu"

# 학습 때 쓴 시스템 프롬프트를 시험파일에서 그대로 가져온다(형식 불일치 방지)
SYS = json.loads(open(os.path.join(HERE, "test_e2e_v2.jsonl"), encoding="utf-8").readline())["messages"][0]["content"]
chunks = json.load(open(os.path.join(HERE, "chunks.json"), encoding="utf-8"))

print("e5 로딩...", flush=True)
e5d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--intfloat--multilingual-e5-base/snapshots/*"))[0]
e5t = AutoTokenizer.from_pretrained(e5d); e5 = AutoModel.from_pretrained(e5d).eval()
@torch.no_grad()
def emb(texts):
    enc = e5t(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    h = e5(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    v = (h * m).sum(1) / m.sum(1)                      # mean pooling
    return F.normalize(v, dim=-1)
print("조각 임베딩...", flush=True)
C = emb(["passage: " + c["text"] for c in chunks])

print("4B+LoRA 로딩(시간 걸림)...", flush=True)
base = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*"))[0]
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).to(dev)
model = PeftModel.from_pretrained(model, LORA).eval()
print(f"준비 끝. http://127.0.0.1:{PORT}  (LoRA={os.path.basename(LORA)})", flush=True)

def answer(q):
    s = (emb(["query: " + q]) @ C.T)[0]
    top = [int(k) for k in (-s).argsort()[:5]]
    docs = "\n\n".join(f"[S{i+1}]\n{chunks[k]['text']}" for i, k in enumerate(top))
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": f"[자료]\n{docs}\n\n[질문]\n{q}"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
    with torch.no_grad():
        g = model.generate(input_ids=ids, max_new_tokens=250, do_sample=False)
    out = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
    ev = [{"S": i + 1, "src": chunks[k]["src"], "text": chunks[k]["text"][:160]} for i, k in enumerate(top)]
    return out, ev

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>로컬 모델 테스트</title><style>body{font-family:system-ui;max-width:680px;margin:0 auto;padding:20px;line-height:1.6}
textarea,button{font:inherit;width:100%;box-sizing:border-box;padding:10px;margin:6px 0}#out{white-space:pre-wrap;background:#f3f4f6;border-radius:10px;padding:14px;margin-top:12px}
.ev{font-size:.85em;opacity:.8;border-top:1px solid #ccc;margin-top:10px;padding-top:8px}</style>
<h2>로컬 4B 테스트 <small id=cfg></small></h2><p>질문을 치면 검색된 근거 5개로 학습된 모델이 답해요. CPU라 1~2분 걸려요.</p>
<textarea id=q rows=2 placeholder="예) 토익 800점인데 시작할 수 있나요?"></textarea><button id=go>물어보기</button>
<div id=out></div>
<script>const $=i=>document.getElementById(i);fetch('/cfg').then(r=>r.text()).then(t=>$('cfg').textContent=t);
$('go').onclick=async()=>{const q=$('q').value.trim();if(!q)return;$('go').disabled=1;$('out').textContent='생각 중… (1~2분)';
try{const r=await fetch('/ask',{method:'POST',body:q});const d=await r.json();
$('out').innerHTML='<b>답</b>\\n'+d.answer+'<div class=ev><b>넣어준 근거</b>\\n'+d.ev.map(e=>'S'+e.S+' ('+e.src+'): '+e.text).join('\\n\\n')+'</div>';}
catch(e){$('out').textContent='오류: '+e}$('go').disabled=0};</script>"""

class H(BaseHTTPRequestHandler):
    def _s(self, b, k="application/json"):
        d = b if isinstance(b, bytes) else (b.encode() if isinstance(b, str) else json.dumps(b, ensure_ascii=False).encode())
        self.send_response(200); self.send_header("Content-Type", k + "; charset=utf-8"); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self):
        if self.path == "/cfg": return self._s("(LoRA=" + os.path.basename(LORA) + ")", "text/plain")
        self._s(PAGE, "text/html")
    def do_POST(self):
        q = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()[:500]
        try:
            a, ev = answer(q); self._s({"answer": a, "ev": ev})
        except Exception as e:
            import traceback; traceback.print_exc()
            self._s({"answer": "오류: " + repr(e), "ev": []})
    def log_message(self, *a): pass

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
