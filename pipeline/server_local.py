# -*- coding: utf-8 -*-
"""로컬 책 QA 서버 (유료 API 없음). 로컬 머신에서 직접 구동.

확정 파이프:
  모델    젬마3-12B SFT Q6_K GGUF  (llama-server 백엔드, HTTP로 호출)
  검색    bge-m3 CLS → 전체(166조각) 코사인 top-30 → bge-reranker-v2-m3 리랭크 → best-first top-5
  고정답  answers.md(저자 검수답)를 같은 리랭커로 질문과 맞춰, 관련 높은 것만 [자료]에 함께 붙임
  라우팅  모델 자기판정 (첫 줄 "판정: 답함/저자에게/거절")

실행 (먼저 llama-server 를 띄운 뒤 이 서버를 띄운다):
  1) llama-server 로 GGUF 모델을 GPU에 올려 127.0.0.1:8099 로 서빙:
     CUDA_VISIBLE_DEVICES=0 llama-server -m MODEL.gguf -ngl 999 -c 8192 \
       --no-webui --host 127.0.0.1 --port 8099
  2) torch/transformers 가 있는 환경에서 이 서버 실행:
     python server_local.py        # http://127.0.0.1:8777

같은 폴더 파일(모두 읽기만): codes.txt(접속코드), answers.md(저자 고정답).
못 답한 질문은 pending_local.jsonl 에 쌓는다.
AI 요금 0원 — 전부 로컬.
"""
import glob, hmac, json, os, re, sys, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

HERE = Path(__file__).parent
TRAIN = HERE / "train"
CMP = TRAIN / "cmp"

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8099")
HOST = os.environ.get("BOOK_QA_HOST", "127.0.0.1")
PORT = int(os.environ.get("BOOK_QA_PORT", 8777))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PER_CODE_PER_DAY = 30
ALL_PER_DAY = 500
MAX_QUESTION = 500
TOPK = 5                       # 책 조각 best-first 개수 (확정 파이프)
AUTHOR_MAX = 2                 # 붙일 저자 고정답 최대 개수
# 저자 고정답 리랭커 점수 하한. bge-reranker-v2-m3 는 짧은 FAQ 쌍에 전반적으로 음수를 주므로
# (딱 맞는 답도 -6~-8, 무관은 -9~-11) 하한을 음수로 둬서 '진짜 맞는 FAQ'만 [자료]에 붙인다.
# → 과한 넘김(답할 수 있는 돈/시간 질문)을 메우되, 무관 질문엔 안 붙어 라우팅 오염을 막는다.
AUTHOR_FLOOR = -7.0
N_PREDICT = 320
N_PREDICT_DEEP = 640           # "더 구체적으로" 재답변 때 토큰 한도 (예시·숫자 생략 방지)
DEEP_INSTR = ("\n\n[요청] 사용자가 더 구체적인 답을 원해요. 자료에 있는 구체적인 예시 문장·숫자·항목을 "
              "요약하거나 생략하지 말고 그대로 모두 포함해서 자세히 답하세요.")

# ---------- 학습 때 쓴 시스템 프롬프트를 그대로 사용(형식 일치) ----------
SYS = json.loads((CMP / "test_e2e_v2.jsonl").open(encoding="utf-8").readline())["messages"][0]["content"]
# 안전 규칙: 탈세·정보 유출/판매·타인 명의 등 악용 요청은 거절(어려운 97문항 실측으로 회귀 없이 검증).
SYS = SYS.rstrip() + "\n- 탈세·불법 행위, 통역 내용·환자 정보의 유출이나 판매, 타인 명의 사용 등 악용하려는 요청이면, 주제가 재택 통역과 닿아 있어도 \"판정: 거절\" 한 줄만."
# 내부 자료번호(S태그) 노출 방지: 답 본문에는 자료번호 대신 내용을 직접 쓴다.
SYS = SYS.rstrip() + "\n- '답:' 본문에는 내용을 직접 풀어서 써요. S1·S7 같은 자료 번호를 쓰거나 '~에 나와요'로 미루지 말고, 해당 내용을 문장으로 답해요."
# 회사 실명 노출 방지: 특정 통역회사 상호는 답에 쓰지 않고 일반적 설명으로.
SYS = SYS.rstrip() + "\n- 특정 통역회사의 상호(회사 이름)는 답에 쓰지 말아요. 이름을 묻더라도 콕 집지 말고, 공고로 회사의 단계를 판별하는 법처럼 일반적인 설명으로 답해요."
# 전문영역(세금·법률·비자) 환각 방지: 자료 밖 수치·절차·요건은 생성 금지, 전문가 상담 권고.
SYS = SYS.rstrip() + "\n- 세금·법률·비자 같은 전문 영역은 자료에 나온 내용만 말해요. 자료에 없는 세율·신고 절차·요건·숫자는 지어내지 말고, 일반적으로만 안내한 뒤 전문가 상담을 권해요."

# ---------- 조각/고정답 로드 ----------
CHUNKS = json.loads((TRAIN / "chunks.json").read_text(encoding="utf-8"))   # 166, keys: text/src/lines/ctx


def parse_answers_md(path):
    """answers.md 를 (q, a) 목록으로 파싱. 형식: 'Q. ...' 다음 줄 'A. ...'."""
    if not path.exists():
        return []
    pairs, q, a = [], None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Q."):
            if q is not None and a is not None:
                pairs.append((q.strip(), a.strip()))
            q, a = line[2:].strip(), None
        elif line.startswith("A.") and q is not None:
            a = line[2:].strip()
        elif a is not None and line.strip():
            a += " " + line.strip()
    if q is not None and a is not None:
        pairs.append((q.strip(), a.strip()))
    return pairs


AUTHOR = parse_answers_md(HERE / "answers.md")   # [(q,a), ...]

# ---------- 임베더/리랭커 (build_ctx 와 동일: bge-m3 CLS, bge-reranker-v2-m3) ----------


def snap(repo):
    for d in glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo}/snapshots/*")):
        if (os.path.exists(os.path.join(d, "model.safetensors")) or
                os.path.exists(os.path.join(d, "pytorch_model.bin"))) and \
                os.path.exists(os.path.join(d, "tokenizer.json")):
            return d
    raise SystemExit(f"no complete snapshot for {repo}")


print("임베더/리랭커 로딩...", flush=True)
BGE = snap("BAAI--bge-m3")
RR = snap("BAAI--bge-reranker-v2-m3")
emb_tok = AutoTokenizer.from_pretrained(BGE)
emb_mdl = AutoModel.from_pretrained(BGE).eval().to(DEV)
rr_tok = AutoTokenizer.from_pretrained(RR)
rr_mdl = AutoModelForSequenceClassification.from_pretrained(RR).eval().to(DEV)

# 젬마 토크나이저: 학습과 똑같은 프롬프트 문자열을 만들기 위함(시스템을 user 턴에 합치고 <bos> 선두)
GEM = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--gemma-3-12b-it/snapshots/*"))[0]
gem_tok = AutoTokenizer.from_pretrained(GEM)


@torch.no_grad()
def embed(texts, is_query):
    out = []
    for i in range(0, len(texts), 32):
        enc = emb_tok(texts[i:i + 32], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        h = emb_mdl(**enc).last_hidden_state[:, 0]          # CLS pooling
        out.append(F.normalize(h, dim=-1).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def rerank(q, texts):
    if not texts:
        return np.array([])
    scores = []
    for i in range(0, len(texts), 64):
        enc = rr_tok([[q, t] for t in texts[i:i + 64]], padding=True, truncation=True,
                     max_length=768, return_tensors="pt").to(DEV)
        scores.append(rr_mdl(**enc).logits.view(-1).float().cpu().numpy())
    return np.concatenate(scores)


print("조각 임베딩...", flush=True)
CORPUS_TXT = [c["ctx"] + "\n" + c["text"] for c in CHUNKS]                  # 검색은 맥락글+본문
CORPUS_EMB = embed(CORPUS_TXT, is_query=False)
AUTHOR_TXT = [f"{q} {a}" for q, a in AUTHOR]                                 # 고정답 매칭용 텍스트


def retrieve(q):
    """확정 파이프: bge-m3 top-30 코사인 → 리랭커 → best-first top-5 (모델엔 본문 text 만)."""
    qe = embed([q], is_query=True)[0]
    sims = CORPUS_EMB @ qe
    cand = (-sims).argsort()[:30].tolist()
    rr = rerank(q, [CORPUS_TXT[k] for k in cand])
    order = [cand[i] for i in np.argsort(-rr)][:TOPK]                        # best-first
    book = [{"idx": int(k), "src": CHUNKS[k]["src"], "text": CHUNKS[k]["text"]} for k in order]

    # 저자 고정답: 같은 리랭커로 질문과 맞춰 관련 높은 것만(하한 넘는 것 최대 AUTHOR_MAX)
    author = []
    if AUTHOR:
        arr = rerank(q, AUTHOR_TXT)
        for i in np.argsort(-arr)[:AUTHOR_MAX]:
            if arr[i] >= AUTHOR_FLOOR:
                author.append({"i": int(i), "q": AUTHOR[i][0], "a": AUTHOR[i][1], "score": float(arr[i])})
    return book, author


def build_prompt(q, book, author, deep=False):
    blocks = [f"[S{p + 1}]\n{b['text']}" for p, b in enumerate(book)]
    for j, au in enumerate(author):                                         # 저자 고정답을 이어서 [자료] 로
        blocks.append(f"[S{len(book) + j + 1}] (작가 추가답변)\nQ. {au['q']}\nA. {au['a']}")
    docs = "\n\n".join(blocks)
    user = f"[자료]\n{docs}\n\n[질문]\n{q}" + (DEEP_INSTR if deep else "")   # deep=재답변 시 '자세히' 지시 추가
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": user}]
    s = gem_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return s[len("<bos>"):] if s.startswith("<bos>") else s                 # llama-server 가 bos 를 붙이므로 제거


def call_llama(prompt, n_predict=N_PREDICT):
    body = json.dumps({"prompt": prompt, "temperature": 0, "n_predict": n_predict,
                       "cache_prompt": True, "stop": ["<end_of_turn>", "<eos>"]}).encode()
    req = urllib.request.Request(LLAMA_URL + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["content"].strip()


def parse_out(text):
    """모델 출력 → (verdict, evidence_list, answer_text)."""
    verdict, ev, ans_lines, in_ans = "", [], [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("판정:"):
            v = s[3:].strip()
            verdict = "답함" if "답함" in v else "저자에게" if "저자" in v else "거절" if "거절" in v else v
        elif s.startswith("근거:"):
            ev = re.findall(r"S\d+", s)
        elif s.startswith("답:"):
            in_ans = True
            ans_lines.append(s[2:].strip())
        elif in_ans:
            ans_lines.append(line)
    return verdict, ev, "\n".join(ans_lines).strip()


def answer(q, deep=False):
    """질문 하나 처리. deep=True 면 '더 구체적으로' 재답변(지시+토큰한도↑). 반환: dict(...)."""
    t0 = time.time()
    book, author = retrieve(q)
    prompt = build_prompt(q, book, author, deep)
    raw = call_llama(prompt, N_PREDICT_DEEP if deep else N_PREDICT)
    verdict, ev, ans = parse_out(raw)
    # 근거 S번호 → 장/절(책 조각 src) 또는 작가 추가답변 라벨
    chapters = []
    for s in ev:
        n = int(s[1:])
        if 1 <= n <= len(book):
            chapters.append(book[n - 1]["src"])
        elif len(book) < n <= len(book) + len(author):
            chapters.append("작가 추가답변")
    return {
        "verdict": verdict, "answer": ans, "chapters": chapters, "evidence": ev,
        "book": [{"S": i + 1, "src": b["src"], "text": b["text"][:140]} for i, b in enumerate(book)],
        "author_used": [{"S": len(book) + j + 1, "q": au["q"], "score": round(au["score"], 2)} for j, au in enumerate(author)],
        "raw": raw, "deep": deep, "elapsed": round(time.time() - t0, 1),
    }


# ---------- 접속코드·한도·pending (server.py UX 재사용) ----------
def codes():
    f = HERE / "codes.txt"
    if not f.exists():
        return []
    return [c for c in (ln.split("#")[0].strip() for ln in f.read_text().splitlines()) if c]


lock = threading.Lock()
used = {"day": "", "by_code": {}, "all": 0}


def allow(code):
    with lock:
        today = time.strftime("%Y-%m-%d")
        if used["day"] != today:
            used.update(day=today, by_code={}, all=0)
        n = used["by_code"].get(code, 0)
        if n >= PER_CODE_PER_DAY or used["all"] >= ALL_PER_DAY:
            return False
        used["by_code"][code] = n + 1
        used["all"] += 1
        return True


def to_pending(question):
    with lock, open(HERE / "pending_local.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "question": question}, ensure_ascii=False) + "\n")


log_lock = threading.Lock()
LOG_PATH = HERE / "interactions_local.jsonl"


def log_event(obj):
    """사용자 상호작용 1건을 JSONL 로 적재. 나중에 벤치마크·할루시네이션 측정의 원천."""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **obj}
    try:
        with log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:                                                   # 로그 실패가 응답을 막지 않도록
        print("로그 적재 실패:", repr(e), file=sys.stderr)


HANDOFF_MSG = "책에 없는 내용이라 작가에게 전달했어요. 작가가 답하면 다음부터는 여기서 바로 답이 나와요."
REFUSE_MSG = "이 질문에는 도와드리기 어려워요. 재택 통역 일에 관한 질문을 해주세요."

PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>재택 통역 질문하기 (로컬)</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;padding:24px 16px;line-height:1.7}
input,textarea,#go{font:inherit;width:100%;box-sizing:border-box;padding:10px;margin:6px 0}
#out{white-space:pre-wrap;border-top:1px solid #8884;margin-top:16px;padding-top:16px}
.hi{font-size:1.15em;font-weight:600;margin:4px 0 14px}
.chip{display:block;width:100%;text-align:left;font:inherit;cursor:pointer;background:#f3f4f6;border:1px solid #d1d5db;border-radius:12px;padding:12px 14px;margin:8px 0}
.chip:hover{background:#e5edff;border-color:#9cb8ff}
.muted{opacity:.65;font-size:.92em;margin-top:22px}.meta{opacity:.6;font-size:.85em;margin-top:10px}</style>
<h1>재택 통역, 루트만 알면 됩니다</h1>
<p class="hi">반갑습니다! 돈을 벌어볼까요? 👋<br>궁금한 것을 눌러보세요. (로컬 서버 · AI 요금 0원)</p>
<button class="chip" data-q="집에서 통역으로 진짜 돈 벌 수 있어요? 어떻게 시작해요?">💰 집에서 통역으로 진짜 돈 벌 수 있어요?</button>
<button class="chip" data-q="영어 어느 정도 해야 시작할 수 있어요?">🗣️ 영어 어느 정도 해야 시작할 수 있어요?</button>
<button class="chip" data-q="준비물 뭐 사야 돼요? 돈 많이 들어요?">🎧 준비물 뭐 사야 돼요?</button>
<p class="muted">또는 아래에 직접 질문하세요. 책에 없는 내용은 작가에게 전달돼요.</p>
<input id="code" placeholder="구매자 접속 코드"><textarea id="q" rows="3" maxlength="500" placeholder="예) 토익 800점인데 시작할 수 있나요?"></textarea>
<button id="go">질문하기</button><div id="out"></div>
<script>
const $=id=>document.getElementById(id);try{$('code').value=localStorage.getItem('code')||''}catch(e){}
function fb(id,signal){fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,signal})}).catch(()=>{});}
function addFeedback(id,question,deep){const box=document.createElement('div');box.style.marginTop='14px';
if(!deep){const b1=document.createElement('button');b1.className='chip';b1.textContent='🔎 더 구체적으로 알려주세요';
b1.onclick=()=>{fb(id,'more');ask(question,true);};box.appendChild(b1);}
const b2=document.createElement('button');b2.className='chip';b2.textContent='👍 이걸로 충분해요';
b2.onclick=()=>{fb(id,'enough');box.innerHTML='<div class="meta">피드백 고마워요 🙏 (다음 개선에 쓰여요)</div>';};
box.appendChild(b2);$('out').appendChild(box);}
async function ask(question,deep){const code=$('code').value.trim();question=(question||$('q').value).trim();if(!question)return;
$('q').value=question;
if(!code){$('out').textContent='먼저 구매자 접속 코드를 넣어 주세요.';$('code').focus();return;}
try{localStorage.setItem('code',code)}catch(e){}
$('go').disabled=true;$('out').textContent=deep?'더 자세히 찾는 중이에요…':'답을 찾는 중이에요…';window.scrollTo(0,document.body.scrollHeight);
try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code,question,deep:!!deep})});
const d=await r.json();let t=d.error||d.answer;if(d.chapters&&d.chapters.length)t+='\\n\\n근거: '+d.chapters.join(' / ');
$('out').textContent=t;
if(d.elapsed!=null){const m=document.createElement('div');m.className='meta';m.textContent='응답 '+d.elapsed+'초';$('out').appendChild(m);}
if(d.id&&d.verdict==='답함'){addFeedback(d.id,question,deep);}}
catch(e){$('out').textContent='연결에 실패했어요. 잠시 뒤 다시 시도해 주세요.'}
$('go').disabled=false}
$('go').onclick=()=>ask();
document.querySelectorAll('.chip[data-q]').forEach(b=>b.onclick=()=>ask(b.dataset.q));
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, kind="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send(200, PAGE, "text/html") if self.path == "/" else self.send(404, {"error": "없는 주소예요."})

    def do_POST(self):
        if self.path == "/feedback":
            return self.handle_feedback()
        if self.path != "/ask":
            return self.send(404, {"error": "없는 주소예요."})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size > 4000:
                return self.send(413, {"error": "질문이 너무 길어요."})
            body = json.loads(self.rfile.read(size))
            code, question = str(body["code"]).strip(), str(body["question"]).strip()[:MAX_QUESTION]
            deep = bool(body.get("deep", False))                             # "더 구체적으로" 재답변 요청
        except (ValueError, KeyError, TypeError):
            return self.send(400, {"error": "요청 형식이 잘못됐어요."})
        if not question or not any(hmac.compare_digest(code.encode(), c.encode()) for c in codes()):
            return self.send(403, {"error": "접속 코드가 맞지 않아요."})
        if not allow(code):
            return self.send(429, {"error": "오늘 질문 한도를 다 썼어요. 내일 다시 질문해 주세요."})
        try:
            r = answer(question, deep=deep)
        except Exception as e:
            print("로컬 모델 오류:", repr(e), file=sys.stderr)
            return self.send(503, {"error": "지금은 답할 수 없어요. 잠시 뒤 다시 시도해 주세요."})
        iid = uuid.uuid4().hex[:12]                                          # 상호작용 id (피드백과 묶는 열쇠)
        log_event({"event": "ask", "id": iid, "deep": deep, "code": code, "question": question,
                   "verdict": r["verdict"], "answer": r["answer"], "evidence": r["evidence"],
                   "book": r["book"], "author_used": r["author_used"], "elapsed": r["elapsed"]})
        base = {"id": iid, "verdict": r["verdict"], "deep": deep, "elapsed": r["elapsed"]}
        if r["verdict"] == "답함" and r["answer"]:
            return self.send(200, {**base, "answer": r["answer"], "chapters": r["chapters"]})
        if r["verdict"] == "거절":
            return self.send(200, {**base, "answer": REFUSE_MSG, "chapters": []})
        to_pending(question)                                                 # 저자에게(또는 답 비어있음)
        return self.send(200, {**base, "answer": HANDOFF_MSG, "chapters": []})

    def handle_feedback(self):
        """사용자 피드백 신호(more=더 필요, enough=충분)를 상호작용 id 에 묶어 로그."""
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size > 2000:
                return self.send(413, {"error": "too large"})
            body = json.loads(self.rfile.read(size))
            iid, signal = str(body["id"])[:32], str(body["signal"])[:16]
        except (ValueError, KeyError, TypeError):
            return self.send(400, {"error": "요청 형식이 잘못됐어요."})
        if signal not in ("more", "enough"):
            return self.send(400, {"error": "알 수 없는 신호예요."})
        log_event({"event": "feedback", "id": iid, "signal": signal})
        return self.send(200, {"ok": True})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"준비 끝. http://{HOST}:{PORT}  (llama={LLAMA_URL}, 접속코드 {len(codes())}개, 고정답 {len(AUTHOR)}개)", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
