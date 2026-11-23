"""책 질문 답변 모델 학습·채점. 사용: ft.py train | ft.py eval <시험파일> <결과파일> [lora폴더]"""
import glob, json, os, sys, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# 받아 둔 폴더를 직접 가리켜서 인터넷·로그인 없이 읽는다
_DEF = "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*"
BASE = glob.glob(os.path.expanduser(os.environ.get("QA_BASE_GLOB", _DEF)))[0]
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")


def _ortho(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tp = X.size(0) > X.size(1)
    if tp: X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    if tp: X = X.T
    return X.to(G.dtype)

class Muon:
    """Kimi(Moonshot) 팀이 쓴 Muon: 2차원 가중치는 모멘텀을 직교화해 갱신. 1차원은 AdamW로."""
    def __init__(self, params, lr=2e-2, momentum=0.95, wd=0.0, adamw_lr=1e-4):
        p = [x for x in params if x.requires_grad]
        self.m2 = [x for x in p if x.ndim == 2]; self.m1 = [x for x in p if x.ndim != 2]
        self.lr, self.mom, self.wd = lr, momentum, wd
        self.buf = {id(x): torch.zeros_like(x) for x in self.m2}
        self.adam = torch.optim.AdamW(self.m1, lr=adamw_lr) if self.m1 else None
    def zero_grad(self):
        for x in self.m2:
            if x.grad is not None: x.grad = None
        if self.adam: self.adam.zero_grad()
    def step(self):
        for x in self.m2:
            if x.grad is None: continue
            buf = self.buf[id(x)]; buf.mul_(self.mom).add_(x.grad)
            g = x.grad.add(buf, alpha=self.mom)  # 네스테로프
            u = _ortho(g)
            scale = max(1.0, x.size(0) / x.size(1)) ** 0.5
            if self.wd: x.data.mul_(1 - self.lr * self.wd)
            x.data.add_(u, alpha=-self.lr * scale)
        if self.adam: self.adam.step()

def prompt_ids(x):
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:
        return tok.apply_chat_template(x["messages"], enable_thinking=False, **kw)["input_ids"][0]
    except (TypeError, ValueError):
        return tok.apply_chat_template(x["messages"], **kw)["input_ids"][0]

if sys.argv[1] == "train":
    from peft import LoraConfig, get_peft_model
    _TGT = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    model = get_peft_model(model, LoraConfig(r=int(os.environ.get("QA_RANK", 16)), lora_alpha=int(os.environ.get("QA_ALPHA", 32)),
        lora_dropout=float(os.environ.get("QA_DROPOUT", 0.05)), task_type="CAUSAL_LM",
        target_modules=_TGT[os.environ.get("QA_TARGET", "all")]))
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    _NEFT = float(os.environ.get("QA_NEFT", 0))  # NEFTune: 임베딩에 노이즈 주입(0=끔)
    if _NEFT > 0:
        emb = model.get_input_embeddings()
        def _noise(mod, inp, out):
            if mod.training:
                d = out.shape[-1] * out.shape[-2]
                out = out + torch.empty_like(out).uniform_(-1, 1) * (_NEFT / (d ** 0.5))
            return out
        emb.register_forward_hook(_noise)
    TRAIN = sys.argv[2] if len(sys.argv)>2 else "train.jsonl"
    OUT = sys.argv[3] if len(sys.argv)>3 else "lora"
    SEED = int(sys.argv[4]) if len(sys.argv)>4 else 0
    rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
    EPOCHS, ACC = int(os.environ.get("QA_EPOCHS", 3)), 8
    total = EPOCHS * len(rows) // ACC
    OPT = sys.argv[5] if len(sys.argv) > 5 else "adamw"
    if OPT == "muon":
        opt = Muon([p for p in model.parameters() if p.requires_grad]); sched = None
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(os.environ.get("QA_LR", 1e-4)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10) * max(0.0, 1 - s / total))
    random.seed(SEED); torch.manual_seed(SEED); step = 0; run = 0.0
    model.train()
    for ep in range(EPOCHS):
        random.shuffle(rows)
        for i, x in enumerate(rows):
            p = prompt_ids(x)
            t = tok(x["target"] + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            ids = torch.cat([p, t])[None].to("cuda")
            nk = t.shape[0] + 1                               # 답 토큰 부분만 로짓 계산(긴 문맥 OOM 방지)
            try:
                logits = model(input_ids=ids, logits_to_keep=nk).logits[0]
            except TypeError:
                logits = model(input_ids=ids, num_logits_to_keep=nk).logits[0]
            loss = torch.nn.functional.cross_entropy(logits[:-1].float(), t.to("cuda"))  # 답 부분만 배운다
            (loss / ACC).backward(); run += loss.item()
            if (i + 1) % ACC == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step() if sched else None; opt.zero_grad(); step += 1
                if step % 20 == 0: print(f"바퀴 {ep+1} 걸음 {step}/{total} 손실 {run / (20 * ACC):.3f}", flush=True); run = 0.0
        model.save_pretrained(OUT + f"_e{ep+1}" if os.environ.get("QA_SAVE_EACH") else OUT)
    print("학습 끝", flush=True)
else:
    if len(sys.argv) > 4:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sys.argv[4])
    model.eval(); out = []
    for l in open(sys.argv[2], encoding="utf-8"):
        x = json.loads(l); p = prompt_ids(x)[None].to("cuda")
        with torch.no_grad():
            g = model.generate(input_ids=p, max_new_tokens=300, do_sample=False)
        x["out"] = tok.decode(g[0, p.shape[1]:], skip_special_tokens=True).strip(); del x["messages"]
        out.append(x); print(x.get("kind"), "|", x["q"], "|", x["out"][:60].replace("\n", " "), flush=True)
    open(sys.argv[3], "w", encoding="utf-8").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
