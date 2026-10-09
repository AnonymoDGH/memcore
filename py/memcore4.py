"""MemCore v4: data-efficient small models on GPU (PyTorch).

Same design as the C engine (v3), scaled up:
  * byte-level looped transformer: `blocks` shared blocks applied up to
    `loops` times, with a learned per-loop embedding (variable depth)
  * verifiable synthetic tasks + reference solver (the verifier)
  * adaptive curriculum, hard-example mining, STaR at the frontier
  * adaptive depth and majority voting at inference
  * episodic kNN memory for instant learning of new text

  python memcore4.py train --steps 6000
  python memcore4.py eval  --votes 5
  python memcore4.py solve "4821+977"
"""
import argparse
import collections
import math
import os
import random
import queue
import re
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------- tasks

# name -> curriculum ceiling (levels above it test generalization)
TASKS = collections.OrderedDict(
    add=6, sub=6, mul=4, mul2=3, chain=4, cmp=6, rev=10, sort=10, count=12)


def rnd_digits(nd):
    lo = 0 if nd <= 1 else 10 ** (nd - 1)
    return random.randint(lo, 10 ** nd - 1)


def rnd_str(n, alphabet):
    return "".join(chr(97 + random.randrange(alphabet)) for _ in range(n))


def task_gen(task, L):
    """Prompt for `task` at difficulty `L` (digits or string length)."""
    L = max(1, L)
    a, b = rnd_digits(L), rnd_digits(random.randint(1, L))
    if random.random() < 0.5:
        a, b = b, a
    if task == "add":
        return f"{a}+{b}="
    if task == "sub":
        a, b = max(a, b), min(a, b)
        return f"{a}-{b}="
    if task == "mul":
        return f"{rnd_digits(L)}*{random.randrange(10)}="
    if task == "mul2":
        return f"{rnd_digits(L)}*{rnd_digits(random.randint(1, L))}="
    if task == "chain":
        c = random.randint(0, a + b)
        return f"{a}+{b}-{c}="
    if task == "cmp":
        a = rnd_digits(L)
        b = a if random.random() < 0.1 else rnd_digits(L)
        if L > 1 and random.random() < 0.5:  # shared prefix: harder
            m = 10 ** random.randint(1, L - 1)
            b = a // m * m + random.randrange(m)
        return f"{a}?{b}="
    if task == "rev":
        return f"r:{rnd_str(L, 26)}="
    if task == "sort":
        return f"s:{rnd_str(L, 10)}="
    return f"c:{rnd_str(L, 3)}="


def task_of(prompt):
    if prompt[:2] in ("r:", "s:", "c:"):
        return {"r:": "rev", "s:": "sort", "c:": "count"}[prompt[:2]]
    if "?" in prompt:
        return "cmp"
    ops = re.findall(r"[+\-*]", prompt)
    if ops == ["*"]:
        a, b = prompt[:-1].split("*")
        return "mul" if len(b) == 1 else "mul2"
    if len(ops) == 2:
        return "chain"
    return {"+": "add", "-": "sub"}.get(ops[0] if ops else "", None)


def solve(prompt):
    """Reference solver = verifier. Arithmetic answers are written least
    significant digit first (carry order)."""
    body = prompt[:-1]
    if prompt[:2] in ("r:", "s:", "c:"):
        s = body[2:]
        if prompt[0] == "r":
            return s[::-1]
        if prompt[0] == "s":
            return "".join(sorted(s))
        return str(s.count("a"))
    if "?" in body:
        a, b = map(int, body.split("?"))
        return "<" if a < b else ">" if a > b else "="
    toks = re.findall(r"\d+|[+\-*]", body)
    acc = int(toks[0])
    for op, v in zip(toks[1::2], toks[2::2]):
        v = int(v)
        acc = acc + v if op == "+" else acc - v if op == "-" else acc * v
    return str(acc)[::-1]


def pretty(prompt, ans):
    return ans[::-1] if prompt[0].isdigit() and "?" not in prompt else ans


def parse_user(text):
    t = text.strip().rstrip("=")
    for w, tag in (("rev ", "r:"), ("sort ", "s:"), ("count ", "c:")):
        if t.startswith(w):
            return f"{tag}{t[len(w):]}="
    if t.startswith("cmp "):
        a, b = t[4:].split()
        return f"{a}?{b}="
    t = t.replace(" ", "").replace("x", "*")
    if re.fullmatch(r"\d+([+\-*]\d+)+", t):
        return t + "="
    return None

# ----------------------------------------------------------------- model


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.n1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n2 = nn.RMSNorm(d)
        self.gu = nn.Linear(d, 2 * ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=mask is None)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        g, u = self.gu(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class Net(nn.Module):
    def __init__(self, d=384, heads=6, blocks=4, loops=3, ff=1024, seq=160):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, blocks=blocks, loops=loops, ff=ff, seq=seq)
        self.emb = nn.Embedding(256, d)
        self.loop_emb = nn.Parameter(torch.zeros(loops, d))
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(blocks))
        self.norm = nn.RMSNorm(d)
        hd = d // heads
        inv = 10000 ** (-torch.arange(0, hd // 2) / (hd // 2))
        ang = torch.arange(seq)[:, None] * inv[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "loop_emb" not in n:
                std = 0.02 / math.sqrt(2 * blocks * loops) if n.endswith(("o.weight", "down.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx, loops=None, mask=None, hidden=False):
        loops = min(loops or self.cfg["loops"], self.cfg["loops"])
        T = idx.shape[1]
        cos, sin = self.cos[:T].to(self.emb.weight.dtype), self.sin[:T].to(self.emb.weight.dtype)
        x = self.emb(idx)
        for r in range(loops):
            x = x + self.loop_emb[r]
            for blk in self.blocks:
                x = blk(x, cos, sin, mask)
        h = self.norm(x)
        logits = h @ self.emb.weight.t()
        return (logits, h) if hidden else logits

# ----------------------------------------------------------------- decoding


def left_pad(seqs, dev):
    T = max(len(s) for s in seqs)
    idx = torch.zeros(len(seqs), T, dtype=torch.long)
    valid = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        idx[i, T - len(s):] = torch.tensor(s)
        valid[i, T - len(s):] = True
    return idx.to(dev), valid.to(dev)


def pad_mask(valid):
    T = valid.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=valid.device).tril()
    eye = torch.eye(T, dtype=torch.bool, device=valid.device)
    # pads attend only to themselves so no row is fully masked
    return (causal[None] & valid[:, None, :]) | eye[None]


@torch.no_grad()
def answer(net, prompts, loops=None, temp=0.0, max_new=16):
    """Batched decoding. Returns (answers, min token prob per answer)."""
    dev = net.emb.weight.device
    seqs = [list(b"\n" + p.encode()) for p in prompts]
    outs = [[] for _ in prompts]
    conf = [1.0] * len(prompts)
    done = [False] * len(prompts)
    for _ in range(max_new):
        idx, valid = left_pad(seqs, dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = net(idx, loops, pad_mask(valid)[:, None])[:, -1].float()
        p = lg.softmax(-1)
        nxt = torch.multinomial((lg / temp).softmax(-1), 1)[:, 0] if temp > 0 else p.argmax(-1)
        pn = p.gather(1, nxt[:, None])[:, 0]
        for i in range(len(prompts)):
            if done[i]:
                continue
            t = int(nxt[i])
            conf[i] = min(conf[i], float(pn[i]))
            if t == 10:
                done[i] = True
            else:
                outs[i].append(t)
                seqs[i].append(t)
        if all(done):
            break
    return [bytes(o).decode("latin-1") for o in outs], conf


@torch.no_grad()
def solve_batch(net, prompts, votes=1, conf_thresh=0.9):
    """Adaptive depth: shallow pass first; unsure rows re-run at full depth.
    votes > 1 adds sampled answers and takes the majority."""
    L = net.cfg["loops"]
    ans, conf = answer(net, prompts, loops=1)
    depth = [1] * len(prompts)
    redo = [i for i, c in enumerate(conf) if c < conf_thresh]
    if redo and L > 1:
        a2, c2 = answer(net, [prompts[i] for i in redo], loops=L)
        for j, i in enumerate(redo):
            ans[i], conf[i], depth[i] = a2[j], c2[j], L
    agree = [1.0] * len(prompts)
    if votes > 1:
        pools = [collections.Counter([a]) for a in ans]
        for _ in range(votes - 1):
            s, _ = answer(net, prompts, loops=L, temp=0.7)
            for i, a in enumerate(s):
                pools[i][a] += 1
        for i, c in enumerate(pools):
            ans[i], n = c.most_common(1)[0]
            agree[i] = n / votes
            depth[i] = L
    return ans, conf, depth, agree

# ----------------------------------------------------------------- memory


class Memory:
    """Episodic kNN memory: (hidden state -> next byte), no gradients."""

    def __init__(self, d, dev):
        self.k = torch.zeros(0, d, device=dev)
        self.v = torch.zeros(0, dtype=torch.long, device=dev)

    @torch.no_grad()
    def learn(self, net, text):
        b = list(text.encode())
        seq = net.cfg["seq"]
        for s in range(0, max(1, len(b) - 1), seq // 2):
            chunk = b[s:s + seq]
            if len(chunk) < 2:
                break
            idx = torch.tensor([chunk], device=self.k.device)
            _, h = net(idx, hidden=True)
            keep = 0 if s == 0 else seq // 2
            self.k = torch.cat([self.k, h[0, keep:-1].float()])
            self.v = torch.cat([self.v, idx[0, keep + 1:]])

    def mix(self, h, p, lam=0.9, k=8):
        if len(self.v) == 0:
            return p
        d2 = ((self.k - h) ** 2).sum(-1)
        dk, ik = d2.topk(min(k, len(d2)), largest=False)
        tau = 0.05 * h.numel()
        w = (-(dk - dk.min()) / tau).exp()
        pk = torch.zeros_like(p).index_add_(0, self.v[ik], w)
        g = lam * math.exp(-float(dk.min()) / tau)
        return (1 - g) * p + g * pk / w.sum()


@torch.no_grad()
def generate(net, prompt, n=200, temp=0.0, mem=None):
    dev = net.emb.weight.device
    ctx = list(b"\n" + prompt.encode())
    for _ in range(n):
        idx = torch.tensor([ctx[-net.cfg["seq"]:]], device=dev)
        lg, h = net(idx, hidden=True)
        p = lg[0, -1].float().softmax(-1)
        if mem is not None:
            p = mem.mix(h[0, -1].float(), p)
        if temp > 0:
            p = p ** (1 / temp)
            t = int(torch.multinomial(p / p.sum(), 1))
        else:
            t = int(p.argmax())
        ctx.append(t)
    return bytes(ctx[len(prompt) + 1:]).decode("latin-1")

# ----------------------------------------------------------------- training


class Trainer:
    def __init__(self, net, args, levels=None, text=None):
        self.net, self.a = net, args
        self.levels = levels or {t: 1 for t in TASKS}
        self.hard = collections.deque(maxlen=4096)  # failed problems
        self.self_ = collections.deque(maxlen=4096)  # STaR: solved frontier
        self.text = text

    def sample(self):
        r = random.random()
        if r < 0.15 and self.hard:
            return random.choice(self.hard)
        if r < 0.25 and self.self_:
            return random.choice(self.self_)
        t = random.choice(list(TASKS))
        L = self.levels[t]
        return task_gen(t, L if random.random() < 0.6 else random.randint(1, L))

    def row(self, T):
        if self.text is not None and random.random() < self.a.text_mix:
            s = random.randrange(len(self.text) - T - 1)
            b = list(self.text[s:s + T + 1])
            return b[:-1], b[1:]
        s, m = [10], [0]
        while len(s) <= T:
            p = self.sample()
            a = solve(p)
            for c in p.encode():
                s.append(c); m.append(0)
            for c in (a + "\n").encode():
                s.append(c); m.append(1)
        s, m = s[:T + 1], m[:T + 1]
        return s[:-1], [c if k else -100 for c, k in zip(s[1:], m[1:])]

    def batch(self):
        rows = [self.row(self.a.T) for _ in range(self.a.batch)]
        x = torch.tensor([r[0] for r in rows])
        y = torch.tensor([r[1] for r in rows])
        return x.to(self.a.dev, non_blocking=True), y.to(self.a.dev, non_blocking=True)

    def curriculum(self, n=128):
        net = self.net
        net.eval()
        line = []
        for t in TASKS:
            L = self.levels[t]
            ps = [task_gen(t, L) for _ in range(n)]
            ans, _ = answer(net, ps)
            ok = [solve(p) == a for p, a in zip(ps, ans)]
            self.hard.extend(p for p, o in zip(ps, ok) if not o)
            acc = sum(ok) / n
            star = 0
            if L < TASKS[t]:  # STaR: attempt the next level, keep verified
                fp = [task_gen(t, L + 1) for _ in range(64)]
                for _ in range(2):
                    fa, _ = answer(net, fp, temp=0.8)
                    good = [p for p, a in zip(fp, fa) if solve(p) == a]
                    self.self_.extend(good)
                    star += len(good)
            line.append(f"{t}@{L}={acc:.0%}" + (f"(+{star})" if star else ""))
            if acc >= 0.9 and L < TASKS[t]:
                self.levels[t] = L + 1
        net.train()
        return " ".join(line)

    def run(self, steps, ckpt):
        net, a = self.net, self.a
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95),
                                weight_decay=0.05, fused=a.dev == "cuda")
        scaler = torch.amp.GradScaler(enabled=a.dev == "cuda")
        L = net.cfg["loops"]
        # build batches on the CPU while the GPU trains (pinned, async copy)
        q = queue.Queue(maxsize=4)

        def producer():
            while True:
                rows = [self.row(a.T) for _ in range(a.batch)]
                q.put((torch.tensor([r[0] for r in rows]).pin_memory() if a.dev == "cuda" else torch.tensor([r[0] for r in rows]),
                       torch.tensor([r[1] for r in rows])))

        threading.Thread(target=producer, daemon=True).start()
        t0, tok, lsum = time.time(), 0, 0.0
        for s in range(1, steps + 1):
            lr = a.lr * min(1, s / 300) * (0.1 + 0.45 * (1 + math.cos(math.pi * s / steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = q.get()
            x, y = x.to(a.dev, non_blocking=True), y.to(a.dev, non_blocking=True)
            loops = L if random.random() < 0.7 else random.randint(1, L)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.dev == "cuda"):
                lg = net(x, loops)
                loss = F.cross_entropy(lg.float().view(-1, 256), y.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            lsum += loss.item()
            tok += x.numel()
            if s % 100 == 0:
                dt = time.time() - t0
                print(f"step {s:6d} loss {lsum / 100:.4f} lr {lr:.2e} {tok / dt:,.0f} tok/s {dt:.0f}s", flush=True)
                lsum = 0.0
            if s % a.eval_every == 0 or s == steps:
                print("  curriculum: " + self.curriculum() +
                      f"  hard={len(self.hard)} self={len(self.self_)}", flush=True)
                save(net, self.levels, ckpt)

# ----------------------------------------------------------------- io / cli


def save(net, levels, path):
    torch.save({"cfg": net.cfg, "model": net.state_dict(), "levels": levels}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    net = Net(**ck["cfg"]).to(dev)
    net.load_state_dict(ck["model"])
    return net, ck["levels"]


def eval_report(net, n, votes):
    net.eval()
    print(f"{'task':6s}" + "".join(f"  L{l:<4d}" for l in range(1, 15)) + "   (* = beyond training ceiling)")
    tot, cells = 0.0, 0
    for t, top in TASKS.items():
        row = f"{t:6s}"
        for L in range(1, min(top + 2, 14) + 1):
            ps = [task_gen(t, L) for _ in range(n)]
            ans, _, _, _ = solve_batch(net, ps, votes)
            acc = sum(solve(p) == a for p, a in zip(ps, ans)) / n
            row += f"  {acc * 100:3.0f}%{'*' if L > top else ' '}"
            if L <= top:
                tot += acc; cells += 1
        print(row, flush=True)
    print(f"mean accuracy within training ceiling: {100 * tot / cells:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "solve", "gen", "info"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("-o", "--ckpt", default="memcore4.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--text")
    ap.add_argument("--text-mix", type=float, default=0.3)
    ap.add_argument("--memory", help="text file memorized before gen")
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    if a.cmd == "train":
        if os.path.exists(a.ckpt):
            net, levels = load(a.ckpt, a.dev)
            print(f"[resuming {a.ckpt}]")
        else:
            net = Net(a.d, a.heads, a.blocks, a.loops, a.ff, max(160, a.T)).to(a.dev)
            levels = None
        text = open(a.text, "rb").read() if a.text else None
        print(f"params {sum(p.numel() for p in net.parameters()):,}  cfg {net.cfg}  device {a.dev}")
        Trainer(net, a, levels, text).run(a.steps, a.ckpt)
        return

    net, levels = load(a.ckpt, a.dev)
    if a.cmd == "info":
        print(net.cfg, f"params {sum(p.numel() for p in net.parameters()):,}", levels)
    elif a.cmd == "eval":
        eval_report(net, a.n, a.votes)
    elif a.cmd == "solve":
        p = parse_user(a.arg)
        if p is None:
            raise SystemExit(f"cannot parse {a.arg!r}")
        ans, conf, depth, agree = solve_batch(net.eval(), [p], a.votes)
        ok = ans[0] == solve(p)
        print(f"{p} {pretty(p, ans[0])}   [{task_of(p)}, depth {depth[0]}, conf {conf[0]:.2f}, "
              f"agree {agree[0]:.0%}] {'OK' if ok else 'WRONG (expected ' + pretty(p, solve(p)) + ')'}")
    elif a.cmd == "gen":
        mem = None
        if a.memory:
            mem = Memory(net.cfg["d"], a.dev)
            mem.learn(net.eval(), open(a.memory, encoding="utf-8").read())
        print(a.arg + generate(net.eval(), a.arg, a.n, a.temp, mem))


if __name__ == "__main__":
    main()
