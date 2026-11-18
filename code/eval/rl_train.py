"""Plan B steps 2-3 on ScienceWorld: RL on the LLM decision points of the PinPatch controller, with the learned
constraint head's blocks as a local penalty (internalization) and an optional learned sparse gate for credit
assignment. TRAIN rollouts only (rl_rollout.sh); dev/test use a serial server with the trained adapter.

  rl_train.py IT OUT_ADAPTER [--gate] [--init PREV_ADAPTER] [--lam 0.01]

Episode advantage: GRPO-style, (R - mean_group) / (std_group + 0.1), R = final score / 100, groups = the K samples
of one task. Per decision i of an episode: adv_i = A_ep * w_i + pen_i, pen_i = -0.5 when the action the decision
produced was blocked by the learned head. w_i = 1 (uniform) or, with --gate, the normalised sparse gate g_i:
a gate g_i = sigmoid(a.h_i + b) and a contribution c_i = u.h_i over the policy's hidden state h_i at the decision
(layer 20, last prompt token) are fit so that sum_i g_i c_i explains A_ep, with an L1 penalty lam * sum_i g_i; the
weights are g_i / mean(g) within the episode, so credit concentrates on the decisions that explain the outcome.
Loss: -adv_i * mean token log-prob of the decision's response (on-policy samples, one epoch). LoRA r=16."""
import glob, json, math, os, random, sys

import torch

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
MODEL = R + '/models/Qwen2.5-7B-Instruct'
BLOCK = 'Blocked: this action is predicted'


def load(it):
    eps = []
    pre = os.environ.get('RLP', 'it')  # run-name prefix: it (default) or fp (FOCUS_POLICY variant)
    for f in sorted(glob.glob(f'{R}/runs/rl/{pre}{it}_s*/episodes.jsonl')):
        for l in open(f):
            try:
                eps.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return eps


def samples(eps):
    by = {}
    for e in eps:
        by.setdefault(e['id'], []).append(e)
    out, stats = [], dict(groups=0, flat=0, blocked=0)
    for i, g in by.items():
        rs = [e['score'] / 100.0 for e in g]
        mu = sum(rs) / len(rs); sd = (sum((r - mu) ** 2 for r in rs) / len(rs)) ** 0.5
        stats['groups'] += 1; stats['flat'] += sd < 1e-6
        for e, r in zip(g, rs):
            A = (r - mu) / (sd + 0.1)
            for k, a in enumerate(e.get('asks', [])):
                st = a['step']; blocked = st < len(e['trace']) and e['trace'][st][1].startswith(BLOCK)
                stats['blocked'] += blocked
                out.append(dict(ep=f"{e['id']}_{e.get('seed', '')}_{id(e)}", A=A, pen=-0.5 if blocked else 0.0,
                                msgs=a['msgs'], text=a['text'], n=len(e['asks'])))
    return out, stats


def build(tok, msgs, text):
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pi = tok(p, add_special_tokens=False)['input_ids']; ri = tok(text + tok.eos_token, add_special_tokens=False)['input_ids']
    ids = (pi + ri)[-4096:]; n_resp = min(len(ri), len(ids))
    return ids, n_resp, len(pi)


def gate_weights(model, tok, S, lam):
    """Fit the sparse gate on frozen hidden states (adapter disabled) and return per-sample weights."""
    H = []
    with torch.no_grad(), model.disable_adapter():
        for s in S:
            p = tok.apply_chat_template(s['msgs'], tokenize=False, add_generation_prompt=True)
            ids = tok(p, add_special_tokens=False, return_tensors='pt')['input_ids'][:, -3072:].cuda()
            H.append(model(input_ids=ids, output_hidden_states=True).hidden_states[20][0, -1].float().cpu())
    H = torch.stack(H); H = (H - H.mean(0)) / (H.std(0) + 1e-6)
    eps = sorted({s['ep'] for s in S}); idx = {e: k for k, e in enumerate(eps)}
    E = torch.tensor([idx[s['ep']] for s in S]); A = torch.zeros(len(eps))
    for s in S:
        A[idx[s['ep']]] = s['A']
    a = torch.zeros(H.shape[1], requires_grad=True); u = torch.zeros(H.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam([a, u, b], lr=1e-3)
    for _ in range(400):
        g = torch.sigmoid(H @ a + b); c = H @ u
        pred = torch.zeros(len(eps)).index_add(0, E, g * c)
        loss = ((pred - A) ** 2).mean() + lam * g.mean() + 1e-4 * (a.pow(2).sum() + u.pow(2).sum())
        opt.zero_grad(); loss.backward(); opt.step()
    g = torch.sigmoid(H @ a + b).detach()
    mean_g = torch.zeros(len(eps)).index_add(0, E, g) / torch.zeros(len(eps)).index_add(0, E, torch.ones(len(g)))
    w = g / mean_g[E].clamp_min(1e-6)
    print(f'gate: mean g {g.mean():.3f}, frac g>0.5 {(g > 0.5).float().mean():.3f}, fit mse {((pred - A) ** 2).mean():.4f}')
    return w.tolist()


def main():
    it = int(sys.argv[1]); out = sys.argv[2]
    gate = '--gate' in sys.argv
    init = sys.argv[sys.argv.index('--init') + 1] if '--init' in sys.argv else None
    lam = float(sys.argv[sys.argv.index('--lam') + 1]) if '--lam' in sys.argv else 0.01
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda')
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = (PeftModel.from_pretrained(base, init, is_trainable=True) if init else
             get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM')))
    S, st = samples(load(it))
    w = gate_weights(model, tok, S, lam) if gate else [1.0] * len(S)
    for s, wi in zip(S, w):
        s['adv'] = s['A'] * wi + s['pen']
    S = [s for s in S if abs(s['adv']) > 1e-6]
    if os.environ.get('RL_LIMIT'):  # smoke tests only
        S = S[: int(os.environ['RL_LIMIT'])]
    print(f'it{it}: {st}, {len(S)} decision samples with non-zero advantage, gate={gate}')
    random.Random(it).shuffle(S)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.0)
    model.train(); acc, tot = 8, 0.0
    for k, s in enumerate(S):
        ids, n_resp, _ = build(tok, s['msgs'], s['text'])
        x = torch.tensor([ids]).cuda()
        logits = model(input_ids=x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, x[0, 1:, None])[:, 0][-n_resp:]
        loss = -s['adv'] * lp.mean() / acc
        loss.backward(); tot += loss.item()
        if (k + 1) % acc == 0 or k + 1 == len(S):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        if (k + 1) % 200 == 0:
            print(f'  {k + 1}/{len(S)} loss {tot / (k + 1):.4f}', flush=True)
    model.save_pretrained(out)
    json.dump(dict(it=it, gate=gate, lam=lam, init=init, n=len(S), stats=st), open(out + '/train_info.json', 'w'))
    print('saved', out)


if __name__ == '__main__':
    main()
