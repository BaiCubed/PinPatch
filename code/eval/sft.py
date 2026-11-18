"""P0-3 trained baseline: LoRA SFT of Qwen2.5-7B-Instruct on AgentGym's AgentTraj-L (alfworld, sciworld, textcraft,
webshop train files), with every conversation whose item_id is one of our TEST or DEV tasks removed.
Loss only on assistant ("gpt") turns marked loss=True. Fixed hyperparameters (pre-registered): r=16, alpha=32, all
linear projections, lr 1e-4 cosine, 2 epochs, effective batch 16, max length 4096 (longer conversations truncated).
  sft.py prep            -> runs/sft/train.jsonl (+ exclusion report)
  sft.py train OUT_DIR   -> LoRA adapter"""
import json, math, os, random, sys

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')); OUT = f'{R}/runs/sft'; MODEL = f'{R}/models/Qwen2.5-7B-Instruct'
ENVS = ('alfworld', 'sciworld', 'textcraft', 'webshop')


def prep():
    os.makedirs(OUT, exist_ok=True); rows, rep = [], {}
    mt = json.load(open(f'{R}/code/AgentGym/agentenv-alfworld/configs/mappings_test.json'))
    mtr = json.load(open(f'{R}/code/AgentGym/agentenv-alfworld/configs/mappings_train.json'))
    alf_dev = set(json.load(open(f'{R}/data/splits/alfworld.json'))['dev'])  # our ALFWorld dev = indices of the train mapping
    alf_bad = {m['task_type'] + '_' + m['task_id'] for m in mt} | {m['task_type'] + '_' + m['task_id'] for m in mtr if m['item_id'] in alf_dev}
    for e in ENVS:
        d = json.load(open(f'{R}/data/AgentTraj-L/{e}_train.json'))
        if e == 'alfworld':
            bad = alf_bad
        else:
            s = json.load(open(f'{R}/data/splits/{e}.json'))
            bad = {f'{e}_{i}' for sp in ('test', 'dev') for i in s.get(sp, [])}
        keep = [x for x in d if x['item_id'] not in bad]
        rep[e] = dict(total=len(d), removed_test_or_dev=len(d) - len(keep))
        for x in keep:
            msgs = [dict(role='user' if c['from'] == 'human' else 'assistant', content=c['value'], loss=bool(c.get('loss')))
                    for c in x['conversations']]
            rows.append(dict(env=e, item_id=x['item_id'], messages=msgs))
    random.Random(0).shuffle(rows)
    with open(f'{OUT}/train.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    json.dump(rep, open(f'{OUT}/prep_report.json', 'w'), indent=1)
    print(rep, len(rows), 'conversations')


def encode(tok, msgs, maxlen=4096):
    ids, lab = [], []
    for k, m in enumerate(msgs):
        # render turn by turn with the chat template; the system prompt comes from the template on the first turn
        full = tok.apply_chat_template([{'role': x['role'], 'content': x['content']} for x in msgs[:k + 1]], tokenize=False)
        prev = tok.apply_chat_template([{'role': x['role'], 'content': x['content']} for x in msgs[:k]], tokenize=False) if k else ''
        piece = tok(full[len(prev):], add_special_tokens=False)['input_ids']
        ids += piece; lab += piece if (m['role'] == 'assistant' and m['loss']) else [-100] * len(piece)
        if len(ids) >= maxlen:
            break
    return ids[:maxlen], lab[:maxlen]


def train(out_dir):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(f'{OUT}/train.jsonl')]
    data = [encode(tok, r['messages']) for r in rows]
    data = [x for x in data if any(l != -100 for l in x[1])]
    print(len(data), 'examples, tokens', sum(len(x[0]) for x in data), flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda')
    m.gradient_checkpointing_enable(); m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
                                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    EPOCHS, ACC = 2, 16
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    steps = math.ceil(len(data) * EPOCHS / ACC); sch = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
    m.train(); k = 0; run = 0.0
    for ep in range(EPOCHS):
        order = list(range(len(data))); random.Random(ep).shuffle(order)
        for j, i in enumerate(order):
            ids, lab = data[i]
            x = torch.tensor([ids], device='cuda'); y = torch.tensor([lab], device='cuda')
            loss = m(input_ids=x, labels=y).loss / ACC; loss.backward(); run += loss.item()
            if (j + 1) % ACC == 0 or j == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); k += 1
                if k % 20 == 0:
                    print(f'epoch {ep} step {k}/{steps} loss {run / 20:.4f}', flush=True); run = 0.0
        m.save_pretrained(f'{out_dir}/epoch{ep + 1}')
    m.save_pretrained(out_dir); print('SFT_DONE', out_dir)


if __name__ == '__main__':
    prep() if sys.argv[1] == 'prep' else train(sys.argv[2])
