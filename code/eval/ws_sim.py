"""TRAIN-only data for the WebShop item scorer: for each train goal and query form, buy each of the top-10 results
(with the goal's option values clicked when present) in a fresh episode and record the reward. No LLM involved."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import run_eval as E

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'train'  # P1-2: 'train_lab' = train[2400:4400]
out = f'{E.R}/runs/ws_sim/{SPLIT}_{N}.jsonl'


def one(i):
    rows = []
    env = E.WebShop()
    obs = env.reset(i); goal = E.ws_parse_goal(E.ws_instruction(obs))
    for qf in ('desc_opts', 'desc'):
        q = goal['desc'] if qf == 'desc' else goal['desc'] + ' ' + ' '.join(goal['opts'].values())
        env.reset(i); obs, _, _ = env.step(f'search[{q}]')
        res = E.ws_results(obs)
        for r, (asin, title, price) in enumerate(res[:10]):
            env.reset(i); env.step(f'search[{q}]'); env.step(f'click[{asin}]')
            clk = [c for c in env.available() if c not in E.WS_FIXED]
            picks = [E.ws_match_option(v, clk) for v in goal['opts'].values()]
            for c in picks:
                if c:
                    env.step(f'click[{c}]')
            _, reward, _ = env.step('click[buy now]')
            rows.append(dict(id=i, q=qf, rank=r, asin=asin, title=title, price=price, n_match=sum(p is not None for p in picks),
                             n_opt=len(goal['opts']), n_clk=len(clk), reward=reward, goal=goal))
    env.close()
    return rows


import os
os.makedirs(os.path.dirname(out), exist_ok=True)
ids = E.ids('webshop', SPLIT)[:N]
with open(out, 'w') as f, ThreadPoolExecutor(8) as ex:
    for k, rows in enumerate(ex.map(one, ids)):
        for r in rows:
            f.write(json.dumps(r) + '\n')
        if k % 25 == 0:
            print(k, flush=True)
print('SIM_DONE', out)
