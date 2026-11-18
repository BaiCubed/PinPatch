"""OntoGuard ALFWorld demonstrations (TRAIN slice) in the ALFWorld env: the MetaAct repo's demos_alf procedure
(textworld + AlfredExpert HANDCODED plan) on the game files the AgentGym server uses for our train ids.
usage (alf env python): og_demos_alf.py TASKS_JSONL OUT"""
import json, os, re, sys
TASKS, OUT = sys.argv[1], sys.argv[2]  # alfworld's import rewrites sys.argv

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
sys.path.insert(0, R + '/code/MetaAct/src')
os.environ.setdefault('ALFWORLD_DATA', R + '/data/alfworld')
import textworld
import textworld.gym as tw_gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType
from metaact.envs.alfworld_env import _clean_obs

maps = json.load(open(R + '/code/AgentGym/agentenv-alfworld/configs/mappings_train.json'))
tasks = [json.loads(l) for l in open(TASKS)]
out = []
for t in tasks:
    m = maps[t['ag_idx']]
    gf = os.path.join(os.environ['ALFWORLD_DATA'], 'json_2.1.1', 'train', m['task_type'], m['task_id'], 'game.tw-pddl')
    req = textworld.EnvInfos(won=True, admissible_commands=True, extras=['gamefile', 'expert_plan'])
    env = tw_gym.make(tw_gym.register_game(gf, req, max_episode_steps=60,
                                           wrappers=[AlfredDemangler(), AlfredExpert(AlfredExpertType.HANDCODED)]))
    obs, infos = env.reset(); obs0 = _clean_obs(obs); steps, won = [], False
    for _ in range(60):
        plan = infos.get('extra.expert_plan') or []
        if not plan:
            break
        obs, _, done, infos = env.step(plan[0]); steps.append((plan[0], _clean_obs(obs)))
        won = bool(infos.get('won', False))
        if done or won:
            break
    env.close()
    g = re.search(r'Your task is to:\s*(.+)', obs0)
    out.append(dict(id=t['id'], task_type=m['task_type'].split('-')[0], goal=g.group(1).strip() if g else '', obs0=obs0, steps=steps,
                    won=won))
    print('  alf demo', t['id'], won, len(steps), flush=True)
ok = [d for d in out if d['won']]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    for d in ok:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print(f'[alfworld] demos={len(out)} won={len(ok)} -> {OUT}')
