"""Post-freeze analysis. Descriptive only: nothing here feeds back into the method."""
import json, math, sys, random, hashlib
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinpatch_tc.paths import WORK, ADAPT_RESULTS, V154_CODE, DATA
if V154_CODE: sys.path.insert(0, V154_CODE)
try:
    import composition_rules_v154 as v154  # the ORIGINAL frozen file from the closeout package
except ImportError:
    v154 = None
D = WORK
ad = json.load(open(ADAPT_RESULTS))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [round(100*(c-h), 1), round(100*(c+h), 1)]
out = {}
main = json.load(open(D + 'results/test_pinpatch_tc.json'))
ours = {e['id']: e['won'] for e in main['episodes']}
adapt = {k: ad[k]['reward'] > 0 for k in ad if k.startswith('env_')}
w = sum(ours[k] and not adapt[k] for k in ours); l = sum(adapt[k] and not ours[k] for k in ours)
t = sum(ours[k] == adapt[k] for k in ours)
# exact two-sided sign test on discordant pairs
n = w + l; p = min(1, 2 * sum(math.comb(n, i) for i in range(0, min(w, l) + 1)) / 2**n)
out['paired_vs_ADaPT_gpt35'] = {'ours_wins': w, 'ours_losses': l, 'ties': t, 'adapt_success': sum(adapt.values()), 'sign_test_p': p}
arms = {}
for a in ['pinpatch_tc', 'no_substitution', 'no_online_correction', 'literal_menu_031']:
    r = json.load(open(D + f'results/test_{a}.json'))
    arms[a] = {'success': r['success'], 'rate': r['rate'], 'wilson95': wilson(r['success'], 200),
               'invalid_actions': r['invalid_actions'], 'mean_steps_success': round(r['mean_steps_success'], 2)}
seeds = [json.load(open(D + f'results/test_pinpatch_tc{"" if s == 0 else f"_s{s}"}.json'))['success'] for s in range(5)]
dseeds = [json.load(open(D + f'results/dev_pinpatch_tc{"" if s == 0 else f"_s{s}"}.json'))['success'] for s in range(5)]
out['test_arms'] = arms
out['test_seeds'] = {'success': seeds, 'mean_rate': sum(seeds) / 1000, 'min': min(seeds), 'max': max(seeds)}
out['dev_seeds'] = {'success': dseeds, 'n': 111}
# failures by whether the task has a generic ingredient (literal command contains an ungettable tag slot)
import re
test = {x['id']: x for x in json.load(open(DATA + 'test.json'))}
fails = [e for e in main['episodes'] if not e['won']]
out['failures'] = [{'id': e['id'], 'goal': e['goal'], 'steps': e['steps'], 'invalid': e['invalid'],
                    'fallback_actions': e.get('fallback', 0)} for e in fails]
out['adapt_on_our_failures'] = sum(adapt[e['id']] for e in fails)
# ---- v154 blocker reproduction with the frozen code: context coverage of TRAIN-fitted operators
def row(inst):
    obs = inst['commands'] + 'Goal: ' + inst['problem']
    return {'initial': obs, 'goal': inst['problem'], 'menu': [l for l in inst['commands'].split('\n')[1:] if l], 'inventory': 'Inventory: '}
if v154 is None:
    print('V154_CODE not set: skipping the v154 context-coverage check'); cov = None
else:
    train_ctx = {v154.context(row(x)) for x in json.load(open(DATA + 'train.json'))}
    cov = {s: sum(v154.context(row(x)) in train_ctx for x in json.load(open(DATA + f'{s}.json'))) for s in ['dev', 'test']}
out['v154_context_coverage_of_train_operators'] = None if cov is None else {'train_contexts': len(train_ctx), 'dev_covered': cov['dev'], 'test_covered': cov['test'],
    'note': 'v154 operators are keyed by sha256(initial, goal, menu); a context never seen in TRAIN has zero operators, so every decision falls back to the LM, matching the 031 record (0 composition triggers).'}
# generic-ingredient prevalence on test (instances whose solution needs a substituted command)
json.dump(out, open(D + 'results/analysis.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'failures'}, indent=1)); print('failures:', [(f['goal'], f['steps'], f['fallback_actions']) for f in out['failures']])
