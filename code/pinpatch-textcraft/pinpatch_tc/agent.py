"""PinPatch: lifted, feedback-learned dynamics + model-based planning + online correction.

Fix for the v154 blocker: v154 stored operators keyed by sha(initial, goal, menu) and the
literal action string, so a new task context had zero coverage. Here every learned
quantity is LIFTED: it is a function of an action's argument slots (verb, clause role,
count, entity), never of the task context hash, so it transfers to unseen goals.

Priors that are given, not learned (stated explicitly):
  * a generic count-entity chunker: an action is "<verb> <chunk>[, <chunk>] [<connector> <chunk>, ...]"
    with chunk = "<integer> <entity words>" and connector in CONNECTORS;
  * the agent may query its inventory (free, as in the 031 protocol).
Everything else is fitted from TRAIN interaction feedback:
  * effect coefficient per (verb, clause role)  in {-1, 0, +1} x count;
  * precondition form (consumed entities must be held in >= count);
  * a gettability model for "get" actions;
  * a generic-ingredient grounding model (which held item may stand in for a generic slot).
"""
import math, re, random, json
from collections import defaultdict, Counter

CONNECTORS = ('using', 'from', 'with')
RISK = '__risk__'
RISK_PENALTY = 4.0
CHUNK = re.compile(r'^(\d+) ([a-z][a-z ]*?)$')


# ----------------------------------------------------------------- parsing (prior)
def parse_action(a):
    """-> (verb, [(role, count, entity), ...]) or None. Role 0 = before connector, 1 = after."""
    a = a.strip()
    m = re.match(r'^([a-z]+) (.*)$', a)
    if not m:
        return None
    verb, rest = m.group(1), m.group(2)
    clauses = [rest]
    for c in CONNECTORS:
        if f' {c} ' in rest:
            clauses = rest.split(f' {c} ', 1)
            break
    slots = []
    for role, cl in enumerate(clauses):
        for part in cl.split(', '):
            cm = CHUNK.match(part.strip())
            if not cm:
                return None
            slots.append((role, int(cm.group(1)), cm.group(2)))
    return verb, slots


def render(verb, slots):
    by = defaultdict(list)
    for role, n, e in slots:
        by[role].append(f'{n} {e}')
    s = f'{verb} ' + ', '.join(by[0])
    if 1 in by:
        s += ' using ' + ', '.join(by[1])
    return s


def parse_commands(observation):
    cmds = []
    body = observation.split('Goal: ')[0]
    for line in body.split('\n'):
        p = parse_action(line)
        if p:
            cmds.append((line, p))
    goal = re.search(r'Goal: craft (.+?)\.', observation).group(1)
    return cmds, goal


def singular(w):
    if w.endswith('ies'):
        return w[:-3] + 'y'
    if w.endswith('es') and w[:-2].endswith(('sh', 'ch', 'x')):
        return w[:-2]
    if w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w


# ----------------------------------------------------------------- learned model
def _pair():
    return [0, 0]


class LiftedModel:
    def __init__(self):
        # effect evidence: (verb, role) -> Counter{coef: votes}
        self.effect_votes = defaultdict(Counter)
        # precondition evidence: list of (holds_all_consumed, valid)
        self.pre_evidence = Counter()
        # gettability: feature tuple -> [ok, fail]
        self.get_stats = defaultdict(_pair)
        # grounding: logistic weights over pair features + exact-pair memory
        self.w = [0.0, 0.0, 0.0, 0.0]
        self.pair_memory = {}  # (generic, specific) -> bool
        self.pair_data = []

    # ---- learned effect / precondition
    def coef(self, verb, role):
        v = self.effect_votes.get((verb, role))
        if not v:
            return None
        return v.most_common(1)[0][0]

    def known_verb(self, verb):
        return any(k[0] == verb for k in self.effect_votes)

    def requires_holding(self):
        # hypothesis H1: action valid only if every consumed entity is held >= count
        a = self.pre_evidence
        # (holds, valid) counts; H1 predicts valid==holds for crafting-like verbs
        if sum(a.values()) == 0:
            return True
        agree = a[(True, True)] + a[(False, False)]
        return agree >= 0.9 * sum(a.values())

    def predict_delta(self, verb, slots):
        d = Counter()
        for role, n, e in slots:
            c = self.coef(verb, role)
            if c is None:
                return None
            if c:
                d[e] += c * n
        return d

    # ---- gettability
    @staticmethod
    def get_features(entity, ctx):
        produced = entity in ctx['produced']
        as_ingredient = entity in ctx['ingredients']
        tail = ' '.join(entity.split()[:-1] + [singular(entity.split()[-1])])
        # entity name is a strict suffix of some produced item's name ("planks" < "oak planks")
        generic_like = any(p != entity and (' ' + ' '.join(p.split()[:-1] + [singular(p.split()[-1])])).endswith(' ' + tail)
                           for p in ctx['produced'])
        return (produced, as_ingredient, generic_like)

    def p_gettable(self, entity, ctx):
        ok, fail = self.get_stats.get(self.get_features(entity, ctx), (0, 0))  # read must not insert a key
        return (ok + 1) / (ok + fail + 2)

    # ---- grounding of generic slots
    @staticmethod
    def pair_features(g, s, ctx):
        gt, st = g.split(), s.split()
        head = singular(gt[-1]) == singular(st[-1])
        overlap = len(set(map(singular, gt)) & set(map(singular, st))) / max(1, len(set(gt) | set(st)))
        prod = s in ctx['produced']
        return [1.0, float(head), overlap, float(prod)]

    def p_ground(self, g, s, ctx):
        if (g, s) in self.pair_memory:
            return 0.99 if self.pair_memory[(g, s)] else 0.0
        f = self.pair_features(g, s, ctx)
        z = sum(wi * fi for wi, fi in zip(self.w, f))
        return 1 / (1 + math.exp(-z))

    def fit_grounding(self, epochs=200, lr=0.5):
        if not self.pair_data:
            return
        w = [0.0] * 4
        for _ in range(epochs):
            for f, y in self.pair_data:
                z = sum(wi * fi for wi, fi in zip(w, f))
                p = 1 / (1 + math.exp(-z))
                for i in range(4):
                    w[i] += lr * (y - p) * f[i] / len(self.pair_data) * 10
        self.w = w

    # ---- learning from one observed transition
    def observe(self, action, inv_before, inv_after, valid, ctx, substitution=None):
        p = parse_action(action)
        if not p:
            return
        verb, slots = p
        if valid and verb != 'get':
            # cross-task experience: a craft whose effect TRAIN feedback has confirmed (context-free)
            if not hasattr(self, 'verified'):
                self.verified = {}
            self.verified[action] = self.verified.get(action, 0) + 1
        if valid:
            delta = Counter({k: inv_after.get(k, 0) - inv_before.get(k, 0)
                             for k in set(inv_before) | set(inv_after)})
            for role, n, e in slots:
                obs = delta.get(e, 0)
                if obs == 0:
                    c = 0
                elif obs % n == 0 and obs // n in (-1, 1):
                    c = obs // n
                else:
                    c = 1 if obs > 0 else -1  # output count may differ; sign still informative
                self.effect_votes[(verb, role)][c] += 1
        consumed = [(n, e) for role, n, e in slots if (self.coef(verb, role) or 0) < 0]
        if consumed:
            holds = all(inv_before.get(e, 0) >= n for n, e in consumed)
            self.pre_evidence[(holds, valid)] += 1
        if verb == 'get' and len(slots) == 1:
            e = slots[0][2]
            self.get_stats[self.get_features(e, ctx)][0 if valid else 1] += 1
        if substitution:
            for g, s in substitution.items():
                f = self.pair_features(g, s, ctx)
                holds = all(inv_before.get(e, 0) >= n for role, n, e in slots
                            if (self.coef(verb, role) or 0) < 0)
                if holds:  # only informative if the precondition was met
                    self.pair_memory[(g, s)] = bool(valid)
                    self.pair_data.append((f, 1.0 if valid else 0.0))

    def to_json(self):
        return {'effects': {f'{k[0]}|{k[1]}': dict(v) for k, v in self.effect_votes.items()},
                'precondition_evidence': {f'holds={k[0]},valid={k[1]}': v for k, v in self.pre_evidence.items()},
                'requires_holding': self.requires_holding(),
                'get_stats': {str(k): v for k, v in self.get_stats.items()},
                'grounding_weights': self.w, 'pair_memory_size': len(self.pair_memory)}


# ----------------------------------------------------------------- planner
class Planner:
    def __init__(self, model, cmds, goal, rng, explore=0.0, use_substitution=True, use_memory=False, memory_gate=False):
        self.m = model; self.cmds = cmds; self.goal = goal; self.rng = rng
        self.memory_gate = memory_gate  # memory only after a get of the item was actually rejected
        self.explore = explore; self.use_sub = use_substitution
        self.ctx = {'produced': set(), 'ingredients': set()}
        self.producers = defaultdict(list)
        for line, (verb, slots) in cmds:
            for role, n, e in slots:
                c = model.coef(verb, role)
                if c and c > 0:
                    self.ctx['produced'].add(e); self.producers[e].append((line, verb, slots))
                elif c and c < 0:
                    self.ctx['ingredients'].add(e)
        # v2 (post-hoc): verified TRAIN crafts as fallback producers for items with no visible producer
        self.memory_producers = defaultdict(list)
        if use_memory:
            for a in getattr(model, 'verified', {}):
                p = parse_action(a)
                if p:
                    for role, n, e in p[1]:
                        if (model.coef(p[0], role) or 0) > 0 and e not in self.producers:
                            self.memory_producers[e].append((a, p[0], p[1]))
        # episode-local online corrections
        self.get_failed = set(); self.bad_pairs = set(); self.bad_actions = Counter()

    def mem_ready(self, e):
        """A gated memory candidate exists and has not been refuted, so prefer it over grounding."""
        return self.memory_gate and e in self.get_failed and any(
            self.bad_actions[c[0]] < 2 for c in self.memory_producers.get(e, []))

    def p_get(self, e):
        if e in self.get_failed:
            return 0.0
        return self.m.p_gettable(e, self.ctx)

    def gettable(self, e, lo=0.15):
        return self.p_get(e) >= lo

    def candidates(self, g, inv):
        """Specific items that may fill generic slot g, best first."""
        pool = set(self.ctx['produced']) | set(inv) | set(self.ctx['ingredients'])
        pool.discard(g); pool.discard(RISK)
        scored = [(self.m.p_ground(g, s, self.ctx), s) for s in pool if (g, s) not in self.bad_pairs]
        scored = [x for x in scored if x[0] >= 0.3]
        scored.sort(key=lambda x: (-x[0], x[1]))
        if self.explore and len(scored) > 1 and self.rng.random() < self.explore:
            self.rng.shuffle(scored)
        return [s for _, s in scored]

    def plan(self, inv, budget):
        sim = Counter(inv); out = []
        ok = self._need(self.goal, 1, sim, out, frozenset(), budget, depth=0)
        return out if ok else None

    def _need(self, e, q, sim, out, stack, budget, depth):
        if sim[e] >= q and e != self.goal:
            return True
        if len(out) > budget or depth > 12 or e in stack:
            return False
        options = []
        for line, verb, slots in self.producers.get(e, []):
            options.append(('craft', (line, verb, slots)))
        if self.gettable(e) and e != self.goal:
            options.append(('get', None))
        if not self.producers.get(e) and self.p_get(e) < 0.5 and (not self.memory_gate or e in self.get_failed):
            for line, verb, slots in self.memory_producers.get(e, []):
                options.append(('craft', (line, verb, slots)))
        if self.use_sub and not self.producers.get(e) and self.p_get(e) < 0.5 and not self.mem_ready(e):
            # generic slot: satisfy it with a grounded specific item
            for s in self.candidates(e, sim)[:4]:
                options.append(('ground', s))
        best = None
        for kind, arg in options:
            sim2, out2 = Counter(sim), list(out)
            if self._realize(kind, arg, e, q, sim2, out2, stack | {e}, budget, depth):
                cost = len(out2) + sim2[RISK]
                if best is None or cost < best[2]:
                    best = (sim2, out2, cost)
        if best is None:
            return False
        sim.clear(); sim.update(best[0]); out[:] = best[1]
        return True

    def _realize(self, kind, arg, e, q, sim, out, stack, budget, depth):
        if kind == 'get':
            n = q - sim[e]
            a = f'get {n} {e}'
            if self.bad_actions[a] >= 1:
                return False
            out.append(a); sim[e] += n
            sim[RISK] += (1 - self.p_get(e)) * RISK_PENALTY  # expected cost of a failed probe
            return True
        if kind == 'ground':
            # the generic entity is represented by the specific item s
            return self._need(arg, q, sim, out, stack, budget, depth + 1) and self._alias(e, arg, q, sim)
        line, verb, slots = arg
        outn = sum(n for role, n, x in slots if x == e and (self.m.coef(verb, role) or 0) > 0)
        if outn <= 0:
            return False
        times = math.ceil(max(1, q - sim[e]) / outn) if e != self.goal else 1
        inputs = [(n, x) for role, n, x in slots if (self.m.coef(verb, role) or 0) < 0]
        # resolve each input; generic inputs get grounded to a specific held/producible item
        resolved = []
        for n, x in inputs:
            need = n * times
            fills = []
            if self.producers.get(x) or self.gettable(x) or not self.use_sub or self.memory_producers.get(x):
                fills.append(x)
            if self.use_sub and not self.producers.get(x) and self.p_get(x) < 0.5 and not self.mem_ready(x):
                fills += [s for s in self.candidates(x, sim)[:4] if s != x]
            best = None
            for s in fills:
                sim2, out2 = Counter(sim), list(out)
                if self._need(s, need, sim2, out2, stack, budget, depth + 1):
                    cost = len(out2) + sim2[RISK]
                    if best is None or cost < best[0]:
                        best = (cost, s, sim2, out2)
            if best is None:
                return False
            _, s, sim2, out2 = best
            sim.clear(); sim.update(sim2); out[:] = out2
            resolved.append((n, x, s))
        new_slots = []
        for role, n, x in slots:
            sub = next((s for (nn, xx, s) in resolved if xx == x), x)
            new_slots.append((role, n, sub))
        a = render(verb, new_slots)
        if self.bad_actions[a] >= 2:
            return False
        for _ in range(times):
            for nn, xx, s in resolved:
                sim[s] -= nn
                if sim[s] < 0:
                    return False
            sim[e] += outn
            out.append(a)
        return True

    def _alias(self, e, s, q, sim):
        return True  # the specific item is consumed when the parent craft renders it


def analogize(entity, model, limit=5):
    """Candidate recipes for `entity` by analogy: a verified recipe for an item with the same head
    noun, with its modifier swapped ("oak fence" -> "crimson fence", "oak planks" -> "crimson planks")."""
    words = entity.split()
    head, prefix = singular(words[-1]), ' '.join(words[:-1])
    if not prefix:  # a bare head noun ("planks") is a generic slot, handled by grounding
        return []
    seen, out = set(), []
    for a in sorted(getattr(model, 'verified', {})):
        p = parse_action(a)
        if not p:
            continue
        verb, slots = p
        made = [e for role, n, e in slots if (model.coef(verb, role) or 0) > 0]
        if len(made) != 1 or made[0] == entity or singular(made[0].split()[-1]) != head:
            continue
        old = ' '.join(made[0].split()[:-1])
        def swap(e):
            if old and prefix and e.startswith(old + ' '):
                return prefix + e[len(old):]
            return e
        new = [(role, n, entity if e == made[0] else swap(e)) for role, n, e in slots]
        line = render(verb, new)
        if line not in seen:
            seen.add(line); out.append((line, verb, new))
    return out[:limit]


def substitution_of(action, cmds):
    """If `action` is a visible command with some entities replaced, return {generic: specific}."""
    p = parse_action(action)
    if not p or p[0] == 'get':
        return None
    verb, slots = p
    for line, (v2, s2) in cmds:
        if v2 == verb and len(s2) == len(slots) and all(a[0] == b[0] and a[1] == b[1] for a, b in zip(slots, s2)):
            diff = {b[2]: a[2] for a, b in zip(slots, s2) if a[2] != b[2]}
            if len(diff) < len(slots) and all(a[2] == b[2] for a, b in zip(slots, s2) if a[0] == 0):
                return diff or None
    return None
