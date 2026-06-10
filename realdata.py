"""Assemble REAL (demos, program) pairs from re-arc + arc-dsl."""
import json, os, random, re

ARC_DSL_SOLVERS = "arc-dsl/solvers.py"
RE_ARC_TASKS = "re-arc/re_arc_data/re_arc/tasks"

def load_programs(path=ARC_DSL_SOLVERS):
    text = open(path).read()
    progs = {}
    for b in re.split(r'\ndef solve_', text)[1:]:
        task_id = b[:8]
        ops = []
        for ln in b.splitlines()[1:]:
            s = ln.strip()
            if s.startswith('return'): break
            if s and '=' in s:
                ops.append(s.split('=',1)[1].strip())
        progs[task_id] = ops
    return progs

def list_task_ids(d=RE_ARC_TASKS):
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith('.json'))

def sample_demo_set(task_id, k, d=RE_ARC_TASKS):
    ex = json.load(open(os.path.join(d, task_id+'.json')))
    return [{"input": e["input"], "output": e["output"]} for e in random.sample(ex, k)]

if __name__ == "__main__":
    random.seed(0)
    progs = load_programs()
    ids = list_task_ids()
    common = [t for t in ids if t in progs]
    print("programs parsed     :", len(progs))
    print("tasks with re-arc   :", len(ids))
    print("usable (have both)  :", len(common))
    # program length distribution
    import statistics
    lens = [len(progs[t]) for t in common]
    print("program length: min %d, median %d, max %d" % (min(lens), int(statistics.median(lens)), max(lens)))
    # show a simple and a complex example
    simple = min(common, key=lambda t: len(progs[t]))
    print("\nSIMPLE task", simple, "-> program ops:", progs[simple])
    demos = sample_demo_set(simple, 3)
    print("  sampled", len(demos), "demos; first input is", len(demos[0]['input']),"x",len(demos[0]['input'][0]))
