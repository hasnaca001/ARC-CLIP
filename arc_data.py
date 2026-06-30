"""
arc_data.py
-----------
Assemble real (demonstrations, program) pairs for ARC-CLIP from two sources:

  arc-dsl  (Hodel)  -> one verified DSL solver program per ARC training task
  re-arc   (Hodel)  -> 1000 procedurally generated (input, output) pairs per task

Together they give labeled data: for any task we have many demonstration sets
and the ground-truth program that produces them.

Expects these folders cloned next to the project:
  arc-dsl/solvers.py
  re-arc/re_arc_data/re_arc/tasks/<task_id>.json
"""

import json
import os
import re
import random

ARC_DSL_SOLVERS = "arc-dsl/solvers.py"
RE_ARC_TASKS = "re-arc/re_arc_data/re_arc/tasks"


def load_programs(path=ARC_DSL_SOLVERS):
    """Parse solvers.py into {task_id: [list of operation strings]}."""
    text = open(path).read()
    programs = {}
    for block in re.split(r"\ndef solve_", text)[1:]:
        task_id = block[:8]
        ops = []
        for line in block.splitlines()[1:]:
            s = line.strip()
            if s.startswith("return"):
                break
            if s and "=" in s:
                ops.append(s.split("=", 1)[1].strip())
        programs[task_id] = ops
    return programs


def list_task_ids(tasks_dir=RE_ARC_TASKS):
    return sorted(f[:-5] for f in os.listdir(tasks_dir) if f.endswith(".json"))


def load_task_examples(task_id, limit=None, tasks_dir=RE_ARC_TASKS):
    """Return the generated (input, output) examples for one task."""
    ex = json.load(open(os.path.join(tasks_dir, task_id + ".json")))
    return ex[:limit] if limit else ex


def sample_demo_set(task_id, k):
    """Sample k demonstration pairs for a task."""
    ex = load_task_examples(task_id)
    return [{"input": e["input"], "output": e["output"]} for e in random.sample(ex, k)]


if __name__ == "__main__":
    programs = load_programs()
    ids = list_task_ids()
    common = [t for t in ids if t in programs]
    print("programs parsed   :", len(programs))
    print("tasks with re-arc :", len(ids))
    print("usable (both)     :", len(common))
    t = common[0]
    print("example task", t, "-> program ops:", programs[t][:3], "...")
