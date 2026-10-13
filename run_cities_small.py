import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch
import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.run_episode import run_episode

from experiment_setup.create_env import create_eval_env
from mapf_gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

MODELS = ["2M", "6M"]
AGENT_COUNTS = [64, 128]
MAP_NAMES = ['Berlin_1_256_00', 'Boston_0_256_00', 'London_2_256_00', 'Milan_0_256_00', 'Moscow_0_256_00', 'NewYork_1_256_00', 'Paris_1_256_00', 'Paris_2_256_00']
MAX_STEPS = 256
SEED = 0

torch.set_num_threads(4)
with open("eval_configs/04-movingai/maps.yaml", encoding="utf-8") as f:
    maps = yaml.safe_load(f)
for name in MAP_NAMES:
    if name not in maps:
        raise KeyError(f"Карта не найдена: {name}")
ToolboxRegistry.register_maps(maps)
MAPFGPTInference.build()

out = Path("results") / datetime.now().strftime("cities_small_%Y%m%d_%H%M%S")
out.mkdir(parents=True, exist_ok=False)
metadata = {
    "models": MODELS, "maps": MAP_NAMES, "agent_counts": AGENT_COUNTS,
    "max_episode_steps": MAX_STEPS, "environment_seed": SEED,
    "policy_seed": 42, "device": "cpu", "torch_threads": 4,
    "torch_version": torch.__version__, "on_target": "nothing",
    "collision_system": "soft", "obs_radius": 5,
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
(out / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
fields = ["model", "num_agents", "map_name", "seed", "ISR", "CSR",
          "ep_length", "SoC", "makespan", "runtime", "wall_seconds"]
rows = []
total = len(MODELS) * len(AGENT_COUNTS) * len(MAP_NAMES)
print(f"Всего эпизодов: {total}. Результаты: {out.resolve()}", flush=True)

with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    f.flush()
    for model in MODELS:
        algo = MAPFGPTInference(MAPFGPTInferenceConfig(
            path_to_weights=f"weights/MAPF-GPT-{model}.pt", device="cpu"))
        for num_agents in AGENT_COUNTS:
            for map_name in MAP_NAMES:
                print(f"[{len(rows)+1}/{total}] {model}, {num_agents} агентов, {map_name}", flush=True)
                torch.manual_seed(42)
                env = create_eval_env(Environment(
                    with_animation=False, observation_type="MAPF",
                    on_target="nothing", map_name=map_name,
                    num_agents=num_agents, seed=SEED, obs_radius=5,
                    collision_system="soft", max_episode_steps=MAX_STEPS))
                algo.reset_states()
                started = time.perf_counter()
                try:
                    metrics = run_episode(env, algo)
                    wall_seconds = time.perf_counter() - started
                finally:
                    env.close()
                row = dict(model=model, num_agents=num_agents, map_name=map_name,
                           seed=SEED, wall_seconds=wall_seconds)
                row.update({key: metrics[key] for key in fields[4:-1]})
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(f"  CSR={row['CSR']}, SoC={row['SoC']}, шаги={row['ep_length']}, время={wall_seconds:.2f} с", flush=True)

summary_fields = ["model", "num_agents", "episodes", "success_rate",
                  "mean_ISR", "mean_SoC_success", "mean_makespan_success",
                  "mean_runtime", "mean_wall_seconds"]
with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    for model in MODELS:
        for n in AGENT_COUNTS:
            group = [r for r in rows if r["model"] == model and r["num_agents"] == n]
            successful = [r for r in group if r["CSR"] == 1]
            summary = dict(
                model=model, num_agents=n, episodes=len(group),
                success_rate=len(successful)/len(group), mean_ISR=mean(r["ISR"] for r in group),
                mean_SoC_success=mean(r["SoC"] for r in successful) if successful else "",
                mean_makespan_success=mean(r["makespan"] for r in successful) if successful else "",
                mean_runtime=mean(r["runtime"] for r in group),
                mean_wall_seconds=mean(r["wall_seconds"] for r in group))
            writer.writerow(summary)
            print(f"{model}, агентов={n}: успех {len(successful)}/{len(group)}, SoC успешных={summary['mean_SoC_success']}", flush=True)
print(f"Готово. Файлы сохранены в {out.resolve()}", flush=True)
