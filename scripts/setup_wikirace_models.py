"""Install the pinned base checkpoint and calibrate both local app checkpoints."""
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from wikirace.train import REVISION

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/wikirace/base"
TUNED = ROOT / "outputs/wikirace/tuned"


def calibrated_config(folder, temperature):
    path = folder / "rl_agent_config.json"
    cfg = json.loads(path.read_text())
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg["temperature_by_options"] = {f"choice:{bucket}": temperature
                                      for bucket in ("2", "3-5", "6-10", "11+")}
    path.write_text(json.dumps(cfg, indent=2))


def main():
    report = json.loads((TUNED / "report.json").read_text())
    if report.get("status") != "completed" or report.get("base_revision") != REVISION:
        raise ValueError("Kaggle run is incomplete or uses another base revision")
    source = Path(snapshot_download("convaiinnovations/laya", revision=REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"]))
    BASE.mkdir(parents=True, exist_ok=True)
    for relative in ("model.safetensors", "rl_agent_config.json", "encoder/config.json"):
        destination = BASE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    shutil.copytree(source / "tokenizer", BASE / "tokenizer", dirs_exist_ok=True)
    calibrated_config(BASE, report["base"]["decisions"]["temperature"])
    calibrated_config(TUNED, report["tuned"]["decisions"]["temperature"])
    print("Base and tuned checkpoints ready for app/server.py")


if __name__ == "__main__":
    main()
