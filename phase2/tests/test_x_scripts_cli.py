import subprocess, sys
from pathlib import Path
import pytest
from x_helpers import ROOT, load_script

SCRIPTS = sorted(p.stem for p in (ROOT / "scripts").glob("x_0*.py"))


def test_all_nine_scripts_present():
    assert len(SCRIPTS) == 9, SCRIPTS


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_imports_and_help_exits_zero(name):
    load_script(name)                                              # resolves every import against the REAL p2/p2x APIs
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_expensive_scripts_default_to_dry_run():
    for name in ("x_04_train_agent_a", "x_07_poison_eval"):
        src = (ROOT / "scripts" / f"{name}.py").read_text(); assert '"--run"' in src and "DRY RUN" in src
    r = subprocess.run([sys.executable, str(ROOT / "scripts/x_04_train_agent_a.py"), "--out", str(ROOT / "results/v2/_dry")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "DRY RUN" in r.stderr + r.stdout and not (ROOT / "results/v2/_dry").exists()


def test_importing_modules_has_no_side_effects(tmp_path):
    code = "import sys; sys.path.insert(0, %r); import p2x, p2x.config, p2x.policies, p2x.telemetry, p2x.rollout, p2x.stats, p2x.train_long, p2x.analysis, p2x.parts, p2x.manifest, p2x.cli; print('ok')" % str(ROOT / "src")
    before = set(p for p in (ROOT / "results").rglob("*")) if (ROOT / "results").exists() else set()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path); assert r.stdout.strip() == "ok", r.stderr
    assert (set(p for p in (ROOT / "results").rglob("*")) if (ROOT / "results").exists() else set()) == before
