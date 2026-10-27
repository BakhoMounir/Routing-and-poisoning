"""End-to-end script logic with STUB models on whatever topology is on disk. In a sandbox without the real NED this is a logic test only."""
import copy, json, argparse
from pathlib import Path
import numpy as np
import pytest
from p2.sim import Network
from p2x.config import load_config
from x_helpers import StubModel, load_script, stub_hash

CFG = load_config()


@pytest.fixture(scope="module")
def net(): return Network()


def ns(**kw):
    d = dict(out=None, overwrite=True, config=None, verbose=False, smoke=False, heuristic=None, n_per_load=None, boot=200, preset="smoke", run=True, baselines_only=False, agent=None, models_dir=None, seeds=None, ckpts=["best"])
    d.update(kw); return argparse.Namespace(**d)


def stub_cfg(tmp_path):
    c = copy.deepcopy(CFG)
    for ag in c["confirmatory"]["agents"]:
        d = tmp_path / ag["id"]
        for s in (42, 43):
            (d / f"seed{s}").mkdir(parents=True, exist_ok=True); (d / f"seed{s}" / "best.zip").write_bytes(b"stub")
        ag["models_dir"] = str(d)
    return c


def stub_loader(models_dir, seed, ckpt):
    return StubModel("replace" if "A_" in str(models_dir) else "enhance", seed)


def test_x03_runs_and_split_half_selection_is_disjoint(net, tmp_path):
    m = load_script("x_03_diag_baselines"); r = m.execute(CFG, ns(n_per_load=8, out=tmp_path / "o"), net, "TEST-ONLY")
    d = json.loads((tmp_path / "o/decisions.json").read_text()); sel = json.loads((tmp_path / "o/heuristic_selection.json").read_text())
    assert "EXPLORATORY" in d["status"] and "HELDOUT" in "".join(d) and sel["selection_rule"].startswith("max mean") and (tmp_path / "o/manifest.json").is_file()
    assert json.loads((tmp_path / "o/manifest.json").read_text())["status"] == "completed"
    with pytest.raises(SystemExit): m.execute(CFG, ns(n_per_load=8, out=tmp_path / "o", overwrite=False), net, "x")        # refuses to overwrite a completed run


def test_x02_diag_with_stub_models(net, tmp_path):
    m = load_script("x_02_diag_checkpoints")
    for agent in ("enhance", "replace"):
        s, rows = m.execute(CFG, ns(agent=agent, models_dir=tmp_path, seeds=[42], ckpts=["best"], n_per_load=3, out=tmp_path / agent), net, "TEST-ONLY", loader=lambda d, sd, ck, a=agent: StubModel(a, sd))
        assert len(rows) == 1 and s["status"] == "EXPLORATORY"
    assert "frac_route_equals_unit_dijkstra" in rows[0] or "policy_entropy_mean" in rows[0]


def test_x06_smoke_then_x08_analysis(net, tmp_path):
    m = load_script("x_06_confirmatory_eval"); c = stub_cfg(tmp_path)
    out = m.execute(c, ns(smoke=True, out=tmp_path / "conf"), net, "TEST-ONLY", loader=stub_loader)
    import pandas as pd
    df = pd.read_csv(out / "episodes.csv"); assert df.replay_ok.all() and {"minhop", "util_weight_dijkstra", "A_legacy_150k", "B_legacy_150k"} <= set(df.method) and "A_600k" in set(df.method)
    assert set(df.scenario_seed) <= set(range(11_000_000, 11_000_050))                  # smoke used DEVELOPMENT seeds only
    a8 = load_script("x_08_analyze"); import sys
    sys.argv = ["x_08", "--stage", "confirmatory", "--run-dir", str(out), "--overwrite"]; a8.main(); assert (out / "analysis/primary_comparisons_pp.csv").is_file()


def test_x06_real_run_requires_freeze_and_range_guard(net, tmp_path):
    m = load_script("x_06_confirmatory_eval"); c = stub_cfg(tmp_path); c["results_root"] = str(tmp_path / "res")
    with pytest.raises(SystemExit): m.execute(c, ns(smoke=False, out=tmp_path / "x"), net, "TEST-ONLY", loader=stub_loader)       # no freeze -> refuses; no LOCK created
    assert not list((tmp_path / "res").rglob("LOCK.json"))


def _frozen(tmp_path):
    f5 = load_script("x_05_freeze"); c = stub_cfg(tmp_path); c["results_root"] = str(tmp_path / "res"); c["confirmatory"]["n_per_load"] = 4
    for ag in c["confirmatory"]["agents"]:
        ag["seeds"] = [42, 43]
        for s in (42, 43): (Path(ag["models_dir"]) / f"seed{s}" / "meta.json").write_text(json.dumps({"best_val": 0.9, "versions": {"torch": "t"}}))
    h = tmp_path / "h.json"; h.write_text(json.dumps({"selected": dict(p=1.0, floor=0.1, infeasible_multiplier=1.0)}))
    from p2x.config import atomic_write_text, freeze_path
    fr = f5.build_freeze(c, h, "TEST-ONLY"); atomic_write_text(freeze_path(c), json.dumps(fr)); return c, fr


def test_x05_freeze_records_hashes_and_handles_absent_optional(tmp_path):
    from p2x.config import sha256_file
    c, fr = _frozen(tmp_path)
    assert set(fr["models"]) == {"A_legacy_150k", "A_600k", "B_legacy_150k"} and all(fr["models"][k]["42"]["sha256"] == sha256_file(Path(fr["models"][k]["42"]["path"])) for k in fr["models"])
    f5 = load_script("x_05_freeze"); (Path(c["confirmatory"]["agents"][1]["models_dir"]) / "seed43" / "best.zip").unlink()
    assert f5.build_freeze(c, tmp_path / "h.json", "x")["models"]["A_600k"] == {}                       # optional + incomplete -> treated as absent
    (Path(c["confirmatory"]["agents"][0]["models_dir"]) / "seed43" / "best.zip").unlink()
    with pytest.raises(SystemExit): f5.build_freeze(c, tmp_path / "h.json", "x")                         # required + incomplete -> refuse


def test_x06_confirmatory_lock_resume_and_tamper_detection(net, tmp_path):
    from p2x.config import lock_path, results_dir
    m = load_script("x_06_confirmatory_eval"); c, fr = _frozen(tmp_path)
    c_net_sha = "TEST-ONLY"
    out = m.execute(c, ns(smoke=False), net, c_net_sha, loader=stub_loader)
    import pandas as pd; df = pd.read_csv(out / "episodes.csv")
    assert lock_path(c, c["confirmatory"]["run_id"]).is_file() and set(df.scenario_seed) <= set(range(30_000_000, 30_000_004)) and df.replay_ok.all()
    n1 = len(df); (out / "parts" / "minhop__-1.csv").unlink(); m.execute(c, ns(smoke=False), net, c_net_sha, loader=stub_loader)       # RESUME regenerates only the missing part
    assert len(pd.read_csv(out / "episodes.csv")) == n1
    Path(c["confirmatory"]["agents"][2]["models_dir"], "seed42", "best.zip").write_bytes(b"tampered")                              # checkpoint changed after freeze
    with pytest.raises(SystemExit): m.execute(c, ns(smoke=False), net, c_net_sha, loader=stub_loader)


def test_x07_dry_run_writes_nothing_then_smoke_run_checks_invariants(net, tmp_path):
    m = load_script("x_07_poison_eval"); c = stub_cfg(tmp_path)
    assert m.execute(c, ns(run=False, out=tmp_path / "dry"), net, "TEST-ONLY") is None and not (tmp_path / "dry").exists()
    out = m.execute(c, ns(run=True, out=tmp_path / "p"), net, "TEST-ONLY", loader=stub_loader, hash_fn=stub_hash)
    mf = json.loads((out / "manifest.json").read_text()); inv = mf["extra"]["invariants"]
    assert mf["status"] == "completed" and inv["all_replay_ok"] and inv["all_caps_ok"] and inv["max_delta_over_budget"] <= 1e-9
    import pandas as pd; df = pd.read_csv(out / "episodes.csv")
    assert {"clean", "random_sign@0.2", "conceal_topk@0.2", "fake_topk@0.2"} <= set(df.attack) and df[df.model_seed >= 0].param_unchanged.all()
    cl = df[df.attack == "clean"]; assert (cl.tele_max_abs_delta == 0).all()
    mh = df[df.method == "minhop"].groupby("attack").accepted_offered_ratio.mean(); assert mh.nunique() == 1          # obs-independent control is unaffected by every attack


def test_x07_nonsmoke_is_gated(net, tmp_path):
    m = load_script("x_07_poison_eval"); c = stub_cfg(tmp_path); c["results_root"] = str(tmp_path / "res")
    with pytest.raises(SystemExit): m.execute(c, ns(preset="pilot", run=True, out=tmp_path / "p"), net, "TEST-ONLY", loader=stub_loader, hash_fn=stub_hash)
