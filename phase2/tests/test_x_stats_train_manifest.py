import json
import numpy as np
import pytest
from p2x import analysis
from p2x.manifest import RunManifest, code_tree_sha256
from p2x.parts import merge_parts, part_name, write_part
from p2x.stats import holm, summarize_diff
from p2x.train_long import plateau_stop, select_best, smoothed
from p2x.config import load_config

CFG = load_config()


def test_holm_known_values_and_monotone():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]); assert np.allclose(holm([0.5]), [0.5])
    p = np.random.default_rng(0).uniform(size=20); a = holm(p); assert np.all(a >= p) and np.all(a <= 1)


def test_summarize_diff_recovers_effect_and_separates_variance_sources():
    rng = np.random.default_rng(5); D = 0.01 + rng.normal(0, 0.03, (1, 400))                       # scenario noise only
    s = summarize_diff(D, B=3000, rng=np.random.default_rng(1)); assert s["ci_scen_lo"] < 1.0 < s["ci_scen_hi"] and np.isnan(s["ci_model_t_lo"])
    D2 = 0.01 + rng.normal(0, 0.002, (10, 300)) + rng.normal(0, 0.01, (10, 1))                       # large between-model spread
    s2 = summarize_diff(D2, B=3000, rng=np.random.default_rng(2))
    assert s2["sd_across_models_pp"] > 0.5 and (s2["ci_cross_hi"] - s2["ci_cross_lo"]) > 3 * (s2["ci_scen_hi"] - s2["ci_scen_lo"])   # crossed CI reflects training variance, scenario-only does not


def test_null_effect_gives_large_p_and_bonferroni_is_wider():
    D = np.random.default_rng(3).normal(0, 0.02, (5, 200)); D -= D.mean()
    s = summarize_diff(D, B=2000, n_tests=16, rng=np.random.default_rng(4)); assert s["p_boot"] > 0.5
    assert s["ci_cross_bonf_hi"] - s["ci_cross_bonf_lo"] > s["ci_cross_hi"] - s["ci_cross_lo"]


def test_summarize_rejects_bad_input():
    with pytest.raises(ValueError): summarize_diff(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError): summarize_diff(np.zeros(5))


def test_plateau_rule_and_best_selection():
    s = CFG["train_agent_a"]["stopping"]; kw = dict(smooth=s["smooth_evals"], window=s["window_evals"], min_evals=s["min_evals"], min_improvement=s["min_improvement"])
    assert plateau_stop([0.8 + 0.01 * i for i in range(20)], **kw)[0] is False                    # still improving
    assert plateau_stop([0.8 + 0.01 * i for i in range(8)] + [0.88] * 12, **kw)[0] is True        # flat
    assert plateau_stop([0.5] * 5, **kw)[0] is False                                              # too few evals: never stops before min_evals
    assert select_best([0.1, 0.3, 0.3, 0.2]) == 1 and np.allclose(smoothed([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])


def test_manifest_records_status_and_failure(tmp_path):
    p = tmp_path / "m.json"
    with RunManifest(p, "t", CFG, {"a": 1}, seeds={"x": 1}) as m: m.set("k", 2)
    d = json.loads(p.read_text()); assert d["status"] == "completed" and d["config_sha256"] == CFG["_sha256"] and d["extra"]["k"] == 2 and "python" in d["environment"] and d["code_tree_sha256"]
    with pytest.raises(RuntimeError):
        with RunManifest(p, "t", CFG): raise RuntimeError("boom")
    d = json.loads(p.read_text()); assert d["status"] == "failed" and "boom" in d["error"]


def test_code_hash_changes_with_source(tmp_path):
    (tmp_path / "src/p2").mkdir(parents=True); f = tmp_path / "src/p2/a.py"; f.write_text("x=1"); h1 = code_tree_sha256(tmp_path, ("src/p2",)); f.write_text("x=2")
    assert h1 != code_tree_sha256(tmp_path, ("src/p2",))


def test_parts_roundtrip_union_columns(tmp_path):
    write_part(tmp_path / part_name("A@1", 3), [dict(a=1, b=2)]); write_part(tmp_path / part_name("B", -1), [dict(a=3, c=4)])
    n = merge_parts(tmp_path, tmp_path.parent / "m.csv"); import csv; R = list(csv.DictReader(open(tmp_path.parent / "m.csv")))
    assert n == 2 and set(R[0]) == {"a", "b", "c"}


def _grid(effects, n=60, seeds=4, loads=(1.0, 2.0)):
    import pandas as pd
    rng = np.random.default_rng(0); rows = []
    base = {L: rng.uniform(0.7, 0.9, n) for L in loads}
    for m, (eff, is_model) in effects.items():
        for s in (range(seeds) if is_model else [-1]):
            for L in loads:
                v = base[L] + eff + (rng.normal(0, 0.002, n) if is_model else 0)
                for i in range(n): rows.append(dict(method=m, model_seed=s, scenario_idx=i, load=L, accepted_offered_ratio=v[i], attack="clean"))
    return pd.DataFrame(rows)


def test_compare_detects_effect_holm_and_marks_absent():
    df = _grid({"ref": (0.0, False), "A": (0.02, True)}); rng = np.random.default_rng(1)
    t = analysis.compare(df, [("P1", "A", "ref"), ("P2", "Z_absent", "ref")], [1.0, 2.0], 0.05, 1000, rng)
    c = t[t.computed]; assert len(c) == 2 and (c.mean_pp.between(1.5, 2.5)).all() and c.reject_holm_alpha.all(); assert (~t[t.test == "Z_absent"].computed).all()
    with pytest.raises(ValueError): analysis.paired_D(np.zeros((1, 3)), np.zeros((1, 4)))


def test_poison_tables_paired_delta_and_contrast():
    import pandas as pd
    rng = np.random.default_rng(2); n = 40; base = rng.uniform(0.7, 0.9, n); rows = []
    for s in range(3):
        for att, d in (("clean", 0.0), ("conceal_topk@0.2", -0.05), ("random_sign@0.2", -0.01)):
            for i in range(n): rows.append(dict(method="A_600k", model_seed=s, scenario_idx=i, load=2.0, accepted_offered_ratio=base[i] + d, attack=att))
    t = analysis.poison_tables(pd.DataFrame(rows), ["A_600k"], ["conceal_topk", "random_sign"], [0.2], [2.0], dict(targets=["A_600k"], attacks=["conceal_topk"], eps=0.2, load=2.0), 0.05, 1000, np.random.default_rng(3))
    r = t[t.attack == "conceal_topk"].iloc[0]; assert abs(r.mean_pp + 5.0) < 1e-6 and abs(r.contrast_mean_pp + 4.0) < 1e-6 and r.primary


def test_training_resume_state_validation():
    from p2x.train_long import validate_resume_state

    state = {
        "seed": 42,
        "resumes": 0,
        "init_val": 0.80,
        "history": [
            {
                "step": 24000,
                "val": 0.85,
                "per_load": {"1.0": 0.86, "2.0": 0.84},
                "train_last100": 0.83,
                "wall_s": 12.0,
            }
        ],
        "next_eval": 48000,
        "best_val": 0.85,
        "best_step": 24000,
        "wall_s": 12.0,
        "ep_rets": [0.8, 0.9],
        "stop_reason": None,
    }

    assert validate_resume_state(state, seed=42, checkpoint_step=24000)

    with pytest.raises(ValueError, match="seed"):
        validate_resume_state(state, seed=43, checkpoint_step=24000)

    with pytest.raises(ValueError, match="step"):
        validate_resume_state(state, seed=42, checkpoint_step=48000)

    with pytest.raises(ValueError, match="state"):
        validate_resume_state({}, seed=42, checkpoint_step=24000)

    bad = {**state, "wall_s": -1}
    with pytest.raises(ValueError):
        validate_resume_state(bad, seed=42, checkpoint_step=24000)

    bad = {**state, "best_step": 12000}
    with pytest.raises(ValueError):
        validate_resume_state(bad, seed=42, checkpoint_step=24000)

    bad = {**state, "ep_rets": "not-a-list"}
    with pytest.raises(ValueError):
        validate_resume_state(bad, seed=42, checkpoint_step=24000)

def test_training_resume_checkpoint_pair_guard(tmp_path):
    from p2x.train_long import check_resume_checkpoint_pair

    state = tmp_path / "state.json"
    checkpoint = tmp_path / "last.zip"

    assert check_resume_checkpoint_pair(state, checkpoint) == "fresh"
    state.write_text("{}")
    with pytest.raises(RuntimeError, match="incomplete"):
        check_resume_checkpoint_pair(state, checkpoint)
    state.unlink()
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="incomplete"):
        check_resume_checkpoint_pair(state, checkpoint)
    state.write_text("{}")
    assert check_resume_checkpoint_pair(state, checkpoint) == "resume"
