import ast, copy, json, re
from pathlib import Path
import pytest
from p2x import config as C
from x_helpers import ROOT, PROJECT_ROOT

CFG = C.load_config()
R = {k: tuple(v) for k, v in CFG["seeds"]["ranges"].items()}


def consts(path):
    out = {}
    for n in ast.parse((PROJECT_ROOT / path).read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try: out[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception: pass
    return out


def test_ranges_pairwise_disjoint_materialized():
    names = list(R)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert R[a][1] <= R[b][0] or R[b][1] <= R[a][0], (a, b)


def test_legacy_ranges_match_what_the_legacy_code_actually_uses():
    s = consts("src/p2/sim.py"); V, T = s["VAL_SEED_BASE"], s["TEST_SEED_BASE"]
    assert R["train_online"] == tuple(s["TRAIN_SEED_RANGE"])
    cal = (PROJECT_ROOT / "scripts/calibrate_load.py").read_text(); assert "VAL_SEED_BASE + i" in cal and "range(100)" in cal
    assert R["val_calibration_v1"] == (V, V + 100)
    tr = (PROJECT_ROOT / "scripts/train_ppo.py").read_text(); assert "VAL_SEED_BASE + 1000 + i" in tr and 'default=25' in tr
    assert R["val_selection_v1"] == (V + 1000, V + 1000 + 25)
    ev = (PROJECT_ROOT / "scripts/eval_clean.py").read_text(); assert "TEST_SEED_BASE + i" in ev and "default=200" in ev
    assert R["test_firstlook_v1"] == (T, T + 200)


def test_new_sets_disjoint_from_every_legacy_seed_actually_used():
    s = consts("src/p2/sim.py")
    legacy = set(range(*s["TRAIN_SEED_RANGE"])) | set(range(*R["val_calibration_v1"])) | set(range(*R["val_selection_v1"])) | set(range(*R["test_firstlook_v1"]))
    new = {"val_v2": set(range(*R["val_v2"])), "confirmatory": set(range(*R["test_confirmatory_v2"])), "poison": set(range(*R["test_poison_v2"]))}
    for k, v in new.items(): assert not (v & legacy), k
    assert not (new["val_v2"] & new["confirmatory"]) and not (new["val_v2"] & new["poison"]) and not (new["confirmatory"] & new["poison"])


def test_scenarios_actually_generated_are_disjoint(monkeypatch):
    s = {}
    for role, rng, n in (("development", "val_v2", CFG["diagnostics"]["n_per_load"]), ("confirmatory", "test_confirmatory_v2", CFG["confirmatory"]["n_per_load"]),
                         ("poison", "test_poison_v2", CFG["poisoning"]["presets"]["full"]["n_per_load"])):
        s[role] = {sc.seed for _, _, sc in C.scenarios(CFG, rng, [1.0], n, role)}
    assert not (s["development"] & s["confirmatory"]) and not (s["development"] & s["poison"]) and not (s["confirmatory"] & s["poison"])
    assert all(x >= 11_000_000 for v in s.values() for x in v)


@pytest.mark.parametrize("role,rng", [("development", "test_confirmatory_v2"), ("confirmatory", "val_v2"), ("poison", "test_firstlook_v1"), ("development", "val_selection_v1")])
def test_role_guard_blocks_cross_use(role, rng):
    with pytest.raises(C.ConfigError): C.scenarios(CFG, rng, [1.0], 2, role)


def test_overlapping_config_is_rejected():
    bad = copy.deepcopy(CFG); bad["seeds"]["ranges"]["val_v2"] = [20_000_100, 20_000_150]
    with pytest.raises(C.ConfigError): C.validate(bad)
    bad = copy.deepcopy(CFG); bad["seeds"]["allowed_roles"]["development"] = ["test_firstlook_v1"]
    with pytest.raises(C.ConfigError): C.validate(bad)


def test_unsafe_paths_and_stopping_rule_rejected():
    bad = copy.deepcopy(CFG); bad["results_root"] = "results/models/new"
    with pytest.raises(C.ConfigError): C.validate(bad)
    with pytest.raises(C.ConfigError): C.assert_not_protected(CFG, PROJECT_ROOT / "results/models/replace/seed42")
    bad = copy.deepcopy(CFG); bad["train_agent_a"]["eval_every"] = 25000
    with pytest.raises(C.ConfigError): C.validate(bad)


def test_ppo_hp_identical_to_legacy_train_script():
    src = (PROJECT_ROOT / "scripts/train_ppo.py").read_text(); node = [n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "HP"][0]
    hp = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
    assert hp == {k: v for k, v in CFG["ppo_hp"].items() if not k.startswith("_")}


def test_topology_guard_rejects_wrong_hash_and_missing(tmp_path):
    bad = copy.deepcopy(CFG); bad["topology"]["sha256"] = "0" * 64
    with pytest.raises(C.ConfigError): C.require_real_topology(bad)
    bad = copy.deepcopy(CFG); bad["topology"]["ned"] = "does/not/exist.ned"
    with pytest.raises(C.ConfigError): C.require_real_topology(bad)


def test_topology_guard_accepts_the_recorded_file_when_present():
    ned = PROJECT_ROOT / CFG["topology"]["ned"]
    if not ned.is_file() or C.sha256_file(ned) != CFG["topology"]["sha256"]:
        pytest.skip("recorded NSFNET-bw NED not present here (stand-in or missing); guard correctly refuses it")
    assert C.require_real_topology(CFG) == CFG["topology"]["sha256"]
