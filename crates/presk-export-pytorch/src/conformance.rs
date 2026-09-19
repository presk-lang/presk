//! `test_conformance.py`: compares the exported PyTorch code with the
//! reference interpreter on the same inputs. The data (`conformance/inputs.json`,
//! `conformance/expected.json`) is produced by the interpreter at export time;
//! the test loads it, builds the model, loads the parameter tree, runs the
//! forward pass and compares.

/// What the test needs to know about the entry point.
#[derive(Clone, Debug)]
pub struct ConformanceData {
    /// `MiniGPT.apply`, `mse`, `LayerNorm{8}.apply`
    pub entry: String,
    /// Python file (stem) defining the entry.
    pub module_file: String,
    /// For a module entry: class name and constructor arguments (dimension values).
    pub class: Option<(String, Vec<u64>)>,
    /// For a function entry: its name and, per module-typed parameter, the constructor call.
    pub function: Option<(String, Vec<(String, String)>)>,
    /// For a function entry: per `params M` parameter, the constructor call of `M`
    /// (the parameter tree is flattened against that instance).
    pub theta_modules: Vec<(String, String)>,
    /// For an optimizer entry (`adamw.step`): the model's constructor and the
    /// hyperparameters, compared with `torch.optim.AdamW`.
    pub optimizer: Option<OptimizerConformance>,
    /// For `train:K`: replay the interpreter's `K` steps with `train.py`'s
    /// components and compare the final parameters.
    pub train: Option<TrainConformance>,
    /// `--mode graph`: also test `model.onnx` under ONNX Runtime.
    pub graph: bool,
    /// Parameter names in call order (θ first for module entries).
    pub inputs: Vec<String>,
    pub seed: u64,
    pub bits: u32,
    pub rtol: f64,
    pub atol: f64,
}

/// `train:K`: the interpreter's `K` steps replayed in PyTorch (same θ₀, same
/// batches, `train.py`'s optimizer and schedule) and compared on θ_K.
#[derive(Clone, Debug)]
pub struct TrainConformance {
    pub k: u64,
    pub ctor: String,
    pub objective: String,
    pub rtol: f64,
    pub atol: f64,
}

fn train_test(t: &TrainConformance, theta: &str) -> String {
    format!(
        r#"    from pskd import PskdDataset, collate
    from train import BETAS, DATA, EPS, GRAD_CLIP, LR, WEIGHT_DECAY, schedule
    RTOL_TRAIN, ATOL_TRAIN = {rtol}, {atol}
    model = {ctor}.to(DTYPE)
    model.load_state_dict(flatten_params(inputs["{theta}"], model))
    theta = dict(model.named_parameters())
    dataset = PskdDataset(DATA, DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
    batches = load("batches.json")
    total = len(batches)
    assert total == {k}
    for t, idxs in enumerate(batches, start=1):
        batch = collate([dataset[i] for i in idxs])
        for group in opt.param_groups:
            group["lr"] = LR * schedule(t, total)
        opt.zero_grad()
        {objective}(theta, batch).backward()
        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    exp = flatten_params(expected, model)
    worst = 0.0
    for name, p in model.named_parameters():
        e = exp[name]
        worst = max(worst, ((p.detach() - e).abs() / (e.abs() + ATOL_TRAIN)).max().item())
    for name, p in model.named_parameters():
        assert torch.allclose(p.detach(), exp[name], rtol=RTOL_TRAIN, atol=ATOL_TRAIN), f"θ after {{total}} steps, {{name}}: worst relative gap {{worst:.3e}}"
    print(f"K-step conformance: {{total}} steps, worst relative gap {{worst:.3e}}")
    return
"#,
        rtol = t.rtol,
        atol = t.atol,
        ctor = t.ctor,
        theta = theta,
        k = t.k,
        objective = t.objective,
    )
}

/// `--mode graph`: `model.onnx` under ONNX Runtime against the interpreter (float32).
fn onnx_test(d: &ConformanceData) -> String {
    let names: Vec<String> = d.inputs.iter().skip(1).map(|n| format!("{n:?}")).collect();
    format!(
        r#"

def test_onnx():
    # `model.onnx` (from export.py, float32) against the interpreter's expected output.
    import numpy as np
    import onnxruntime as ort
    from export import export_all
    paths = export_all()
    inputs = {{k: to_value(v) for k, v in load("inputs.json").items()}}
    expected = to_value(load("expected.json"))
    args = [inputs[n] for n in [{names}]]
    session = ort.InferenceSession(paths["onnx"])
    feeds = {{i.name: (a.to(torch.float32) if a.is_floating_point() else a).numpy() for i, a in zip(session.get_inputs(), args)}}
    out = session.run(None, feeds)[0]
    exp = expected.to(torch.float32).numpy()
    assert np.allclose(out, exp, rtol=1e-4, atol=1e-5), f"onnx: max gap {{np.abs(out - exp).max()}}"
"#,
        names = names.join(", "),
    )
}

/// `adamw.step` against `torch.optim.AdamW` plus `clip_grad_norm_` and the schedule.
#[derive(Clone, Debug)]
pub struct OptimizerConformance {
    pub ctor: String,
    pub lr: String,
    pub betas: (String, String),
    pub eps: String,
    pub weight_decay: String,
    pub grad_clip: Option<String>,
    /// `constant`, `linear` or `cosine`, and the warm-up steps.
    pub schedule: (String, u64),
}

fn optimizer_test(o: &OptimizerConformance) -> String {
    let (kind, warmup) = &o.schedule;
    let sched = match kind.as_str() {
        "linear" => format!("min(1.0, t / {w}) * (1.0 - min(1.0, max(0.0, (t - {w}) / max(total - {w}, 1))))", w = *warmup as f64),
        "cosine" => format!("min(1.0, t / {w}) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, (t - {w}) / max(total - {w}, 1)))))", w = *warmup as f64),
        _ => "1.0".into(),
    };
    let sched = if *warmup == 0 && kind != "constant" { sched.replace("min(1.0, t / 0)", "1.0") } else { sched };
    let clip = match &o.grad_clip {
        Some(c) => format!("    torch.nn.utils.clip_grad_norm_(model.parameters(), {c})\n"),
        None => String::new(),
    };
    format!(
        r#"    import math
    model = {ctor}.to(DTYPE)
    model.load_state_dict(flatten_params(inputs["θ"], model))
    m = flatten_params(inputs["s"][0], model)
    v = flatten_params(inputs["s"][1], model)
    g = flatten_params(inputs["g"], model)
    t, total = float(inputs["t"]), float(inputs["total"])
    lr = {lr} * ({sched})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=({b1}, {b2}), eps={eps}, weight_decay={wd})
    for name, p in model.named_parameters():
        p.grad = g[name].clone()
        opt.state[p] = {{"step": torch.tensor(t - 1.0), "exp_avg": m[name].clone(), "exp_avg_sq": v[name].clone()}}
{clip}    opt.step()
    exp_theta = flatten_params(expected[0], model)
    exp_m = flatten_params(expected[1][0], model)
    exp_v = flatten_params(expected[1][1], model)
    for name, p in model.named_parameters():
        assert torch.allclose(p.detach(), exp_theta[name], rtol=RTOL, atol=ATOL), f"θ'.{{name}}: {{p.detach()}} vs {{exp_theta[name]}}"
        assert torch.allclose(opt.state[p]["exp_avg"], exp_m[name], rtol=RTOL, atol=ATOL), f"m'.{{name}}"
        assert torch.allclose(opt.state[p]["exp_avg_sq"], exp_v[name], rtol=RTOL, atol=ATOL), f"v'.{{name}}"
    return
"#,
        ctor = o.ctor,
        lr = o.lr,
        b1 = o.betas.0,
        b2 = o.betas.1,
        eps = o.eps,
        wd = o.weight_decay,
    )
}

pub fn test_file(d: &ConformanceData) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        r#"# Generated by `presk export`: conformance of the PyTorch export with the Presk
# reference interpreter (MPFR, {bits} bits, seed {seed}) on entry `{entry}`.
# Run with `pytest` (or `python test_conformance.py`) where PyTorch is installed.
# The gap between the exact semantics and float64 is what the tolerance measures.
import json, os
import torch
import torch.nn as nn
from {module_file} import *

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.float64
RTOL, ATOL = {rtol}, {atol}


def load(name):
    with open(os.path.join(HERE, "conformance", name)) as f:
        return json.load(f)


def to_value(j):
    """A JSON value of the interpreter to a tensor, a number or a dict."""
    if isinstance(j, dict) and "data" in j and "axes" in j:
        if "bound" in j:
            return torch.tensor(j["data"], dtype=torch.long)
        def conv(x):
            return [conv(y) for y in x] if isinstance(x, list) else float(x)
        return torch.tensor(conv(j["data"]), dtype=DTYPE)
    if isinstance(j, dict):
        return {{k: to_value(v) for k, v in j.items()}}
    if isinstance(j, list):
        return tuple(to_value(v) for v in j)
    return j


def slice_tree(v, i):
    """The i-th slice of every tensor leaf of a stacked parameter tree."""
    if isinstance(v, dict):
        return {{k: slice_tree(t, i) for k, t in v.items()}}
    return v[i] if torch.is_tensor(v) else v


def flatten_params(tree, module, prefix=""):
    """The interpreter's parameter tree as a state dict of `module`: a
    stacked sub-module (nn.ModuleList) splits the leading axis of its leaves."""
    out = {{}}
    for name, v in tree.items():
        sub = getattr(module, name, None) if module is not None else None
        key = prefix + name
        if isinstance(sub, nn.ModuleList):
            for i, m in enumerate(sub):
                out.update(flatten_params(slice_tree(v, i), m, f"{{key}}.{{i}}."))
        elif isinstance(v, dict):
            out.update(flatten_params(v, sub, key + "."))
        else:
            out[key] = v
    return out


def test_conformance():
    inputs = {{k: to_value(v) for k, v in load("inputs.json").items()}}
    expected = to_value(load("expected.json"))
"#,
        bits = d.bits,
        seed = d.seed,
        entry = d.entry,
        module_file = d.module_file,
        rtol = d.rtol,
        atol = d.atol,
    ));
    if let Some(o) = &d.optimizer {
        s.push_str(&optimizer_test(o));
    }
    if let Some(t) = &d.train {
        s.push_str(&train_test(t, d.inputs.first().map(String::as_str).unwrap_or("θ")));
    }
    match (&d.class, &d.function) {
        _ if d.optimizer.is_some() || d.train.is_some() => {}
        (Some((class, args)), _) => {
            let a: Vec<String> = args.iter().map(|x| x.to_string()).collect();
            s.push_str(&format!("    torch.manual_seed(0)\n    model = {class}({}).to(DTYPE)\n", a.join(", ")));
            s.push_str(&format!("    theta = flatten_params(inputs[\"{}\"], model)\n", d.inputs[0]));
            s.push_str("    model.load_state_dict(theta)\n");
            let rest: Vec<String> = d.inputs[1..].iter().map(|n| format!("inputs[\"{n}\"]")).collect();
            s.push_str(&format!("    with torch.no_grad():\n        out = model({})\n", rest.join(", ")));
        }
        (None, Some((fname, modules))) => {
            for (pname, ctor) in modules {
                s.push_str(&format!("    {pname} = {ctor}.to(DTYPE)\n"));
            }
            let mut call_args: Vec<String> = Vec::new();
            for n in &d.inputs {
                if modules.iter().any(|(p, _)| p == n) {
                    call_args.push(n.clone());
                } else if let Some((_, ctor)) = d.theta_modules.iter().find(|(p, _)| p == n) {
                    // A parameter tree: flatten against the instance it belongs to
                    // (a module-typed argument of the same instance if there is one).
                    let m = modules.first().map(|(p, _)| p.clone()).unwrap_or_else(|| ctor.clone());
                    s.push_str(&format!("    {n} = flatten_params(inputs[\"{n}\"], {m})\n"));
                    call_args.push(n.clone());
                } else {
                    call_args.push(format!("inputs[\"{n}\"]"));
                }
            }
            s.push_str(&format!("    out = {fname}({})\n", call_args.join(", ")));
        }
        (None, None) => s.push_str("    raise RuntimeError(\"no entry\")\n"),
    }
    s.push_str(
        r#"    if isinstance(expected, dict):
        got = flatten_params(out if isinstance(out, dict) else dict(out), None) if isinstance(out, dict) else out
        exp = flatten_params(expected, None)
        for k, e in exp.items():
            assert torch.allclose(got[k].to(DTYPE), e, rtol=RTOL, atol=ATOL), f"{k}: {got[k]} vs {e}"
    else:
        assert torch.allclose(out.to(DTYPE) if torch.is_tensor(out) else torch.tensor(out, dtype=DTYPE), expected, rtol=RTOL, atol=ATOL), f"{out} vs {expected}"


if __name__ == "__main__":
    test_conformance()
    print("conformance: ok")
"#,
    );
    if d.graph && d.class.is_some() {
        s.push_str(&onnx_test(d));
    }
    s
}
