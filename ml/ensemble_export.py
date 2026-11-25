#!/usr/bin/env python3
"""ensemble_export: K trained students -> one ONNX file with the same inputs (front, bev, state, ids)
and the same single 2-dim output, whose action is the mean of the members' de-normalised actions
(optionally the median of their steering, so two members that pick opposite sides of an obstacle do
not average into the obstacle). A drop-in for policy_bridge: nothing on the car changes except the
file and K times the (small) inference cost.

    python3 ml/ensemble_export.py --out runs/ens_k4/student.onnx runs/a_s0 runs/a_s1 runs/a_s2 runs/a_s3 \
        [--weights best|last] [--median-steer]

Each member is rebuilt from <run>/<weights>.pt and <run>/action_norm.npy exactly as train_policy.export
does (masks baked in, EmbeddingBag replaced), and the file is checked with onnxruntime against the
members' own student.onnx files on random inputs.
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
from train_student import Student                    # noqa: E402
import policy_io as PIO                               # noqa: E402


class MeanBag(nn.Module):
    """EmbeddingBag(mode='mean', padding_idx=0) with ops the legacy ONNX exporter accepts."""
    def __init__(s, bag): super().__init__(); s.w = bag.weight
    def forward(s, ids):
        m = (ids != 0).float().unsqueeze(-1)
        return (s.w[ids] * m).sum(1) / m.sum(1).clamp(min=1.0)


class Member(nn.Module):
    def __init__(self, run, weights):
        super().__init__()
        sd = torch.load(os.path.join(run, weights + '.pt'), map_location='cpu')
        m = Student(0)
        for k in ('state_mask', 'cam_on', 'lidar_on'):
            if k not in sd: raise SystemExit(f'{run}/{weights}.pt carries no {k}: not a train_policy.py model')
            m.register_buffer(k, sd[k].clone())
        m.load_state_dict(sd); m.eval()
        ids = torch.tensor([PIO.text_ids('turn left, then go straight to the end and stop')])
        ref = m.txt(ids); m.txt = MeanBag(m.txt.emb)
        assert torch.allclose(ref, m.txt(ids), atol=1e-6), 'MeanBag != EmbeddingBag'
        norm = np.load(os.path.join(run, 'action_norm.npy'))
        self.register_buffer('mu', torch.tensor(norm[0], dtype=torch.float32))
        self.register_buffer('sd', torch.tensor(norm[1], dtype=torch.float32))
        self.m = m

    def forward(self, front, bev, state, ids):
        m = self.m
        f = m.front(front) * m.cam_on; b = m.bev(bev) * m.lidar_on
        t = m.txt(ids); st = m.state(state * m.state_mask)
        return m.head(torch.cat([f, b, t, st], 1)) * self.sd + self.mu


class Ensemble(nn.Module):
    def __init__(self, members, median_steer):
        super().__init__(); self.members = nn.ModuleList(members); self.median_steer = median_steer

    def forward(self, front, bev, state, ids):
        acts = torch.stack([mm(front, bev, state, ids) for mm in self.members], 0)     # [K, B, 2]
        out = acts.mean(0)
        if self.median_steer:
            out = torch.stack([out[:, 0], acts[:, :, 1].median(0).values], 1)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='run directories (each with <weights>.pt, action_norm.npy, student.onnx)')
    ap.add_argument('--out', required=True); ap.add_argument('--weights', default='best', choices=['best', 'last'])
    ap.add_argument('--median-steer', action='store_true')
    a = ap.parse_args()
    import onnx, onnxruntime as ort
    ens = Ensemble([Member(r, a.weights) for r in a.runs], a.median_steer).eval()
    dummy = (torch.zeros(1, 3, *PIO.FRONT_HW), torch.zeros(1, 1, *PIO.BEV_HW), torch.zeros(1, 5),
             torch.zeros(1, PIO.MAX_TOK, dtype=torch.long))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = f'{a.out}.tmp{os.getpid()}'
    torch.onnx.export(ens, dummy, tmp, input_names=['front', 'bev', 'state', 'ids'], output_names=['action'],
                      opset_version=17, dynamo=False,
                      dynamic_axes={k: {0: 'batch'} for k in ('front', 'bev', 'state', 'ids', 'action')})
    m = onnx.load(tmp)
    # the members' config (sensor geometry, flags) travels along: policy_bridge reads sensor_geom from it
    m0 = onnx.load(os.path.join(a.runs[0], 'student.onnx'))
    cfg = json.loads({p.key: p.value for p in m0.metadata_props}.get('config', '{}'))
    cfg.update(ensemble={'members': [os.path.abspath(r) for r in a.runs], 'weights': a.weights,
                         'median_steer': a.median_steer})
    for k, v in (('action_order', ','.join(PIO.ACTION_ORDER)), ('trainer', 'ml/ensemble_export.py'),
                 ('config', json.dumps(cfg))):
        e = m.metadata_props.add(); e.key, e.value = k, v
    onnx.save(m, tmp)
    # check: onnxruntime on the file == torch, and == the mean of the members' own files (when
    # those were exported from the same weights)
    rng = np.random.default_rng(0)
    feed = {'front': rng.random((8, 3, *PIO.FRONT_HW), dtype=np.float32),
            'bev': (rng.random((8, 1, *PIO.BEV_HW)) < 0.1).astype(np.float32),
            'state': rng.normal(0, 1, (8, 5)).astype(np.float32),
            'ids': rng.integers(0, 50, (8, PIO.MAX_TOK)).astype(np.int64)}
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    out = ort.InferenceSession(tmp, so, providers=['CPUExecutionProvider']).run(['action'], feed)[0]
    with torch.no_grad(): ref = ens(*[torch.from_numpy(v) for v in feed.values()]).numpy()
    d_torch = float(np.abs(out - ref).max())
    singles = np.stack([ort.InferenceSession(os.path.join(r, 'student.onnx'), so, providers=['CPUExecutionProvider'])
                        .run(['action'], feed)[0] for r in a.runs])
    d_members = float(np.abs(singles.mean(0)[:, 0] - out[:, 0]).max())    # speed is always the mean
    if d_torch > 1e-4: raise SystemExit(f'onnxruntime differs from torch by {d_torch}')
    fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
    os.replace(tmp, a.out)
    print(f'{a.out}: {len(a.runs)} members ({a.weights}.pt, {"median" if a.median_steer else "mean"} steering); '
          f'onnxruntime vs torch {d_torch:.1e}, vs the members\' own files (speed) {d_members:.1e}', flush=True)


if __name__ == '__main__':
    main()
