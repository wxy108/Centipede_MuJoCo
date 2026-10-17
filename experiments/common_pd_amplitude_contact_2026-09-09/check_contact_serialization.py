"""Exercise the real persistent-channel result writer, with zero physics steps."""
from pathlib import Path
import argparse, hashlib, inspect, json, tempfile
import contact_erase as ce

c = ce.c


def exercise(method):
    with tempfile.TemporaryDirectory(prefix='contact_record_test_', dir=c.ROOT/'outputs') as directory:
        channel = ce.Channel.__new__(ce.Channel)
        channel.task = dict(erasure=dict(probability=0., seed=0,
            kind='persistent_ground_only_bilateral_pair'), duration=6.)
        channel.applied = True
        channel.times = [1.5, 1.501]
        channel.center = [[0., 0., 0.], [-.001, 0., 0.]]
        channel.uniforms = c.np.arange(19, dtype=float) / 19
        channel.selected = c.np.zeros(19, dtype=bool)
        channel.changed = []
        channel.folder = Path(directory)
        channel.upright = [1., 1.]
        channel.pose_t = [1.5]
        channel.qpos = [c.np.zeros(7)]
        result = dict(metrics=dict(displacement_mm=1.))
        method(channel, result)
        saved = json.loads((channel.folder/'schedule.json').read_text())
        assert saved == result['channel']
        assert saved['kind'] == channel.task['erasure']['kind']
        assert saved['raw_D_recomputed'] == 1. and saved['erased_pairs'] == []
        with c.np.load(channel.folder/'contact_trajectory.npz') as z:
            c.np.testing.assert_array_equal(z['center'], channel.center)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect-original-failure', action='store_true')
    args = parser.parse_args()
    archive = c.ROOT/'outputs/amendments/p0_metadata_20260909_0923'
    original = archive/'contact_erase.py'
    import ast
    source = original.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Channel')
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'finish')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = dict(c=c, np=c.np, hashlib=hashlib)
    exec(compile(ast.fix_missing_locations(module), str(original), 'exec'), namespace)
    try:
        exercise(namespace['finish'])
    except TypeError as error:
        assert "multiple values for keyword argument 'kind'" in str(error)
        original_error = str(error)
    else:
        raise AssertionError('Archived bug was not reproduced')
    if args.expect_original_failure:
        print('Original finish method reproduced:', original_error)
    else:
        exercise(ce.Channel.finish)
        c.write(archive/'serialization_regression.json', dict(passed=True,
            original_error=original_error, old_code_sha256=c.sha(original),
            corrected_code_sha256=c.sha(ce.__file__), test_sha256=c.sha(__file__),
            physics_steps_executed=0, final_writer_roundtrip=True))
        print('Archived error reproduced; corrected result writer roundtrip passed; zero physics steps.')
