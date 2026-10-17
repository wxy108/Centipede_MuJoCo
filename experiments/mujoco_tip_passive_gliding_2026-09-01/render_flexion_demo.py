"""Exact recorded-pose MuJoCo demos, with no physical or temporal smoothing."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('flexion_run', ROOT / 'run_tip_flexion_search.py')
fx = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fx
spec.loader.exec_module(fx)
np, mujoco = fx.np, fx.mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 640, 360
FONT = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 19)
FONT_BOLD = ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf', 21)


def line(scene, start, end, rgba, width=2.0):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                       np.eye(3).reshape(9), np.asarray(rgba, dtype=float))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
    scene.ngeom += 1


def project(scene, point):
    a, b = scene.camera[0], scene.camera[1]
    pos = (a.pos + b.pos) * .5
    forward = (a.forward + b.forward) * .5
    forward /= np.linalg.norm(forward)
    up = (a.up + b.up) * .5
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    v = point - pos
    near = .5 * (a.frustum_near + b.frustum_near)
    bottom = .5 * (a.frustum_bottom + b.frustum_bottom)
    top = .5 * (a.frustum_top + b.frustum_top)
    center = .5 * (a.frustum_center + b.frustum_center)
    xx = np.dot(v, right) * near / np.dot(v, forward)
    yy = np.dot(v, up) * near / np.dot(v, forward)
    return np.asarray((320 + 360 * (xx - center) / (top - bottom), 360 - 360 * (yy - bottom) / (top - bottom)))


def light_and_color(model, ground):
    original_groups = model.geom_group.copy()
    collision = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    model.geom_group[collision] = 5
    if model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_PLANE:
        model.geom_group[ground] = 0
        model.geom_matid[ground] = -1
        model.geom_rgba[ground] = (.80, .83, .86, 1)
    else:
        # Hfield stays invisible: do not overlay it on the smooth visual mesh.
        model.geom_rgba[ground, 3] = 0
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if original_groups[g] == 2 and ('locked_visual' in name or name.startswith('link_') and name.endswith('_visual') and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH):
            model.geom_rgba[g] = (.64, .25, .11, 1)
        if name == 'terrain_visual_geom':
            model.geom_rgba[g] = (.68, .72, .75, 1)
        if '_tip_visual' in name:
            model.geom_rgba[g] = (1., .24, .025, 1)
    model.light_diffuse[:] *= .35
    model.light_specular[:] = 0
    model.light_castshadow[:] = 0
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .42
    model.vis.headlight.diffuse[:] = .60
    model.vis.headlight.specular[:] = 0


class Replay:
    def __init__(self, record_path):
        self.path = Path(record_path)
        self.record = json.loads(self.path.read_text(encoding='utf-8'))
        model_path = Path(self.record['model_path'])
        assert fx.sha256(model_path) == self.record['model_sha256']
        original = mujoco.MjModel.from_xml_path(str(model_path))
        tree = ET.parse(model_path)
        ET.SubElement(tree.getroot().find('asset'), 'texture', type='skybox', builtin='gradient',
                      rgb1='.69 .74 .79', rgb2='.69 .74 .79', width='256', height='1536')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode'))
        for field in ('body_mass','body_inertia','geom_size','geom_pos','geom_quat','jnt_axis','jnt_range','dof_damping'):
            assert np.array_equal(getattr(original,field),getattr(self.model,field)), field
        self.index = fx.opt.tip.LeanIndex(self.model)
        fx.pilot.apply_impedance(self.model, self.index, self.record['config'])
        self.data = mujoco.MjData(self.model)
        self.ground = fx.opt.tip.stage1.ground_geom_id(self.model)
        self.tr = dict(np.load(self.path.with_suffix('.npz')))
        self.sites = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'site_foot_{s}_{side}') for s in range(19) for side in range(2)])
        light_and_color(self.model, self.ground)
        # MuJoCo's default near plane scales with whole-model extent and clips
        # a millimetre-scale distal close-up. This is rendering-only.
        self.model.vis.map.znear = 1e-4
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[[0, 2]] = 1
        self.option.sitegroup[:] = 0
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)

    def pose(self, t):
        i = int(np.argmin(np.abs(self.tr['t'] - t)))
        tr, ix, d = self.tr, self.index, self.data
        d.qpos[:3] = tr['xpos'][i, 0]
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, np.asarray(tr['rootmat'][i], dtype=float).reshape(9))
        d.qpos[3:7] = q
        d.qpos[ix.qpos_addresses[ix.body_positions]] = tr['bq'][i, :, 0]
        d.qpos[ix.qpos_addresses[ix.pitch_positions]] = tr['bq'][i, :, 1]
        d.qpos[ix.roll_qpos_addresses] = tr['bq'][i, :, 2]
        d.qpos[ix.qpos_addresses[ix.leg_positions.reshape(-1)]] = tr['lq'][i]
        d.qpos[ix.tip_qpos_addresses] = tr['tq'][i]
        d.time = float(tr['t'][i])
        mujoco.mj_kinematics(self.model, d)
        return i

    def panel(self, az, el, distance, local=False):
        ix, m, d = self.index, self.model, self.data
        focus_joint = int(ix.tip_joint_ids[18])
        focus_body = int(m.jnt_bodyid[focus_joint])
        center = (d.xpos[focus_body] + d.site_xpos[self.sites[18]]) * .5 if local else d.xpos[ix.chassis_body_ids].mean(axis=0)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = az, el, distance
        cam.lookat[:] = center if local else (center[0], center[1], .004 if self.record['terrain'] == 'flat' else center[2] - .001)
        self.renderer.update_scene(d, camera=cam, scene_option=self.option)
        scene = self.renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        if self.record['terrain'] == 'flat' and not local:
            for axis in (0, 1):
                for j in range(int(np.floor((center[axis] - .12) / .01)), int(np.ceil((center[axis] + .12) / .01)) + 1):
                    a = np.array([center[0] - .12, center[1] - .12, 8e-6])
                    b = np.array([center[0] + .12, center[1] + .12, 8e-6])
                    a[axis] = b[axis] = j * .01
                    line(scene, a, b, [.47, .50, .53, 1], 1)
        overlays = []
        if local:
            # Actual tip and neutral tip reference; both use the same proximal pose.
            for seg in (9,):
                for side in (0,):
                    joint = int(ix.tip_joint_ids[seg * 2 + side])
                    body = int(m.jnt_bodyid[joint])
                    pivot = d.xpos[body].copy()
                    actual = d.site_xpos[self.sites[seg * 2 + side]].copy()
                    angle = float(d.qpos[m.jnt_qposadr[joint]])
                    axis = d.xmat[body].reshape(3, 3) @ m.jnt_axis[joint]
                    v = actual - pivot
                    # Undo the hinge angle in world space (Rodrigues).
                    neutral = pivot + v * np.cos(angle) - np.cross(axis, v) * np.sin(angle) + axis * np.dot(axis, v) * (1 - np.cos(angle))
                    line(scene, pivot, neutral, [.03, .62, .90, 1], 2)
                    line(scene, pivot, actual, [1., .25, .03, 1], 3)
                    overlays.append((pivot, neutral, actual))
        panel = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(panel)
        for pivot, neutral, actual in overlays:
            p, n, a = (project(scene, v) for v in (pivot, neutral, actual))
            for u in np.arange(0., 1., .2):
                q1, q2 = p + u * (n-p), p + min(1.,u+.12) * (n-p)
                draw.line([tuple(q1),tuple(q2)],fill=(0,211,255),width=3)
            draw.line([tuple(p),tuple(a)],fill=(255,99,0),width=2)
            draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=(245,245,235))
        return panel

    def check_pose(self):
        be = fe = 0.
        for t in np.linspace(0, float(self.tr['t'][-1]), 24):
            i = self.pose(t)
            be = max(be, float(np.abs(self.data.xpos[self.index.chassis_body_ids] - self.tr['xpos'][i]).max()))
            fe = max(fe, float(np.abs(self.data.site_xpos[self.sites] - self.tr['fq'][i]).max()))
        assert max(be, fe) < 1e-7, (be, fe)
        return dict(record=str(self.path),body_error_m=be,foot_error_m=fe,checked_poses=24)


def make(paths, output, end=6., local=False):
    states = [Replay(path) for path in paths]
    views = [('TOP LOCAL / LEG 10-L', 90, -90, .012), ('SIDE LOCAL / LEG 10-L', 0, -18, .012), ('OBLIQUE LOCAL / LEG 10-L', 45, -30, .014)] if local else [('FRONT', 180, -20, .12), ('SIDE', 90, -18, .115), ('OBLIQUE', 135, -30, .13)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pose_checks = [s.check_pose() for s in states]
    writer = imageio.get_writer(output, fps=30, codec='libx264', quality=9, macro_block_size=1)
    try:
        for frame in range(int(end * 30)):
            t = frame / 30
            canvas = Image.new('RGB', (1920, 440 * len(states)), (23, 28, 34))
            draw = ImageDraw.Draw(canvas)
            for row, s in enumerate(states):
                i = s.pose(t)
                r = s.record
                y = row * 440
                failed = [k for k, v in r['gates'].items() if not v]
                status = 'DIAGNOSTIC: NOT CERTIFIED' if failed else 'ABSOLUTE GATES PASSED | CANDIDATE'
                draw.text((14, y + 4), f"TIP-LOCAL Z FLEXION | {r['terrain'].upper()} | wave={r['case']['wave']:g} | t={t:.2f}s | {status}", font=FONT_BOLD, fill=(240, 238, 232))
                cand = r['model_candidate']
                draw.text((14, y + 30), f"k={cand['stiffness_nm_rad']:.3g} Nm/rad | c={cand['damping_nms_rad']:.3g} Nms/rad | erasure OFF | orange: tip; cyan: neutral reference" if local else f"k={cand['stiffness_nm_rad']:.3g} Nm/rad | c={cand['damping_nms_rad']:.3g} Nms/rad | fixed controller/amplitudes | contact erasure OFF", font=FONT, fill=(203, 216, 226))
                for col, (name, az, el, distance) in enumerate(views):
                    draw.text((col * 640 + 14, y + 55), name, font=FONT, fill=(151, 212, 238))
                    canvas.paste(s.panel(az, el, distance, local), (col * 640, y + 80))
            writer.append_data(np.asarray(canvas))
            if frame in (0, 90, 150):
                canvas.save(output.with_name(output.stem + f'_frame{frame:03d}.png'))
    finally:
        writer.close()
        for s in states:
            s.renderer.close()
    reader = imageio.get_reader(output)
    count = reader.count_frames()
    shape = list(reader.get_data(0).shape)
    reader.close()
    info = dict(video=str(output),frames=count,shape=shape,records=[str(p) for p in paths],
                smoothing=False,physics_changed=False,pose_checks=pose_checks,
                render_only='collision geoms hidden; hfield not overlaid; diffuse lighting; neutral-axis lines')
    output.with_suffix('.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--records', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--end', type=float, default=6.)
    p.add_argument('--local', action='store_true')
    a = p.parse_args()
    make(a.records, a.output, a.end, a.local)
