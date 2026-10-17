"""Controlled side-contact bench test of the same distal leg assembly.

One tip DOF, identical geometry/mass/inertia/k/c, old versus local-Z axis.
The proximal rigid leg is prescribed by a mocap body; this is a functional
mechanism test, not a locomotion or animal-validation experiment.
"""
from __future__ import annotations
import copy
import argparse
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('probe_fx', ROOT / 'run_tip_flexion_search.py')
fx = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fx
spec.loader.exec_module(fx)
np, mujoco = fx.np, fx.mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

OUT = fx.OUT / 'side_contact_probe'
K = 2.5e-5
DAMPING_SCALE = 4.2
C = fx.damping_for(K, DAMPING_SCALE)
DT = .000125
DURATION = 2.
FONT = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 19)


def make_model(axis: str) -> Path:
    source = ET.parse(fx.source_model('flat')).getroot()
    leg = next(b for b in source.iter('body') if b.get('name') == 'link_leg_0_L_1')
    leg = copy.deepcopy(leg)
    for j in list(leg.findall('joint')):
        leg.remove(j)
    leg.set('name', 'probe_base')
    leg.set('mocap', 'true')
    leg.set('pos', '-0.002 0 0.003')
    tip = next(j for j in leg.iter('joint') if j.get('name', '').startswith('joint_tip_'))
    tip.set('axis', '0 0 1' if axis == 'local_z' else '0 0.866025403784 -0.5')
    tip.set('stiffness', fx.number(K))
    tip.set('damping', fx.number(C))
    root = ET.Element('mujoco', model=f'tip_side_contact_{axis}')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='true', inertiagrouprange='0 0')
    ET.SubElement(root, 'option', timestep=str(DT), integrator='implicitfast', gravity='0 0 0', cone='elliptic')
    root.append(copy.deepcopy(source.find('default')))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='640', offheight='360')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', type='skybox', builtin='gradient', rgb1='.63 .68 .73', rgb2='.63 .68 .73', width='256', height='1536')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -0.004 0.02', dir='0 0 -1', directional='true', diffuse='0.5 0.5 0.5', castshadow='false')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='.03 .03 .001', contype='0', conaffinity='0', rgba='.72 .76 .80 1')
    ET.SubElement(world, 'geom', name='obstacle', type='box', pos='0 -0.0077 0.0015', size='0.0005 0.0005 0.0015', group='2', rgba='.40 .31 .57 1')
    world.append(leg)
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'probe_{axis}.xml'
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return path


def base_x(t):
    u = float(np.clip((t - .1) / 1.5, 0, 1))
    return -.002 + .004 * (3 * u * u - 2 * u * u * u)


def run(axis):
    path = make_model(axis)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'joint_tip_0_L_roll')
    bid = int(m.jnt_bodyid[jid])
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'site_foot_0_0')
    obstacle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle')
    qadr = int(m.jnt_qposadr[jid])
    rows = []
    cf = np.zeros(6)
    for step in range(int(DURATION / DT)):
        t = step * DT
        d.mocap_pos[0, 0] = base_x(t)
        mujoco.mj_step(m, d)
        # Kinematic coordinates refer to this step's pre-integration pose.
        force = 0.
        for j in range(d.ncon):
            con = d.contact[j]
            if obstacle not in (int(con.geom1), int(con.geom2)):
                continue
            mujoco.mj_contactForce(m, d, j, cf)
            force += max(0., float(cf[0]))
        if step % 8 == 0:
            mujoco.mj_kinematics(m, d)
            pivot = d.xpos[bid].copy()
            actual = d.site_xpos[site].copy()
            theta = float(d.qpos[qadr])
            world_axis = d.xmat[bid].reshape(3, 3) @ m.jnt_axis[jid]
            v = actual - pivot
            neutral = pivot + v * np.cos(theta) - np.cross(world_axis, v) * np.sin(theta) + world_axis * np.dot(world_axis, v) * (1 - np.cos(theta))
            rows.append((t, theta, float(d.qvel[0]), force, *actual, *neutral, *pivot, base_x(t)))
    a = np.asarray(rows)
    contact = a[:, 3] > 1e-8
    dx = a[:, 4] - a[:, 7]
    angle = np.degrees(a[:, 1])
    last_contact = float(a[contact, 0][-1]) if contact.any() else None
    summary = dict(
        axis=axis,model=str(path),model_sha256=fx.sha256(path),nq=m.nq,nv=m.nv,
        k_nm_rad=K,c_nms_rad=C,limit_deg=38.,gravity='disabled for controlled side-force bench',
        contact_occurred=bool(contact.any()),peak_abs_angle_deg=float(np.abs(angle).max()),
        peak_normal_force_n=float(a[:, 3].max()),maximum_backward_tip_deflection_mm=float(-dx.min() * 1000),
        contact_mean_dx_mm=float(dx[contact].mean() * 1000) if contact.any() else None,
        last_contact_s=last_contact,final_angle_deg=float(angle[-1]),
        tip_speed_peak_rad_s=float(np.abs(a[:, 2]).max()),
        passed=dict(side_contact=bool(contact.any()),bends_backward=bool(contact.any() and dx[contact].mean() < -1e-5),
                    clears_obstacle=bool(last_contact is not None and last_contact < 1.8),
                    recovers=bool(abs(angle[-1]) < 1),no_limit_overrun=bool(np.abs(angle).max() < 40)),
    )
    np.savez_compressed(OUT / f'probe_{axis}.npz', trajectory=a)
    (OUT / f'probe_{axis}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return summary


def add_line(scene, a, b, color):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).reshape(9), np.asarray(color, dtype=float))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3., a, b)
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
    depth = np.dot(v, forward)
    near = .5 * (a.frustum_near + b.frustum_near)
    bottom = .5 * (a.frustum_bottom + b.frustum_bottom)
    top = .5 * (a.frustum_top + b.frustum_top)
    center = .5 * (a.frustum_center + b.frustum_center)
    width = (top - bottom) * 640 / 360
    xx = np.dot(v, right) * near / depth
    yy = np.dot(v, up) * near / depth
    return (320 + 640 * (xx - center) / width, 360 - 360 * (yy - bottom) / (top - bottom))


def overlay_reference(panel, scene, pivot, neutral, actual):
    draw = ImageDraw.Draw(panel)
    p, n, a = (np.asarray(project(scene, v)) for v in (pivot, neutral, actual))
    for u in np.arange(0., 1., .2):
        q1, q2 = p + u * (n - p), p + min(1., u + .12) * (n - p)
        draw.line([tuple(q1), tuple(q2)], fill=(0, 211, 255), width=3)
    draw.line([tuple(p), tuple(a)], fill=(255, 99, 0), width=2)
    draw.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=(245,245,235))


def render(summaries):
    states = []
    for r in summaries:
        m = mujoco.MjModel.from_xml_path(r['model'])
        d = mujoco.MjData(m)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[[0, 2]] = 1
        opt.sitegroup[:] = 0
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            if name.endswith('_collision'):
                m.geom_group[g] = 5
        m.vis.headlight.ambient[:] = .3
        m.vis.headlight.diffuse[:] = .45
        m.vis.headlight.specular[:] = 0
        a = np.load(OUT / f"probe_{r['axis']}.npz")['trajectory']
        states.append((r, m, d, opt, a, mujoco.Renderer(m, height=360, width=640)))
    path = OUT / 'side_contact_axis_comparison_three_views.mp4'
    writer = imageio.get_writer(path, fps=30, codec='libx264', quality=9, macro_block_size=1)
    views = [('TOP', 90, -90), ('SIDE', 0, -18), ('OBLIQUE', 45, -35)]
    peak_t = float(states[1][4][np.argmax(np.abs(states[1][4][:, 1])), 0])
    peak_frame = int(round(peak_t * 30 * 4))
    try:
        for frame in range(int(DURATION * 30 * 4)):
            t = frame / (30 * 4)
            canvas = Image.new('RGB', (1920, 840), (23, 28, 34))
            draw = ImageDraw.Draw(canvas)
            for row, (r, m, d, opt, a, renderer) in enumerate(states):
                i = int(np.argmin(np.abs(a[:, 0] - t)))
                d.qpos[0] = a[i, 1]
                d.mocap_pos[0] = (a[i, 13], 0, .003)
                mujoco.mj_kinematics(m, d)
                y = row * 420
                draw.text((12, y + 3), f"{'OLD AXIS' if row == 0 else 'TIP-LOCAL Z'} | SAME k/c/geometry | t={t:.3f}s | angle={np.degrees(a[i,1]):+.1f} deg | side force={a[i,3]*1000:.2f} mN", font=FONT, fill=(242, 240, 233))
                draw.text((12, y + 29), 'Controlled side-contact bench | 0.25x playback | orange: actual tip | cyan: neutral | no locomotion controller', font=FONT, fill=(164, 210, 234))
                for col, (label, az, el) in enumerate(views):
                    cam = mujoco.MjvCamera()
                    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    cam.azimuth, cam.elevation, cam.distance = az, el, .009
                    cam.lookat[:] = (0, -.0065, .0014)
                    renderer.update_scene(d, camera=cam, scene_option=opt)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    add_line(renderer.scene, a[i, 10:13], a[i, 7:10], [.03, .60, .88, 1])
                    add_line(renderer.scene, a[i, 10:13], a[i, 4:7], [1., .24, .02, 1])
                    panel = Image.fromarray(renderer.render())
                    overlay_reference(panel, renderer.scene, a[i,10:13], a[i,7:10], a[i,4:7])
                    canvas.paste(panel, (col * 640, y + 60))
                    draw.rectangle((col*640+7,y+62,col*640+120,y+89),fill=(23,28,34))
                    draw.text((col * 640 + 12, y + 62), label, font=FONT, fill=(195, 228, 241))
            writer.append_data(np.asarray(canvas))
            if frame in (0, peak_frame, 225):
                canvas.save(OUT / f'side_contact_frame{frame:03d}.png')
    finally:
        writer.close()
        for state in states:
            state[-1].close()
    print(json.dumps(dict(video=str(path),peak_contact_frame=str(OUT / f'side_contact_frame{peak_frame:03d}.png'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stiffness', type=float, default=K)
    parser.add_argument('--damping-scale', type=float, default=DAMPING_SCALE)
    parser.add_argument('--output-subdir', default='side_contact_probe')
    args = parser.parse_args()
    K = args.stiffness
    DAMPING_SCALE = args.damping_scale
    C = fx.damping_for(K, DAMPING_SCALE)
    OUT = fx.OUT / args.output_subdir
    results = [run(axis) for axis in ('old_radial', 'local_z')]
    render(results)
