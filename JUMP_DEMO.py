#!/usr/bin/env python3
"""Small free-standing K1 jump using the installed AI Sapiens simulator.

Save in the same folder as START_UBUNTU.sh. Run: python3 JUMP_DEMO.py
Authored simulation controller, not a trained or hardware-validated policy.
The floating base remains free. No external lift or runtime pose playback.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record-dir', type=Path, help='Optional numerical verification output')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    candidates = [args.project] if args.project else [
        here / 'ai_sapiens_ubuntu/ai_sapiens_simulator',
        here / 'ai_sapiens_simulator', here,
    ]
    project = next((p.resolve() for p in candidates if p and (p / 'run_simulator.py').is_file()), None)
    if project is None:
        raise SystemExit('JUMP_DEMO.py를 START_UBUNTU.sh와 같은 폴더에 넣어 주세요. 또는 --project 경로를 지정하세요.')
    venv = Path(os.environ.get('SAPIENS_VENV_DIR', str(project / '.venv'))).resolve()
    interpreter = venv / 'bin/python'
    if not interpreter.is_file():
        raise SystemExit('먼저 bash START_UBUNTU.sh로 시뮬레이터 실행 환경을 준비해 주세요.')
    # Small dense control matrices run well with one BLAS thread.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if Path(sys.prefix).resolve() != venv:
        os.execv(str(interpreter), [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.path.insert(0, str(project))

    import mujoco
    import numpy as np
    from run_simulator import DEFAULT_MODEL
    from sapiens_sim.mujoco_engine import MuJoCoEngine
    from sapiens_sim.scenarios import build_flat_lab_scene

    # Generate a separate scene. Keep the original freejoint, masses, collision
    # geometry, actuator definitions, gravity and integrator.
    scene = project / 'generated_scenes/k1_small_jump_demo.xml'
    build_flat_lab_scene(DEFAULT_MODEL, scene)
    tree = ET.parse(scene)
    world = tree.getroot().find('worldbody')
    ball = world.find("body[@name='soccer_ball']")
    if ball is not None:
        world.remove(ball)
    tree.write(scene, encoding='utf-8', xml_declaration=True)

    def orientation_error(rotation, reference):
        delta = reference @ rotation.T
        return .5 * np.array([delta[2, 1] - delta[1, 2],
                              delta[0, 2] - delta[2, 0],
                              delta[1, 0] - delta[0, 1]])

    def smooth5(u):
        u = float(np.clip(u, 0, 1))
        return 10*u**3 - 15*u**4 + 6*u**5

    class JumpEngine(MuJoCoEngine):
        def __init__(self):
            self.controller_ready = False
            super().__init__(scene)
            self.ids = {name: i for i, name in enumerate(self.joint_names)}
            self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         side + '_ankle_roll_link') for side in ('left', 'right')]
            self.pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
            self.floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
            self.soles = {i for i in range(self.model.ngeom) if 'ankle_roll_link_collision' in
                          (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or '')}
            self.upper = np.arange(12, 23)
            self.mass_matrix = np.zeros((self.model.nv, self.model.nv))
            self.set_joint_effort_limits(np.minimum(self.effort_limits, self.hardware_torque_rating))
            self.controller_ready = True
            self.reset()

        def reset(self):
            super().reset()
            if not self.controller_ready:
                return
            # Reset only: initialize a slightly bent-knee standing pose with
            # both soles 1 mm above the floor. No qpos/qvel edits during a jump.
            q = np.zeros(23)
            for side in ('left', 'right'):
                for part, value in [('hip_pitch', -.2), ('knee', .4), ('ankle_pitch', -.2),
                                    ('shoulder_roll', .15 if side == 'left' else -.15), ('elbow', .3)]:
                    q[self.ids[side + '_' + part + '_joint']] = value
            self.set_joint_state(q, np.zeros(23))
            self.data.qpos[2] += .001 - self.foot_clearance()
            mujoco.mj_forward(self.model, self.data)
            self.homeq = q.copy()
            self.homebase = self.data.qpos[:3].copy()
            self.initial_base = self.homebase.copy()
            self.footrefs = [self.foot_pose(body)[0].copy() for body in self.feet]
            self.phase = 'Ready'
            self.jump_enabled = False
            self.start_time = 0.0
            self.launch_time = None
            self.landing_time = None
            self.air_start = None
            self.air_duration = 0.0
            self.max_clearance = 0.0
            self.max_base_height = float(self.homebase[2])
            self.max_tilt = 0.0
            self.peak_limit_ratio = 0.0
            self.peak_external_wrench = 0.0
            self.finite = True
            self._kp[:] = 120.0
            self._kd[:] = 5.0

        def start_jump(self):
            self.jump_enabled = True
            self.start_time = float(self.data.time)

        @property
        def elapsed(self):
            return max(0.0, float(self.data.time) - self.start_time)

        def foot_clearance(self):
            return min(float(self.data.geom_xpos[g, 2] - self.model.geom_size[g, 0]) for g in self.soles)

        def floor_contact(self):
            return any((c.geom1 == self.floor and c.geom2 in self.soles) or
                       (c.geom2 == self.floor and c.geom1 in self.soles) for c in self.data.contact)

        def foot_pose(self, body):
            rotation = self.data.xmat[body].reshape(3, 3)
            return self.data.xpos[body] + rotation @ np.array([.03, 0, -.06]), rotation

        def jacobians(self, point, body):
            jac = np.empty((6, self.model.nv))
            derivative = np.empty_like(jac)
            mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], point, body)
            mujoco.mj_jacDot(self.model, self.data, derivative[:3], derivative[3:], point, body)
            return jac, derivative

        def update_metrics(self, contact):
            data = self.data
            if not self.jump_enabled:
                return
            clearance = self.foot_clearance()
            self.max_clearance = max(self.max_clearance, clearance)
            self.max_base_height = max(self.max_base_height, float(data.qpos[2]))
            tilt = np.arccos(np.clip(data.xmat[self.pelvis].reshape(3, 3)[2, 2], -1, 1))
            self.max_tilt = max(self.max_tilt, float(tilt))
            self.peak_limit_ratio = max(self.peak_limit_ratio,
                                       float(np.max(np.abs(self._commanded_torque) / self.effort_limits)))
            self.peak_external_wrench = max(self.peak_external_wrench, float(np.max(np.abs(data.xfrc_applied))))
            self.finite = self.finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            if self.elapsed > 1.5 and not contact:
                if self.air_start is None:
                    self.air_start = float(data.time)
                self.air_duration = max(self.air_duration, float(data.time) - self.air_start)
            elif contact:
                self.air_start = None

        def _update_control(self):
            if not self.controller_ready or not self.control_enabled:
                return super()._update_control()
            model, data = self.model, self.data
            mujoco.mj_forward(model, data)
            t = self.elapsed if self.jump_enabled else 0.0
            contact = self.floor_contact()
            self.update_metrics(contact)
            if self.phase == 'Push off' and ((not contact and data.qvel[2] > .1) or t > 1.67):
                self.phase = 'Airborne'
                self.launch_time = t
            if (self.phase == 'Airborne' and contact and data.qvel[2] < -.1 and
                    t - self.launch_time > .04):
                self.phase = 'Landing'
                self.landing_time = t
                self.footrefs = [self.foot_pose(body)[0].copy() for body in self.feet]
                for point in self.footrefs:
                    point[2] = .005
                self.homebase[:2] = data.qpos[:2]
            if self.phase == 'Airborne':
                self.joint_target[:] = self.homeq
                return super()._update_control()

            z = float(self.homebase[2])
            vz = az = 0.0
            if .5 < t < 1.3:
                u = (t - .5) / .8
                z -= .055 * smooth5(u)
                vz = -.055 * (30*u**2 - 60*u**3 + 30*u**4) / .8
                az = -.055 * (60*u - 180*u**2 + 120*u**3) / .8**2
                self.phase = 'Crouch'
            elif 1.3 <= t < 1.5:
                z -= .055
                self.phase = 'Crouch'
            elif t >= 1.5 and self.launch_time is None:
                u = t - 1.5
                z = z - .055 + .5 * 8.0 * u*u
                vz, az = 8.0*u, 8.0
                self.phase = 'Push off'
            elif self.landing_time is not None:
                u = (t - self.landing_time - .3) / 1.1
                z -= .025 * (1.0 - smooth5(u))
                self.phase = 'Standing' if u >= 1 else 'Recover'

            # Resolved-acceleration whole-body control. Grounded feet and pelvis
            # pose determine accelerations; free-base dynamics determine the
            # contact wrench; motor torques supply the actuated remainder.
            reference = self.homebase.copy()
            reference[2] = z
            body_jac, body_dot = self.jacobians(data.xpos[self.pelvis], self.pelvis)
            body_velocity = body_jac @ data.qvel
            acceleration = np.r_[
                np.array([0, 0, az]) + 100*(reference - data.qpos[:3]) +
                20*(np.array([0, 0, vz]) - body_velocity[:3]),
                160*orientation_error(data.xmat[self.pelvis].reshape(3, 3), np.eye(3)) -
                25*body_velocity[3:]] - body_dot @ data.qvel
            rows, values, foot_jacs = [body_jac], [acceleration], []
            for body, foot_reference in zip(self.feet, self.footrefs):
                position, rotation = self.foot_pose(body)
                jac, derivative = self.jacobians(position, body)
                foot_jacs.append(jac)
                rows.append(jac)
                values.append(np.r_[250*(foot_reference - position),
                                    200*orientation_error(rotation, np.eye(3))] -
                              30*(jac @ data.qvel) - derivative @ data.qvel)
            upper_jac = np.eye(model.nv)[self._dof_ids[self.upper]]
            rows.append(upper_jac)
            values.append(100*(self.homeq[self.upper] - data.qpos[self._qpos_ids[self.upper]]) -
                          20*data.qvel[self._dof_ids[self.upper]])
            matrix, target = np.vstack(rows), np.concatenate(values)
            qacc = np.linalg.lstsq(np.vstack([matrix, np.eye(model.nv)*.002]),
                                  np.r_[target, np.zeros(model.nv)], rcond=None)[0]
            qacc = np.clip(qacc, -800, 800)
            try:
                mujoco.mj_fullM(model, data, self.mass_matrix)
            except TypeError:
                # Older supported MuJoCo Python binding signature.
                mujoco.mj_fullM(model, self.mass_matrix, data.qM)
            required = self.mass_matrix @ qacc + data.qfrc_bias
            feet_jac = np.vstack(foot_jacs)
            wrench = np.linalg.lstsq(feet_jac[:, :6].T, required[:6], rcond=None)[0]
            torque = (required - feet_jac.T @ wrench)[self._dof_ids]
            self._requested_torque[:] = torque
            self._commanded_torque[:] = np.clip(torque, -self.effort_limits, self.effort_limits)
            data.ctrl[:] = self._commanded_torque

        def report(self):
            mujoco.mj_forward(self.model, self.data)
            self.update_metrics(self.floor_contact())
            tilt = float(np.rad2deg(np.arccos(np.clip(self.data.xmat[self.pelvis].reshape(3, 3)[2, 2], -1, 1))))
            result = {
                'mode': 'free_standing_small_jump', 'mujoco_version': mujoco.__version__,
                'learned_policy': False, 'simulation_only': True,
                'simulation_seconds': float(self.data.time),
                'pelvis_support_constraints': int(self.model.neq),
                'max_external_wrench': self.peak_external_wrench,
                'airborne_seconds': self.air_duration,
                'both_feet_max_clearance_cm': 100*self.max_clearance,
                'pelvis_rise_above_initial_cm': 100*(self.max_base_height - self.initial_base[2]),
                'max_pelvis_tilt_degrees': float(np.rad2deg(self.max_tilt)),
                'final_pelvis_tilt_degrees': tilt,
                'final_pelvis_height_m': float(self.data.qpos[2]),
                'final_base_speed_mps': float(np.linalg.norm(self.data.qvel[:3])),
                'horizontal_displacement_cm': float(100*np.linalg.norm(self.data.qpos[:2] - self.initial_base[:2])),
                'peak_commanded_torque_limit_ratio': self.peak_limit_ratio,
                'finite': self.finite,
                'floor_contact_after_landing': self.floor_contact(),
            }
            result['passed'] = bool(self.finite and self.model.neq == 0 and self.peak_external_wrench == 0 and
                                    self.air_duration > .06 and self.max_clearance > .005 and
                                    self.landing_time is not None and self.floor_contact() and
                                    tilt < 5 and self.data.qpos[2] > .7 and
                                    np.linalg.norm(self.data.qvel[:3]) < .05 and self.peak_limit_ratio <= 1.000001)
            return result

    engine = JumpEngine()
    if args.headless:
        states = []
        engine.start_jump()
        for index in range(round(6.0 / engine.timestep)):
            engine.step()
            if args.record_dir and index % 10 == 0:
                states.append(engine.data.qpos.copy())
        report = engine.report()
        print(json.dumps(report, indent=2))
        if args.record_dir:
            args.record_dir.mkdir(parents=True, exist_ok=True)
            (args.record_dir / 'jump_report.json').write_text(json.dumps(report, indent=2) + '\n')
            np.savez(args.record_dir / 'jump_trajectory.npz', qpos=np.array(states),
                     timestep=engine.timestep*10, model=str(scene))
        raise SystemExit(0 if report['passed'] else 1)

    import tkinter as tk
    from tkinter import ttk
    from sapiens_sim.ui import AISapiensSimulatorApp

    class JumpDemoApp(AISapiensSimulatorApp):
        def __init__(self):
            super().__init__(engine, project / 'outputs')
            self.root.title('AI Sapiens - Small Jump Demo')
            panel = ttk.Frame(self.root, padding=8)
            panel.pack(side='top', fill='x', before=self.root.winfo_children()[0])
            ttk.Label(panel, text='SMALL JUMP | Free-standing simulation',
                      foreground='#ffd166').pack(side='left', padx=8)
            ttk.Button(panel, text='Play Jump', command=self.play_jump).pack(side='left', padx=5)
            ttk.Button(panel, text='Reset Jump', command=self.reset_jump).pack(side='left', padx=5)
            self.jump_info = tk.StringVar(value='Ready - press Play Jump. Default speed: 0.5x')
            ttk.Label(panel, textvariable=self.jump_info).pack(side='left', padx=10)
            for scale in self.joint_scales:
                scale.configure(state=tk.DISABLED)
            self.camera.lookat[:] = (.0, 0, .65)
            self.camera.distance = 2.8
            self.camera.azimuth = 135
            self.camera.elevation = -12
            self.speed.set(.5)
            self.show_contacts.set(False)
            self.pause()
            self.status.set('Jump controller ready')

        def play_jump(self):
            self.reset()
            self.engine.start_jump()
            self.play()

        def reset_jump(self):
            self.pause()
            self.reset()
            self.jump_info.set('Ready - press Play Jump')

        def _tick(self):
            super()._tick()
            if self.running and self.engine.jump_enabled:
                self.jump_info.set(f'{self.engine.phase} | Air {self.engine.air_duration:.2f} s | '
                                   f'Feet clearance {self.engine.max_clearance*100:.1f} cm')
                if self.engine.elapsed >= 4.5:
                    self.pause()
                    self.status.set('Jump complete - Play Jump to repeat')

    JumpDemoApp().run()


if __name__ == '__main__':
    main()
