#!/usr/bin/env python3
"""Ubuntu K1 arm demo: supported pelvis, original MuJoCo dynamics and PD control.

Save beside START_UBUNTU.sh and run: python3 ARM_DEMO.py
Uses the simulator's existing virtual environment. No training or downloads.
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
    parser.add_argument('--frames-dir', type=Path, help='Optional offscreen verification images')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    candidates = [args.project] if args.project else [
        here / 'ai_sapiens_ubuntu/ai_sapiens_simulator',
        here / 'ai_sapiens_simulator', here,
    ]
    project = next((p.resolve() for p in candidates if p and (p / 'run_simulator.py').is_file()), None)
    if project is None:
        raise SystemExit('ARM_DEMO.py를 START_UBUNTU.sh와 같은 폴더에 넣어 주세요. 또는 --project 경로를 지정하세요.')
    venv = Path(os.environ.get('SAPIENS_VENV_DIR', str(project / '.venv'))).resolve()
    interpreter = venv / 'bin/python'
    if not interpreter.is_file():
        raise SystemExit('기존 실행 환경이 없습니다. 먼저 bash START_UBUNTU.sh를 실행해 주세요.')
    if Path(sys.prefix).resolve() != venv:
        os.execv(str(interpreter), [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]])
    if args.headless and args.frames_dir:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, str(project))

    import mujoco
    import numpy as np
    from run_simulator import DEFAULT_MODEL
    from sapiens_sim.mujoco_engine import MuJoCoEngine
    from sapiens_sim.motions import MotionClip, MotionKeyframe, JointMotionPlayer
    from sapiens_sim.scenarios import build_flat_lab_scene

    scene = project / 'generated_scenes/k1_supported_arm_demo.xml'
    build_flat_lab_scene(DEFAULT_MODEL, scene)
    tree = ET.parse(scene)
    xml = tree.getroot()
    equality = xml.find('equality')
    if equality is None:
        equality = ET.SubElement(xml, 'equality')
    ET.SubElement(equality, 'weld', name='arm_demo_pelvis_support', body1='pelvis',
                  solref='0.004 1', solimp='0.999 0.9999 0.001')
    world = xml.find('worldbody')
    for name, pos, size in [
        ('support_base', '-0.32 0 0.04', '0.17 0.22 0.04'),
        ('support_post', '-0.32 0 0.41', '0.035 0.035 0.37'),
        ('support_arm', '-0.16 0 0.7955', '0.16 0.025 0.025'),
    ]:
        ET.SubElement(world, 'geom', name=name, type='box', pos=pos, size=size,
                      rgba='0.27 0.40 0.48 1', contype='0', conaffinity='0', group='1')
    tree.write(scene, encoding='utf-8', xml_declaration=True)

    def arm_clip():
        names = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
                 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']
        def pose(degrees):
            return dict(zip(names, np.deg2rad(degrees)))
        home = pose([0, 0, 0, 0, 0])
        ready = pose([0, -15, 0, 10, 0])
        raised = pose([-60, -15, 0, 25, 0])
        return MotionClip('Supported right-arm raise', (
            MotionKeyframe(0.0, 'Start', home),
            MotionKeyframe(1.0, 'Arm clear of torso', ready),
            MotionKeyframe(3.2, 'Raise arm', raised),
            MotionKeyframe(4.5, 'Hold', raised),
            MotionKeyframe(7.0, 'Lower arm', ready),
            MotionKeyframe(8.0, 'Home', home),
        ))

    class SupportedArmEngine(MuJoCoEngine):
        def __init__(self, path):
            self.arm_player = None
            super().__init__(path)
            self.set_joint_effort_limits(np.minimum(self.effort_limits, self.hardware_torque_rating))

        def _update_control(self):
            if self.arm_player is not None:
                self.arm_player.update()
            super()._update_control()

    engine = SupportedArmEngine(scene)
    engine.arm_player = JointMotionPlayer(engine, arm_clip())

    if args.headless:
        start = engine.snapshot()
        arm_index = engine.joint_names.index('right_shoulder_pitch_joint')
        wrist_id = mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_roll_rubber_hand')
        initial_hand = engine.data.xpos[wrist_id].copy()
        max_base_drift = max_hand_distance = peak_torque_ratio = 0.0
        minimum_shoulder = 0.0
        max_tracking_error = 0.0
        finite = True
        targets_in_range = True
        renderer = None
        saved = set()
        if args.frames_dir:
            from PIL import Image
            args.frames_dir.mkdir(parents=True, exist_ok=True)
            engine.model.vis.global_.offwidth = 960
            engine.model.vis.global_.offheight = 720
            renderer = mujoco.Renderer(engine.model, width=960, height=720)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [0.08, 0, 0.75]
            camera.distance = 2.7
            camera.azimuth = 35
            camera.elevation = -16
        engine.arm_player.start()
        try:
            for _ in range(round(9.0 / engine.timestep)):
                engine.step()
                state = engine.snapshot()
                finite = finite and bool(np.isfinite(engine.data.qpos).all() and np.isfinite(engine.data.qvel).all())
                targets_in_range = targets_in_range and bool(
                    np.all(engine.joint_target >= engine.joint_ranges[:, 0]) and
                    np.all(engine.joint_target <= engine.joint_ranges[:, 1]))
                max_base_drift = max(max_base_drift, float(np.linalg.norm(state.base_position - start.base_position)))
                max_hand_distance = max(max_hand_distance, float(np.linalg.norm(engine.data.xpos[wrist_id] - initial_hand)))
                peak_torque_ratio = max(peak_torque_ratio, float(np.max(np.abs(state.commanded_torque) / engine.effort_limits)))
                minimum_shoulder = min(minimum_shoulder, float(state.joint_position[arm_index]))
                max_tracking_error = max(max_tracking_error, abs(float(state.joint_position[arm_index] - state.joint_target[arm_index])))
                if renderer:
                    for index, moment in enumerate([0.002, 3.8, 8.9]):
                        if state.time >= moment and index not in saved:
                            renderer.update_scene(engine.data, camera)
                            Image.fromarray(renderer.render()).save(args.frames_dir / f'arm_{index}.png')
                            saved.add(index)
        finally:
            if renderer:
                renderer.close()
        result = {
            'mode': 'supported_pelvis_PD_arm_demo', 'learned_policy': False,
            'simulated_seconds': state.time, 'finite': finite,
            'joint_targets_within_limits': targets_in_range,
            'pelvis_max_drift_m': max_base_drift,
            'right_hand_max_displacement_m': max_hand_distance,
            'right_shoulder_min_degrees': float(np.rad2deg(minimum_shoulder)),
            'right_shoulder_final_degrees': float(np.rad2deg(state.joint_position[arm_index])),
            'shoulder_peak_tracking_error_degrees': float(np.rad2deg(max_tracking_error)),
            'peak_commanded_torque_limit_ratio': peak_torque_ratio,
        }
        result['passed'] = bool(finite and targets_in_range and max_base_drift < 0.01 and
                                minimum_shoulder < np.deg2rad(-40) and
                                abs(state.joint_position[arm_index]) < np.deg2rad(8) and
                                peak_torque_ratio <= 1.000001)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['passed'] else 1)

    from tkinter import ttk
    from sapiens_sim.ui import AISapiensSimulatorApp

    class ArmDemoApp(AISapiensSimulatorApp):
        def __init__(self, engine):
            super().__init__(engine, project / 'outputs')
            self.root.title('AI Sapiens - Supported Arm Demo')
            panel = ttk.Frame(self.root, padding=8)
            panel.pack(side='top', fill='x', before=self.root.winfo_children()[0])
            ttk.Label(panel, text='PELVIS SUPPORTED | Right-arm raise: 8 seconds',
                      foreground='#ffd166').pack(side='left', padx=8)
            ttk.Button(panel, text='Play Arm', command=self.play_arm).pack(side='left', padx=5)
            ttk.Button(panel, text='Stop Motion', command=self.stop_arm).pack(side='left', padx=5)
            ttk.Label(panel, text='Stop Motion to use Joints manually. Pause stops physics.',
                      foreground='#b9d2e5').pack(side='left', padx=10)
            self.camera.lookat[:] = (0.08, 0, 0.75)
            self.camera.distance = 2.7
            self.camera.azimuth = 35
            self.camera.elevation = -16
            self.show_contacts.set(False)
            self.pause()
            self.status.set('Supported pelvis - press Play Arm')

        def play_arm(self):
            self.reset()
            self.speed.set(1.0)
            self.engine.arm_player.start()
            self.play()
            self.status.set('Arm motion playing - pelvis supported')

        def stop_arm(self):
            self.engine.arm_player.stop()
            self.engine.set_joint_targets(self.engine.snapshot().joint_position)
            self.status.set('Motion stopped - pelvis supported')

        def reset(self):
            self.engine.arm_player.stop()
            super().reset()

        def _tick(self):
            super()._tick()
            if self.running and self.engine.arm_player.active:
                self.status.set(f'{self.engine.arm_player.phase} - {self.engine.arm_player.elapsed:.1f}/8.0 s')

    ArmDemoApp(engine).run()


if __name__ == '__main__':
    main()
