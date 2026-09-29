#!/usr/bin/env bash
# Put this file beside ai_sapiens_simulator.zip, then run: bash START_UBUNTU.sh
# Optional engine check: bash START_UBUNTU.sh --headless --duration 3
set -Eeuo pipefail
trap 'printf "\n실행이 중단됐습니다. 위의 오류 내용을 확인해 주세요.\n" >&2' ERR

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_dir="${SAPIENS_WORK_DIR:-$script_dir/ai_sapiens_ubuntu}"
python_bin="${SAPIENS_PYTHON:-python3}"
headless=0
for arg in "$@"; do
    if [[ "$arg" == "--headless" ]]; then headless=1; fi
done

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python 3가 필요합니다: sudo apt install python3 python3-venv python3-tk\n' >&2
    exit 1
fi
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 이상이 필요합니다."'

if [[ "$headless" == 0 ]]; then
    if [[ -z "${DISPLAY:-}" ]]; then
        printf 'Ubuntu 바탕화면에서 터미널을 열어 실행하세요. 현재 DISPLAY가 없습니다.\n' >&2
        printf '화면 없는 물리 계산 확인: bash START_UBUNTU.sh --headless --duration 3\n' >&2
        exit 1
    fi
    if ! "$python_bin" -c 'import tkinter' >/dev/null 2>&1; then
        printf '화면 패키지가 필요합니다: sudo apt install python3-tk\n' >&2
        exit 1
    fi
fi

printf '\n[1/3] 시뮬레이터와 ZIP에 포함된 K1 모델을 준비합니다.\n'
"$python_bin" - "$script_dir" "$work_dir" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import stat
import sys

source_dir, root = (Path(p).resolve() for p in sys.argv[1:])
project = root / "ai_sapiens_simulator"

def extract_safely(archive, destination, prefix=None):
    destination = destination.resolve()
    for item in archive.infolist():
        name = item.filename
        if prefix is not None:
            if not name.startswith(prefix):
                continue
            name = name[len(prefix):]
        if not name or item.is_dir():
            continue
        if stat.S_ISLNK(item.external_attr >> 16):
            raise RuntimeError(f"ZIP 심볼릭 링크는 허용하지 않습니다: {name}")
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise RuntimeError(f"잘못된 ZIP 경로: {name}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))

if not (project / "run_simulator.py").is_file():
    archive_path = source_dir / "ai_sapiens_simulator.zip"
    if not archive_path.is_file():
        candidates = sorted(source_dir.glob("ai_sapiens_simulator*.zip"))
        if len(candidates) != 1:
            raise SystemExit("원본 ai_sapiens_simulator.zip과 START_UBUNTU.sh를 같은 폴더에 넣어 주세요.")
        archive_path = candidates[0]
    with ZipFile(archive_path) as archive:
        if "ai_sapiens_simulator/run_simulator.py" not in archive.namelist():
            raise SystemExit("예상한 시뮬레이터 ZIP 구조가 아닙니다.")
        extract_safely(archive, root)

model_root = root / "AI_Sapiens_research/software/ros2/ai_sapiens"
bundle = project / "colab/ai_sapiens_cyclo_stairs.zip"
with ZipFile(bundle) as archive:
    extract_safely(archive, model_root, "cyclo_mjlab/third_party/ai_sapiens/")
description = model_root / "ai_sapiens_description"
required = [description / "mujoco/k1/scene.xml", description / "mujoco/k1/k1.xml"]
if not all(p.is_file() for p in required):
    raise SystemExit("K1 모델을 복원하지 못했습니다.")
meshes = list((description / "meshes/k1_rev1").glob("*.stl"))
if len(meshes) != 25:
    raise SystemExit(f"K1 STL이 25개 필요하지만 {len(meshes)}개를 찾았습니다.")
print(f"프로젝트: {project}")
print(f"모델 준비 완료: STL {len(meshes)}개")
PY

project_dir="$work_dir/ai_sapiens_simulator"
venv_dir="${SAPIENS_VENV_DIR:-$project_dir/.venv}"
printf '\n[2/3] Python 실행 환경을 준비합니다. 첫 실행에는 다운로드가 필요합니다.\n'
if [[ ! -x "$venv_dir/bin/python" ]]; then
    if ! "$python_bin" -m venv "$venv_dir"; then
        printf 'venv 설치 후 다시 실행하세요: sudo apt install python3-venv\n' >&2
        exit 1
    fi
fi
requirements_hash="$(sha256sum "$project_dir/requirements.txt" | cut -d ' ' -f 1)"
installed_hash="$(cat "$venv_dir/.sapiens_requirements_sha256" 2>/dev/null || true)"
if [[ "$requirements_hash" != "$installed_hash" ]] || \
   ! "$venv_dir/bin/python" -c 'import mujoco, numpy, PIL, onnxruntime, yaml' >/dev/null 2>&1; then
    "$venv_dir/bin/python" -m pip install -r "$project_dir/requirements.txt"
    printf '%s\n' "$requirements_hash" > "$venv_dir/.sapiens_requirements_sha256"
fi

cd -- "$project_dir"
if [[ "$headless" == 1 ]]; then
    printf '\n[3/3] 화면 없이 물리 시뮬레이션을 실행합니다.\n'
    exec "$venv_dir/bin/python" run_simulator.py "$@"
fi

printf '\n[3/3] 시뮬레이터 창을 엽니다.\n'
printf '처음에는 Paused 상태입니다. 왼쪽 위 Run을 누르면 물리 계산이 시작됩니다.\n'
printf '기본 보행 정책은 원본 ZIP에 없어 자동 로드하지 않습니다.\n'
printf '보행 정책 없이 Run을 누르면 로봇이 넘어질 수 있습니다.\n\n'
exec "$venv_dir/bin/python" - "$@" <<'PY'
from run_simulator import main
from sapiens_sim.ui import AISapiensSimulatorApp
import sapiens_sim.ui

class InitiallyPausedApp(AISapiensSimulatorApp):
    def run(self):
        self.pause()
        super().run()

# Change initial UI state only; the uploaded physics and control code is unchanged.
sapiens_sim.ui.AISapiensSimulatorApp = InitiallyPausedApp
main()
PY
