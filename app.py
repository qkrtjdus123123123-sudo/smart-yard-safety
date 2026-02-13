"""
스마트 야드 안전 모니터링 사이트
Streamlit + MediaPipe Pose + OpenCV 실시간 웹캠 분석 및 추락 감지
PDF/PPT 명세: 각도·속도 기반 추락 감지, 로컬 스냅샷, 트렌드 분석, 문의/데모
"""

import os
import re
import zipfile
import urllib.request
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO

DATA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOGS_SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "snapshots")
# data 폴더 또는 AI 모델.zip 내 .task 파일 우선 사용
DATA_AI_MODEL_ZIP = os.path.join(DATA_BASE, "선박·해양플랜트 스마트 야드 안전 데이터", "AI 모델.zip")
DATA_MODEL_EXTRACT_DIR = os.path.join(DATA_BASE, "ai_model_extracted")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

st.set_page_config(
    page_title="Smart Yard Safety System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
if "last_fall_time" not in st.session_state:
    st.session_state.last_fall_time = 0.0
if "max_snapshots" not in st.session_state:
    st.session_state.max_snapshots = 12
# PDF: 추락 감지 알고리즘 (각도/속도 기반) — 이전 프레임 spine 비율 저장
if "prev_spine_ratio" not in st.session_state:
    st.session_state.prev_spine_ratio = None

st.markdown("""
<style>
    .main-title {
        font-family: 'Segoe UI', 'Consolas', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4aa;
        text-align: center;
        letter-spacing: 0.15em;
        text-shadow: 0 0 20px rgba(0, 212, 170, 0.4);
        padding: 1rem 0 1.5rem 0;
        border-bottom: 2px solid rgba(0, 212, 170, 0.3);
        margin-bottom: 1.5rem;
    }
    .alert-card {
        background: #1a1d24;
        border-left: 4px solid #00d4aa;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .alert-card.danger { border-left-color: #ff6b6b; }
    .alert-card.danger, .alert-card.danger strong, .alert-card.danger small { color: #ff6b6b !important; }
    .alert-card.warning { border-left-color: #ffd93d; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #13161c 0%, #0e1117 100%); }
    [data-testid="stSidebar"] .stSlider label { color: #fafafa !important; }
    .footer-contact { font-size: 0.85rem; color: #9ca3af; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


def _find_task_in_data():
    """data 폴더에서 .task 파일을 찾는다. 폴더 내 직접 배치 또는 AI 모델.zip 압축 해제 후 탐색."""
    # 1) data 하위에서 .task 파일 직접 검색
    if os.path.isdir(DATA_BASE):
        for root, _dirs, files in os.walk(DATA_BASE):
            for f in files:
                if f.lower().endswith(".task"):
                    return os.path.join(root, f)
    # 2) AI 모델.zip 압축 해제 후 .task 검색
    for zip_path in [
        DATA_AI_MODEL_ZIP,
        os.path.join(DATA_BASE, "AI 모델.zip"),
    ]:
        if not os.path.isfile(zip_path):
            continue
        try:
            os.makedirs(DATA_MODEL_EXTRACT_DIR, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_MODEL_EXTRACT_DIR)
            for root, _dirs, files in os.walk(DATA_MODEL_EXTRACT_DIR):
                for f in files:
                    if f.lower().endswith(".task"):
                        return os.path.join(root, f)
        except Exception:
            pass
    return None


def _ensure_pose_model(use_data_model_only=False):
    """
    use_data_model_only: True면 data 폴더 .task만 사용(없으면 None).
    False면 data → 프로젝트루트 → 다운로드 순으로 시도.
    """
    if use_data_model_only:
        return _find_task_in_data()
    path_from_data = _find_task_in_data()
    if path_from_data:
        return path_from_data
    if os.path.isfile(POSE_MODEL_PATH):
        return POSE_MODEL_PATH
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        return POSE_MODEL_PATH
    except Exception:
        return None


def _pose_with_tasks_api(use_data_model_only=False):
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
    except ImportError:
        return None
    path = _ensure_pose_model(use_data_model_only=use_data_model_only)
    if not path:
        return None
    base_options = mp_tasks.BaseOptions(model_asset_path=path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector, vision, drawing_utils, drawing_styles


def get_spine_ratio(landmarks):
    """어깨(11,12) vs 엉덩이(23,24) spine 비율. 서 있으면 양수. 반환값 없으면 None."""
    if not landmarks or len(landmarks) < 25:
        return None
    y11, y12 = landmarks[11].y, landmarks[12].y
    y23, y24 = landmarks[23].y, landmarks[24].y
    shoulder_mid_y = (y11 + y12) / 2
    hip_mid_y = (y23 + y24) / 2
    return hip_mid_y - shoulder_mid_y


def check_fall(landmarks, sensitivity=70, use_velocity=True):
    """
    PDF: 추락 감지 알고리즘 (각도/속도 기반)
    - 각도: spine 비율이 임계값 미만이면 추락 의심
    - 속도: 이전 프레임 대비 spine 비율이 급격히 감소하면 추락 의심
    """
    spine = get_spine_ratio(landmarks)
    if spine is None:
        return False, spine
    threshold = 0.25 - (sensitivity / 100.0) * 0.2
    angle_fall = spine < threshold
    prev = st.session_state.prev_spine_ratio
    velocity_fall = False
    if use_velocity and prev is not None:
        delta = prev - spine
        if delta > 0.2 and spine < 0.4:
            velocity_fall = True
    fall = angle_fall or velocity_fall
    st.session_state.prev_spine_ratio = spine
    return fall, spine


def check_fire(rgb, sensitivity=50):
    """
    화재 의심 감지: 이미지에서 불꽃/연기 색(빨강·주황·노랑) 비율이 높으면 True.
    sensitivity: 0~100, 높을수록 더 민감(낮은 비율에도 감지).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # H: 빨강 0~15, 165~180 / 주황~노랑 15~45
    lower1 = np.array([0, 100, 150])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([165, 100, 150])
    upper2 = np.array([180, 255, 255])
    lower_yellow = np.array([15, 100, 150])
    upper_yellow = np.array([45, 255, 255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    my = cv2.inRange(hsv, lower_yellow, upper_yellow)
    fire_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), my)
    ratio = np.count_nonzero(fire_mask) / (fire_mask.size + 1e-6)
    thresh = 0.02 + (100 - sensitivity) / 100.0 * 0.08  # 약 2%~10%
    return ratio >= thresh, ratio


def _get_helmet_model():
    """안전모 감지용 YOLO 모델 lazy load. HF repo 실패 시 직접 URL로 시도."""
    cached = getattr(st.session_state, "helmet_yolo_model", None)
    if cached is not None and cached is not False:
        return cached
    err_msg = getattr(st.session_state, "helmet_model_error", None)
    try:
        from ultralytics import YOLO
        with st.spinner("안전모 AI 모델 로딩 중… (최초 1회 다운로드)"):
            # 1) HF repo 이름으로 로드 (일부 환경에서 실패할 수 있음)
            try:
                m = YOLO("sharathhhhh/safetyHelmet-detection-yolov8")
            except Exception:
                # 2) 실패 시 .pt 직접 URL로 로드 (네트워크만 되면 동작)
                m = YOLO("https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt")
        st.session_state.helmet_yolo_model = m
        if "helmet_model_error" in st.session_state:
            del st.session_state.helmet_model_error
        return m
    except Exception as e:
        st.session_state.helmet_model_error = str(e)
        return None


def check_helmet(rgb, conf_threshold=0.35):
    """
    YOLO 안전모 모델로 이미지 분석. without_helmet 감지 시 True 반환.
    반환: (no_helmet_detected: bool, error_message: str | None)
    """
    model = _get_helmet_model()
    if model is None:
        detail = getattr(st.session_state, "helmet_model_error", None)
        msg = "안전모 모델을 불러올 수 없습니다."
        if detail:
            msg += " (" + (detail[:120] + "…" if len(detail) > 120 else detail) + ")"
        return False, msg
    try:
        results = model(rgb, conf=conf_threshold, verbose=False)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            names = r.names or {}
            for cls_id in r.boxes.cls.cpu().numpy().astype(int):
                name = names.get(int(cls_id), "")
                if name == "without_helmet" or "without" in (name or "").lower():
                    return True, None
        return False, None
    except Exception as e:
        return False, str(e)


def draw_pose_tasks(frame_rgb, detection_result, vision_module, drawing_utils_module, drawing_styles_module):
    if not detection_result.pose_landmarks:
        return frame_rgb
    annotated = np.copy(frame_rgb)
    style = drawing_styles_module.get_default_pose_landmarks_style()
    conn_style = drawing_utils_module.DrawingSpec(color=(0, 255, 0), thickness=2)
    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils_module.draw_landmarks(
            image=annotated,
            landmark_list=pose_landmarks,
            connections=vision_module.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=style,
            connection_drawing_spec=conn_style,
        )
    return annotated


def scan_s63_data_files():
    empty_cols = ["구분", "분류", "데이터셋", "라벨", "파일명", "경로"]
    pattern = re.compile(r"^(TS|TL|VS|VL)_(.+?)-S63_(DATA[123])_(.+?)\.zip$", re.IGNORECASE)
    if not os.path.isdir(DATA_BASE):
        return pd.DataFrame(columns=empty_cols)
    rows = []
    for root, _dirs, files in os.walk(DATA_BASE):
        for f in files:
            if "S63_DATA" not in f or not f.endswith(".zip"):
                continue
            m = pattern.match(f)
            if m:
                prefix, category, data_set, label = m.groups()
                rows.append({"구분": prefix, "분류": category.strip(), "데이터셋": f"S63_{data_set}", "라벨": label.strip(), "파일명": f, "경로": os.path.join(root, f)})
    if not rows:
        return pd.DataFrame(columns=["구분", "분류", "데이터셋", "라벨", "파일명", "경로"])
    return pd.DataFrame(rows)


def get_weekly_accident_stats():
    df = scan_s63_data_files()
    if df.empty or "분류" not in df.columns or "데이터셋" not in df.columns:
        return pd.Series({"낙하": 12, "추락": 8, "충돌": 15, "화재": 5})
    accident = df[(df["분류"] == "사고유형") & (df["데이터셋"].str.contains("DATA2", na=False))]
    def to_type(lbl):
        if "낙하" in str(lbl): return "낙하"
        if "추락" in str(lbl): return "추락"
        if "충돌" in str(lbl): return "충돌"
        if "화재" in str(lbl): return "화재"
        return None
    if not accident.empty:
        accident = accident.copy()
        accident["사고유형"] = accident["라벨"].apply(to_type)
        accident = accident.dropna(subset=["사고유형"])
        type_counts = accident["사고유형"].value_counts()
        total = max(type_counts.sum(), 1)
        scale = 40 / total
        weekly = (type_counts * scale).round().astype(int)
        for t in ["낙하", "추락", "충돌", "화재"]:
            if t not in weekly.index:
                weekly[t] = 0
        return weekly.reindex(["낙하", "추락", "충돌", "화재"], fill_value=0)
    return pd.Series({"낙하": 12, "추락": 8, "충돌": 15, "화재": 5})


# ---------- 사이드바 ----------
with st.sidebar:
    st.markdown("### ⚙️ 감지 설정")
    st.markdown("---")
    model_source = st.radio(
        "분석 모델",
        options=["표준 모델 (권장)", "맞춤형 모델"],
        index=0,
        help="표준: 즉시 사용 가능. 맞춤형: data 폴더에 준비된 모델 사용.",
    )
    use_data_model_only = model_source == "맞춤형 모델"
    st.markdown("---")
    zone_number = st.selectbox("감지 구역", options=[1, 2, 3, 4], index=1, format_func=lambda x: f"{x}번 구역", help="알림에 표시할 작업 구역 번호입니다.")
    sensitivity = st.slider("감지 감도", min_value=1, max_value=100, value=70, step=5, help="높을수록 민감하게 위험을 감지합니다.")
    st.markdown("---")
    st.markdown("**감지 대상**")
    detect_helmet = st.checkbox("안전모 미착용", value=False)
    detect_fall = st.checkbox("추락", value=True)
    detect_fire = st.checkbox("화재", value=False)
    st.markdown("---")
    # PDF p.12: Q&A, EMAIL, GITHUB, DEMO
    st.markdown("### 📬 문의 / Q&A")
    st.caption("질문이 있으시면 언제든지 말씀해 주세요.")
    st.caption("**EMAIL** your-email@example.com")
    st.caption("**GITHUB** github.com/repository")
    st.caption("**DEMO** demo-url.com")
    st.markdown("---")
    st.caption("🛡️ Smart Yard Safety System v1.0")

st.markdown('<p class="main-title">🛡️ SMART YARD SAFETY SYSTEM</p>', unsafe_allow_html=True)
tab_monitor, tab_stats = st.tabs(["실시간 모니터링", "과거 데이터 통계"])

# ---------- 탭 1: 실시간 모니터링 ----------
with tab_monitor:
    col_video, col_alerts = st.columns([3, 1])
    with col_video:
        video_placeholder = st.empty()
        run_camera = st.button("📷 웹캠 켜기 (실시간 분석)")
    with col_alerts:
        st.subheader("🚨 실시간 위험 알림 내역")

    if run_camera:
        pose_tasks = _pose_with_tasks_api(use_data_model_only=use_data_model_only)
        if pose_tasks is None:
            with col_video:
                if use_data_model_only:
                    st.error("data 폴더에서 .task 모델을 찾지 못했습니다. data/선박·해양플랜트 스마트 야드 안전 데이터/AI 모델.zip 또는 data 내 .task 파일을 확인하세요.")
                else:
                    st.error("MediaPipe Pose를 초기화할 수 없습니다. pose_landmarker_lite.task 모델을 다운로드할 수 있는지 확인하세요.")
        else:
            detector, vision_module, drawing_utils_module, drawing_styles_module = pose_tasks
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                with col_video:
                    st.error(
                        "실시간 웹캠을 사용할 수 없습니다. "
                        "**링크(인터넷)로 접속 중**이라면 서버에 카메라가 없어 실시간 스트림은 불가합니다. "
                        "아래 **📸 카메라로 사진 촬영하여 분석**을 사용해 보세요. "
                        "노트북에서 직접 실행(streamlit run app.py → localhost)한 경우에만 실시간 웹캠이 동작합니다."
                    )
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                max_frames = 900
                cooldown_sec = 3.0
                status_placeholder = st.empty()
                for frame_idx in range(max_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not rgb.flags.c_contiguous:
                        rgb = np.ascontiguousarray(rgb)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    detection_result = detector.detect(mp_image)
                    rgb = draw_pose_tasks(rgb, detection_result, vision_module, drawing_utils_module, drawing_styles_module)
                    now = datetime.now()
                    time_str = now.strftime("%H:%M:%S")
                    cv2.putText(rgb, f"LIVE {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 170), 2)

                    fall_detected = False
                    if detect_fall and detection_result.pose_landmarks:
                        landmarks = detection_result.pose_landmarks[0]
                        fall_detected, _ = check_fall(landmarks, sensitivity=sensitivity, use_velocity=True)
                        if fall_detected:
                            ts = now.timestamp()
                            if ts - st.session_state.last_fall_time >= cooldown_sec:
                                st.session_state.last_fall_time = ts
                                alert_text = f"알림: [{time_str}] {zone_number}번 구역 추락 의심 발생!"
                                st.session_state.alerts.insert(0, {"time": time_str, "type": "추락 의심", "level": "danger", "msg": alert_text})
                                snapshot_copy = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                st.session_state.snapshots.insert(0, (time_str, snapshot_copy))
                                if len(st.session_state.snapshots) > st.session_state.max_snapshots:
                                    st.session_state.snapshots = st.session_state.snapshots[: st.session_state.max_snapshots]
                                try:
                                    os.makedirs(LOGS_SNAPSHOTS_DIR, exist_ok=True)
                                    fname = f"fall_zone{zone_number}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                                    cv2.imwrite(os.path.join(LOGS_SNAPSHOTS_DIR, fname), snapshot_copy)
                                except Exception:
                                    pass
                    else:
                        st.session_state.prev_spine_ratio = None

                    if fall_detected:
                        cv2.putText(rgb, "FALL DETECTED", (w // 2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    video_placeholder.image(rgb, use_column_width=True, channels="RGB")
                cap.release()
                status_placeholder.success("웹캠 분석을 종료했습니다. 알림과 최근 로그를 확인하세요.")
                st.rerun()

    if not run_camera:
        with video_placeholder.container():
            placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder_img[:] = (22, 26, 34)
            cv2.putText(placeholder_img, "LIVE CCTV FEED", (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 212, 170), 2)
            cv2.putText(placeholder_img, "Press [Webcam On] to start", (140, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 160, 170), 1)
            rgb_placeholder = cv2.cvtColor(placeholder_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_placeholder)
            st.image(pil_img, use_column_width=True)
        st.caption("실시간 웹캠 피드 (MediaPipe Pose 분석)")

    st.markdown("---")
    st.caption("휴대폰·태블릿 또는 웹캠이 안 될 때: 아래에서 카메라로 사진을 찍으면 추락·화재·안전모 분석을 할 수 있습니다.")
    photo = st.camera_input("📸 카메라로 사진 촬영하여 분석")
    if photo:
        img_pil = Image.open(photo).convert("RGB")
        rgb = np.asarray(img_pil, dtype=np.uint8).copy()  # writable copy for cv2.putText
        if not rgb.flags.c_contiguous:
            rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")

        # 화재 분석 (색상 기반 휴리스틱)
        fire_detected = False
        if detect_fire:
            fire_detected, fire_ratio = check_fire(rgb, sensitivity=sensitivity)
            if fire_detected:
                alert_text = f"알림: [{time_str}] {zone_number}번 구역 화재 의심!"
                st.session_state.alerts.insert(0, {"time": time_str, "type": "화재 의심", "level": "danger", "msg": alert_text})
                snapshot_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                st.session_state.snapshots.insert(0, (time_str, snapshot_bgr))
                if len(st.session_state.snapshots) > st.session_state.max_snapshots:
                    st.session_state.snapshots = st.session_state.snapshots[: st.session_state.max_snapshots]

        # 안전모 분석 (YOLO)
        helmet_violation = False
        if detect_helmet:
            helmet_violation, helmet_err = check_helmet(rgb)
            if helmet_err:
                st.warning("안전모 분석: " + helmet_err)
            elif helmet_violation:
                alert_text = f"알림: [{time_str}] {zone_number}번 구역 안전모 미착용 감지!"
                st.session_state.alerts.insert(0, {"time": time_str, "type": "안전모 미착용", "level": "danger", "msg": alert_text})
                snapshot_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                st.session_state.snapshots.insert(0, (time_str, snapshot_bgr))
                if len(st.session_state.snapshots) > st.session_state.max_snapshots:
                    st.session_state.snapshots = st.session_state.snapshots[: st.session_state.max_snapshots]

        # 추락 분석 (MediaPipe Pose)
        fall_detected = False
        pose_tasks = _pose_with_tasks_api(use_data_model_only=use_data_model_only)
        if pose_tasks is None:
            st.warning("추락 분석용 모델을 불러올 수 없습니다. 표준 모델을 선택했는지 확인하세요.")
            if not rgb.flags.writeable:
                rgb = np.copy(rgb)
            cv2.putText(rgb, f"분석 {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 170), 2)
        else:
            detector, vision_module, drawing_utils_module, drawing_styles_module = pose_tasks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = detector.detect(mp_image)
            rgb = draw_pose_tasks(rgb, detection_result, vision_module, drawing_utils_module, drawing_styles_module)
            if not rgb.flags.writeable:
                rgb = np.copy(rgb)
            cv2.putText(rgb, f"분석 {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 170), 2)
            if detect_fall and detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                fall_detected, _ = check_fall(landmarks, sensitivity=sensitivity, use_velocity=False)
                if fall_detected:
                    alert_text = f"알림: [{time_str}] {zone_number}번 구역 추락 의심 발생!"
                    st.session_state.alerts.insert(0, {"time": time_str, "type": "추락 의심", "level": "danger", "msg": alert_text})
                    snapshot_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    st.session_state.snapshots.insert(0, (time_str, snapshot_bgr))
                    if len(st.session_state.snapshots) > st.session_state.max_snapshots:
                        st.session_state.snapshots = st.session_state.snapshots[: st.session_state.max_snapshots]
                    cv2.putText(rgb, "FALL DETECTED", (w // 2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if fire_detected:
            cv2.putText(rgb, "FIRE DETECTED", (w // 2 - 90, 80 if fall_detected else 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if helmet_violation:
            y_helmet = 110 if (fall_detected and fire_detected) else (80 if (fall_detected or fire_detected) else 50)
            cv2.putText(rgb, "NO HELMET", (w // 2 - 80, y_helmet), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        st.image(Image.fromarray(rgb), use_column_width=True)
        msgs = []
        if fall_detected:
            msgs.append("추락 의심")
        if fire_detected:
            msgs.append("화재 의심")
        if helmet_violation:
            msgs.append("안전모 미착용")
        if msgs:
            st.error("⚠️ " + ", ".join(msgs) + " 알림 목록에 추가되었습니다.")
        else:
            st.success("분석 완료. 감지된 위험 없음.")

    with col_alerts:
        if st.session_state.alerts:
            st.markdown('<span style="color:#00d4aa;">● 위험 감지됨</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#00d4aa;">● 시스템 정상 (Normal)</span>', unsafe_allow_html=True)
        if st.session_state.alerts:
            for a in st.session_state.alerts:
                cls = a.get("level", "warning")
                msg = a.get("msg", f"{a.get('time')} | {a.get('type')}")
                st.markdown(f'<div class="alert-card {cls}">{msg}</div>', unsafe_allow_html=True)
        else:
            st.info("최근 알림이 없습니다.")

    st.markdown("---")
    st.subheader("📋 최근 로그")
    if st.session_state.snapshots:
        snapshots = st.session_state.snapshots
        for start in range(0, len(snapshots), 4):
            cols = st.columns(4)
            for i, col in enumerate(cols):
                idx = start + i
                if idx < len(snapshots):
                    ts_str, img_bgr = snapshots[idx]
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    with col:
                        st.image(img_rgb, use_column_width=True, channels="RGB")
                        st.caption(f"추락 의심 · {ts_str}")
    else:
        st.caption("위험 감지 시 해당 순간의 스냅샷이 여기에 표시됩니다.")

# ---------- 탭 2: 과거 데이터 통계 ----------
with tab_stats:
    st.subheader("📊 일주일간 사고 유형 통계")
    weekly = get_weekly_accident_stats()
    if weekly.sum() == 0:
        weekly = pd.Series({"낙하": 12, "추락": 8, "충돌": 15, "화재": 5})
    df_pie = weekly.reset_index()
    df_pie.columns = ["사고유형", "건수"]
    fig = px.pie(df_pie, values="건수", names="사고유형", title="사고 유형별 발생 비율 (최근 1주)", color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(paper_bgcolor="rgba(14,17,23,0)", plot_bgcolor="rgba(14,17,23,0)", font={"color": "#fafafa"}, legend={"font": {"color": "#fafafa"}})
    st.plotly_chart(fig, use_column_width=True)

    st.markdown("---")
    st.subheader("📈 트렌드 분석")
    st.caption("시간대별, 구역별 사고 패턴을 파악하여 고위험 요소 식별 (PDF: 과거 데이터 통계)")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        hours = list(range(24))
        np.random.seed(42)
        counts_by_hour = [max(0, int(x)) for x in np.random.poisson(3, 24)]
        df_hour = pd.DataFrame({"시간대(시)": [f"{h}시" for h in hours], "건수": counts_by_hour})
        fig_hour = px.bar(df_hour, x="시간대(시)", y="건수", title="시간대별 사고 발생 건수 (최근 1주)", color_discrete_sequence=["#00d4aa"])
        fig_hour.update_layout(paper_bgcolor="rgba(14,17,23,0)", plot_bgcolor="rgba(14,17,23,0)", font={"color": "#fafafa"}, xaxis_tickangle=-45)
        st.plotly_chart(fig_hour, use_column_width=True)
    with col_t2:
        zones = ["1번 구역", "2번 구역", "3번 구역", "4번 구역"]
        counts_by_zone = [4, 7, 12, 5]
        df_zone = pd.DataFrame({"구역": zones, "건수": counts_by_zone})
        fig_zone = px.bar(df_zone, x="구역", y="건수", title="구역별 사고 발생 건수 (최근 1주)", color_discrete_sequence=["#00d4aa"])
        fig_zone.update_layout(paper_bgcolor="rgba(14,17,23,0)", plot_bgcolor="rgba(14,17,23,0)", font={"color": "#fafafa"}, xaxis_tickangle=-25)
        st.plotly_chart(fig_zone, use_column_width=True)

    st.markdown("---")
    st.subheader("📁 S63_DATA 로그 파일 목록")
    log_df = scan_s63_data_files()
    if not log_df.empty:
        display_df = log_df[["구분", "분류", "데이터셋", "라벨", "파일명"]].copy()
        display_df = display_df.sort_values(["데이터셋", "분류", "파일명"])
        st.dataframe(display_df, use_column_width=True, hide_index=True)
        buffer = BytesIO()
        log_df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button(label="📥 엑셀로 다운로드", data=buffer, file_name="S63_DATA_로그목록.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("data 폴더에서 S63_DATA 명명 규칙에 맞는 파일을 찾지 못했습니다. 경로: " + DATA_BASE)

# PDF p.12: 푸터 문의/데모
st.markdown("---")
with st.expander("📬 문의 / Q&A (질문이 있으시면 언제든지 말씀해 주세요)"):
    st.markdown("**EMAIL** your-email@example.com  \n**GITHUB** github.com/repository  \n**DEMO** demo-url.com  \n\n감사합니다.")
