import cv2
import os
import yaml
import argparse
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge

# ==============================================================================
# 초기 실행 설정 (Initiation Settings)
#
# - Code Runner 또는 IDE에서 '직접 실행' 시 이 부분을 수정하여 사용하세요.
# - 터미널에서 인자를 직접 지정하면 이 설정은 무시됩니다.
# - 값을 None으로 두면 _config.yaml의 기본 설정을 따릅니다.
# ==============================================================================
INIT_BAG_DIR = None      # 예시: '/path/to/your/bag'
INIT_OUTPUT_DIR = None   # 예시: 'datasets/my_test_output'
INIT_MODE = None         # 추출 모드 (1: 단일, 2: 범위)
# ==============================================================================


def get_topic_type(bag_dir, topic_name):
    """
    ROS2 Bag의 metadata.yaml 파일에서 지정된 토픽의 메시지 타입을 찾아 반환합니다.
    """
    metadata_path = os.path.join(bag_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        print(f"오류: 메타데이터 파일을 찾을 수 없습니다. 경로: {metadata_path}")
        return None
        
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)['rosbag2_bagfile_information']
        for topic_info in metadata['topics_with_message_count']:
            if topic_info['topic_metadata']['name'] == topic_name:
                return topic_info['topic_metadata']['type']
    except Exception as e:
        print(f"메타데이터 파일 처리 중 오류 발생: {e}")
        
    return None

def extract_frames(config, args):
    """
    ROS2 Bag 파일에서 이미지를 추출하여 지정된 폴더에 저장합니다.
    """
    # 1. 설정값 결정 (3단계 우선순위 적용)
    # Priority 1: Command-line args > Priority 2: Initiation block > Priority 3: YAML config
    project_root = os.path.dirname(os.path.abspath(__file__))

    bag_dir = args.bag if args.bag is not None else \
              INIT_BAG_DIR if INIT_BAG_DIR is not None else \
              config['ros2_bag']['directory']

    output_dir_relative = args.output if args.output is not None else \
                          INIT_OUTPUT_DIR if INIT_OUTPUT_DIR is not None else \
                          config['datasets']['sample']
    output_dir = os.path.join(project_root, output_dir_relative)

    mode = args.mode if args.mode is not None else \
           INIT_MODE if INIT_MODE is not None else \
           '1' # mode의 최종 기본값은 '1'

    image_topic = config['ros2_bag']['image_topic']
    
    # --- 설정 확인 및 출력 ---
    print("\n" + "="*50)
    print("ROS2 Bag 이미지 추출을 시작합니다.")
    print("="*50)
    print(f"  - Bag 파일 경로: {bag_dir}")
    print(f"  - 출력 디렉토리: {output_dir}")
    print(f"  - 이미지 토픽: {image_topic}")
    print(f"  - 추출 모드: {'단일 프레임 저장' if mode == '1' else '범위 프레임 저장'}")
    print("="*50)

    # 2. 출력 폴더 생성 및 시작 인덱스 계산
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    image_formats = config['image_format'].split(',')
    existing_files = os.listdir(images_dir)
    if not existing_files:
        start_index = 0
    else:
        indices = [int(os.path.splitext(f)[0]) for f in existing_files if f.split('.')[-1] in image_formats and f.split('.')[0].isdigit()]
        start_index = max(indices) + 1 if indices else 0

    # 3. ROS Bag 리더 설정
    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"오류: ROS Bag을 여는 데 실패했습니다: {e}")
        print("경로, 파일 권한, 파일 손상 여부를 확인하세요.")
        return

    topic_type_str = get_topic_type(bag_dir, image_topic)
    if not topic_type_str:
        print(f"오류: 토픽 '{image_topic}'을 Bag 파일 내에서 찾을 수 없습니다.")
        return
        
    msg_type = get_message(topic_type_str)
    bridge = CvBridge()
    
    # 4. 상태 변수 및 콜백 함수 정의
    is_paused = True
    is_saving = False
    save_single_frame_flag = False
    saved_count = 0
    cv_image = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal save_single_frame_flag, is_saving
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == '1':
                save_single_frame_flag = True
                print(f"-> 단일 프레임 저장 명령 수신!")
            elif mode == '2':
                is_saving = not is_saving
                status_text = "시작" if is_saving else "중지"
                print(f"-> 범위 저장을 {status_text}합니다.")

    base_window_name = "ROS2 Bag Player"
    cv2.namedWindow(base_window_name)
    cv2.setMouseCallback(base_window_name, mouse_callback)

    # 5. 메인 루프
    print("\n조작법: [Space]: 재생/일시정지 | [마우스 좌클릭]: 저장 액션 | [Q]: 종료")
    while reader.has_next():
        if not is_paused or cv_image is None:
            try:
                (topic, data, t) = reader.read_next()
                if topic == image_topic:
                    msg = deserialize_message(data, msg_type)
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                print(f"\nBag 파일 마지막에 도달했거나 읽기 오류가 발생했습니다: {e}")
                break

        if cv_image is None: continue

        display_image = cv_image.copy()
        
        status_info = ""
        if is_paused: status_info += "[PAUSED] "
        if mode == '2' and is_saving: status_info += "[RECORDING...]"
        cv2.putText(display_image, status_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(base_window_name, display_image)
        
        should_save = (mode == '1' and save_single_frame_flag) or \
                      (mode == '2' and is_saving and not is_paused)
        
        if should_save:
            filename = f"{start_index:06d}.{image_formats[0]}"
            filepath = os.path.join(images_dir, filename)
            cv2.imwrite(filepath, cv_image)
            
            if mode == '1':
                print(f"  - 저장 완료: {filename}")
                save_single_frame_flag = False
            elif saved_count % 30 == 0:
                 print(f"  - 저장 중... ({filename})")
            
            saved_count += 1
            start_index += 1

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_paused = not is_paused

    cv2.destroyAllWindows()
    print(f"\n이미지 추출을 종료합니다. 총 {saved_count}개의 이미지를 저장했습니다.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요.")
        exit()

    parser = argparse.ArgumentParser(description="ROS2 Bag 파일에서 이미지를 추출합니다.")
    parser.add_argument('--bag', type=str, default=None, help="ROS2 Bag 디렉토리의 절대 경로.")
    parser.add_argument('--output', type=str, default=None, help="이미지를 저장할 출력 디렉토리의 상대 경로.")
    parser.add_argument('--mode', type=str, default=None, choices=['1', '2'], help="추출 모드 (1: 단일, 2: 범위).")
    args = parser.parse_args()
    
    extract_frames(config, args)