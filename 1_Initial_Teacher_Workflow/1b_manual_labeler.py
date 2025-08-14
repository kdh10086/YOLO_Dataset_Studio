import cv2
import os
import glob
import yaml
import argparse
import numpy as np
from collections import Counter

# ==============================================================================
# 초기 실행 설정 (Initiation Settings)
#
# - Code Runner 또는 IDE에서 '직접 실행' 시 이 부분을 수정하여 사용하세요.
# - 터미널에서 인자를 직접 지정하면 이 설정은 무시됩니다.
# - 값을 None으로 두면 _config.yaml의 기본 설정을 따릅니다.
# ==============================================================================
INIT_DATASET_DIR = None   # 예시: 'datasets/my_labeling_data'
INIT_IMAGE_NAME = None    # 예시: '000123.png'
# ==============================================================================

# --- 전역 변수 및 상수 ---
# 이 변수들은 마우스 콜백 함수와 메인 루프 간의 상태 공유를 위해 사용됩니다.
ref_point = []
current_bboxes = []
history = []
previous_image_new_bboxes = []
drawing = False
deletion_mode = False
current_class_id = 0
clone, img_orig, display_img = None, None, None
h_orig, w_orig, ratio = 0, 0, 1.0
CLASSES = {}
COLORS = {}

# 디스플레이 관련 상수
DISPLAY_WIDTH = 1280
MAGNIFIED_SIZE = 480
MAGNIFIED_REGION = 100

# --- 유틸리티 함수 ---
def pixels_to_yolo(pixel_bbox, img_width, img_height):
    """픽셀 좌표 바운딩 박스를 YOLO 형식으로 변환합니다."""
    class_id, x1, y1, x2, y2 = pixel_bbox
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return class_id, x_center / img_width, y_center / img_height, width / img_width, height / img_height

def draw_dotted_rectangle(img, pt1, pt2, color, thickness, gap=10):
    """삭제 모드에서 사용할 점선 사각형을 그립니다."""
    start_point = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]))
    end_point = (max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))
    for i in range(start_point[0], end_point[0], gap * 2):
        cv2.line(img, (i, start_point[1]), (min(i + gap, end_point[0]), start_point[1]), color, thickness)
        cv2.line(img, (i, end_point[1]), (min(i + gap, end_point[0]), end_point[1]), color, thickness)
    for i in range(start_point[1], end_point[1], gap * 2):
        cv2.line(img, (start_point[0], i), (start_point[0], min(i + gap, end_point[1])), color, thickness)
        cv2.line(img, (end_point[0], i), (end_point[0], min(i + gap, end_point[1])), color, thickness)

def print_controls():
    """사용자에게 조작 키를 안내합니다."""
    print("\n" + "="*40)
    print(" " * 12 + "라벨링 도구 조작법")
    print("="*40)
    print("  - [w]: 그리기 모드 (기본)")
    print("  - [e]: 영역 삭제 모드")
    print("  - [1-9]: 클래스 선택")
    print("  - [s]: 저장 후 다음 이미지로 이동")
    print("  - [a]: 저장 후 이전 이미지로 이동")
    print("  - [v]: 이전 이미지에서 새로 추가한 박스 붙여넣기")
    print("  - [r]: 마지막 작업 취소 (Undo)")
    print("  - [q]: 저장 후 프로그램 종료")
    print("  - [c]: 저장하지 않고 강제 종료")
    print("="*40)

def redraw_boxes(image):
    """현재까지 그린 모든 바운딩 박스를 화면에 다시 그립니다."""
    for bbox in current_bboxes:
        class_id, x1, y1, x2, y2 = bbox
        color = COLORS.get(class_id, (255, 255, 255))
        disp_x1, disp_y1 = int(x1 * ratio), int(y1 * ratio)
        disp_x2, disp_y2 = int(x2 * ratio), int(y2 * ratio)
        cv2.rectangle(image, (disp_x1, disp_y1), (disp_x2, disp_y2), color, 2)
        label = f"{CLASSES.get(class_id, 'Unknown')}"
        cv2.putText(image, label, (disp_x1, disp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def click_and_draw_box(event, x, y, flags, param):
    """마우스 이벤트 처리를 위한 콜백 함수."""
    global ref_point, current_bboxes, drawing, clone, deletion_mode, history

    orig_x, orig_y = int(x / ratio), int(y / ratio)

    if deletion_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if not drawing:
                ref_point = [(orig_x, orig_y)]
                drawing = True
            else:
                drawing = False
                ref_point.append((orig_x, orig_y))
                history.append(current_bboxes.copy())
                del_x1, del_y1 = min(ref_point[0][0], ref_point[1][0]), min(ref_point[0][1], ref_point[1][1])
                del_x2, del_y2 = max(ref_point[0][0], ref_point[1][0]), max(ref_point[0][1], ref_point[1][1])
                
                bboxes_to_keep = []
                for bbox in current_bboxes:
                    # 박스의 중심점이 삭제 영역 내에 있는지 확인
                    bbox_center_x = (bbox[1] + bbox[3]) / 2
                    bbox_center_y = (bbox[2] + bbox[4]) / 2
                    if not (del_x1 < bbox_center_x < del_x2 and del_y1 < bbox_center_y < del_y2):
                        bboxes_to_keep.append(bbox)
                        
                removed_count = len(current_bboxes) - len(bboxes_to_keep)
                if removed_count > 0:
                    print(f"-> {removed_count}개의 바운딩 박스를 제거했습니다. ('r'로 취소 가능)")
                    current_bboxes = bboxes_to_keep
                ref_point = []
                clone = display_img.copy()
                redraw_boxes(clone)
    else: # 그리기 모드
        if event == cv2.EVENT_LBUTTONDOWN:
            if not drawing:
                ref_point = [(orig_x, orig_y)]
                drawing = True
            else:
                history.append(current_bboxes.copy())
                ref_point.append((orig_x, orig_y))
                drawing = False
                x1, y1 = ref_point[0]
                x2, y2 = ref_point[1]
                start = (min(x1, x2), min(y1, y2))
                end = (max(x1, x2), max(y1, y2))
                current_bboxes.append((current_class_id, start[0], start[1], end[0], end[1]))
                ref_point = []
                clone = display_img.copy()
                redraw_boxes(clone)

    if event == cv2.EVENT_MOUSEMOVE:
        clone = display_img.copy()
        redraw_boxes(clone)
        h_disp, w_disp = clone.shape[:2]

        if drawing and ref_point:
            start_point_display = (int(ref_point[0][0] * ratio), int(ref_point[0][1] * ratio))
            if deletion_mode:
                draw_dotted_rectangle(clone, start_point_display, (x, y), (0, 0, 255), 2)
            else:
                color = COLORS.get(current_class_id, (0, 255, 255))
                label = CLASSES.get(current_class_id, "Unknown")
                cv2.rectangle(clone, start_point_display, (x, y), color, 2)
                cv2.putText(clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 십자선 및 돋보기 뷰
        crosshair_color = (0, 0, 255) if deletion_mode else COLORS.get(current_class_id, (0, 255, 255))
        cv2.line(clone, (0, y), (w_disp, y), crosshair_color, 1)
        cv2.line(clone, (x, 0), (x, h_disp), crosshair_color, 1)
        
        x_start = max(0, x - MAGNIFIED_REGION)
        y_start = max(0, y - MAGNIFIED_REGION)
        x_end = min(w_disp, x + MAGNIFIED_REGION)
        y_end = min(h_disp, y + MAGNIFIED_REGION)
        
        magnified_region = clone[y_start:y_end, x_start:x_end]
        if magnified_region.size > 0:
            resized_magnified = cv2.resize(magnified_region, (MAGNIFIED_SIZE, MAGNIFIED_SIZE), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Magnified View", resized_magnified)

def save_labels(label_path, image_name, bboxes, w, h):
    """YOLO 형식으로 라벨을 파일에 저장합니다."""
    label_dir = os.path.dirname(label_path)
    os.makedirs(label_dir, exist_ok=True)
    
    with open(label_path, 'w') as f:
        if not bboxes:
            print(f"  -> 저장: '{image_name}'에 라벨링된 객체가 없어 빈 파일로 저장합니다.")
            return
        for bbox in bboxes:
            yolo_bbox = pixels_to_yolo(bbox, w, h)
            f.write(f"{yolo_bbox[0]} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f} {yolo_bbox[4]:.6f}\n")
            
    class_ids = [bbox[0] for bbox in bboxes]
    count_summary = Counter(class_ids)
    summary_str = ", ".join([f"{CLASSES.get(cid, 'N/A')}: {count}개" for cid, count in sorted(count_summary.items())])
    print(f"  -> 저장 완료: '{os.path.basename(label_path)}' ({summary_str})")

def process_newly_added_boxes(initial_bboxes, final_bboxes):
    """현재 이미지에서 새로 추가된 박스를 다음 이미지에서 붙여넣기 위해 저장합니다."""
    global previous_image_new_bboxes
    initial_set = set(initial_bboxes)
    final_set = set(final_bboxes)
    newly_added = list(final_set - initial_set)
    if newly_added:
        previous_image_new_bboxes = newly_added
        print(f"  -> 다음 이미지용으로 {len(previous_image_new_bboxes)}개의 신규 박스를 복사했습니다. ('v' 키로 붙여넣기)")
    else:
        previous_image_new_bboxes = []

def main(config, args):
    """라벨링 도구의 메인 로직을 실행합니다."""
    global ref_point, current_bboxes, current_class_id, drawing, deletion_mode, history, clone, img_orig, display_img, h_orig, w_orig, ratio, CLASSES, COLORS
    
    CLASSES = config['classes']
    COLORS = {cid: ((cid * 55 + 50) % 256, (cid * 95 + 100) % 256, (cid * 135 + 150) % 256) for cid in CLASSES.keys()}
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_dir_relative = args.dataset if args.dataset is not None else INIT_DATASET_DIR if INIT_DATASET_DIR is not None else config['datasets']['sample']
    dataset_dir = os.path.join(project_root, dataset_dir_relative)

    start_from_image = args.start_image if args.start_image is not None else INIT_IMAGE_NAME

    print("\n" + "="*50); print("라벨링 도구를 시작합니다."); print("="*50)
    print(f"  - 대상 데이터셋: {dataset_dir}")
    if start_from_image: print(f"  - 시작 이미지: {start_from_image}")
    print("="*50)

    images_base_dir = os.path.join(dataset_dir, 'images')
    labels_base_dir = os.path.join(dataset_dir, 'labels')
    os.makedirs(labels_base_dir, exist_ok=True)
    
    image_formats = config['image_format'].split(',')
    all_image_paths = []
    for ext in image_formats: all_image_paths.extend(glob.glob(os.path.join(images_base_dir, f'*.{ext}')))
    
    if not all_image_paths:
        print(f"오류: '{images_base_dir}'에서 이미지를 찾을 수 없습니다."); return
        
    master_image_paths = sorted(all_image_paths)
    print(f"총 {len(master_image_paths)}개의 이미지를 찾았습니다.")
    
    print_controls()
    start_mode = input("시작 모드를 선택하세요 (1: 라벨 없는 이미지부터, 2: 특정 이미지부터, 3: 검토 목록(review_list.txt)으로): ")

    img_index, image_paths, is_review_mode = 0, master_image_paths.copy(), False
    
    if start_mode == '3':
        is_review_mode = True
        review_list_path = os.path.join(dataset_dir, 'review_list.txt')
        if not os.path.exists(review_list_path):
            print(f"오류: '{review_list_path}' 파일이 없어 검토 모드를 시작할 수 없습니다."); return
        
        with open(review_list_path, 'r') as f: review_files = {line.strip() for line in f if line.strip()}
        
        image_paths = [p for p in master_image_paths if os.path.basename(p) in review_files]
        if not image_paths: print("검토 목록에 해당하는 이미지가 데이터셋에 없습니다."); return
        print(f"\n검토 모드: {len(image_paths)}개의 이미지를 검토합니다.")

    elif start_mode == '2':
        start_img_name_input = start_from_image if start_from_image else input("시작할 이미지 이름(확장자 포함)을 입력하세요: ")
        try:
            img_index = [os.path.basename(p) for p in image_paths].index(start_img_name_input)
            print(f"\n'{start_img_name_input}'부터 라벨링을 시작합니다.")
        except ValueError:
            print(f"오류: '{start_img_name_input}' 이미지를 찾을 수 없습니다."); return
    else:
        print("\n일반 모드: 라벨이 없는 첫 이미지부터 시작합니다.")
        for i, path in enumerate(image_paths):
            label_path = os.path.splitext(path.replace('images', 'labels', 1))[0] + '.txt'
            if not os.path.exists(label_path):
                img_index = i; break
        else:
            print("모든 이미지에 라벨 파일이 존재합니다. 첫 이미지부터 시작합니다."); img_index = 0

    total_images = len(image_paths)
    if total_images == 0:
        print("작업할 이미지가 없습니다."); return
        
    cv2.namedWindow("Labeling Tool")
    cv2.namedWindow("Magnified View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Magnified View", MAGNIFIED_SIZE, MAGNIFIED_SIZE)

    quit_flag = False
    while 0 <= img_index < total_images and not quit_flag:
        image_path = image_paths[img_index]
        image_name = os.path.basename(image_path)
        label_path = os.path.splitext(image_path.replace('images', 'labels', 1))[0] + '.txt'
        
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            print(f"경고: '{image_name}' 파일을 불러올 수 없습니다. 건너뜁니다."); img_index += 1; continue
            
        h_orig, w_orig = img_orig.shape[:2]
        ratio = DISPLAY_WIDTH / w_orig
        display_img = cv2.resize(img_orig, (DISPLAY_WIDTH, int(h_orig * ratio)))
            
        clone = display_img.copy()
        current_bboxes, ref_point, drawing, history = [], [], False, []

        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cid, x_c, y_c, bw, bh = int(parts[0]), *map(float, parts[1:])
                        x1, y1 = int((x_c - bw / 2) * w_orig), int((y_c - bh / 2) * h_orig)
                        x2, y2 = int((x_c + bw / 2) * w_orig), int((y_c + bh / 2) * h_orig)
                        current_bboxes.append((cid, x1, y1, x2, y2))
            except Exception as e:
                print(f"경고: '{os.path.basename(label_path)}' 파일 읽기 오류: {e}")
        
        initial_bboxes_for_image = current_bboxes.copy()
        redraw_boxes(clone)
        
        cv2.setMouseCallback("Labeling Tool", click_and_draw_box)
        
        while True:
            mode_text = "영역 삭제" if deletion_mode else "박스 그리기"
            review_text = "[검토 모드] " if is_review_mode else ""
            title = f"{review_text}[{img_index+1}/{total_images}] {image_name} | Class: {CLASSES.get(current_class_id,'N/A')} | Mode: {mode_text}"
            cv2.setWindowTitle("Labeling Tool", title)
            cv2.imshow("Labeling Tool", clone)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                process_newly_added_boxes(initial_bboxes_for_image, current_bboxes)
                save_labels(label_path, image_name, current_bboxes, w_orig, h_orig)
                if is_review_mode and 'review_files' in locals() and image_name in review_files:
                    review_files.remove(image_name)
                quit_flag = True; break
            elif key == ord('s'):
                process_newly_added_boxes(initial_bboxes_for_image, current_bboxes)
                save_labels(label_path, image_name, current_bboxes, w_orig, h_orig)
                if is_review_mode and 'review_files' in locals() and image_name in review_files:
                    review_files.remove(image_name)
                    print(f"-> '{image_name}' 검토 완료. 남은 검토 파일: {len(review_files)}개")
                img_index += 1; break
            elif key == ord('c'):
                print("현재 작업을 저장하지 않고 종료합니다."); quit_flag = True; break
            elif key == ord('r'):
                if drawing:
                    drawing = False; ref_point = []
                    print("-> 박스 그리기를 취소했습니다.")
                elif history:
                    current_bboxes = history.pop(); print("-> 마지막 작업을 취소했습니다.")
                else:
                    print("-> 더 이상 취소할 작업이 없습니다.")
                clone = display_img.copy(); redraw_boxes(clone)
            elif key == ord('a'):
                if img_index > 0:
                    process_newly_added_boxes(initial_bboxes_for_image, current_bboxes)
                    save_labels(label_path, image_name, current_bboxes, w_orig, h_orig)
                    print("-> 이전 이미지로 돌아갑니다."); img_index -= 1; break
                else:
                    print("-> 첫 번째 이미지입니다. 더 이상 뒤로 갈 수 없습니다.")
            elif key == ord('v'):
                if previous_image_new_bboxes:
                    history.append(current_bboxes.copy()); current_bboxes.extend(previous_image_new_bboxes)
                    print(f"-> {len(previous_image_new_bboxes)}개의 박스를 붙여넣었습니다."); clone = display_img.copy(); redraw_boxes(clone)
                else:
                    print("-> 이전 이미지에서 복사할 신규 박스가 없습니다.")
            elif ord('1') <= key <= ord('9'):
                new_class_id = int(chr(key)) - 1
                if new_class_id in CLASSES:
                    current_class_id = new_class_id
                    print(f"-> 클래스 변경: {CLASSES[current_class_id]} ({current_class_id})")
            elif key == ord('e'):
                if not deletion_mode:
                    deletion_mode = True; drawing = False; ref_point = []
                    print("\n>> 영역 삭제 모드로 전환 <<")
            elif key == ord('w'):
                 if deletion_mode:
                    deletion_mode = False; drawing = False; ref_point = []
                    print("\n>> 박스 그리기 모드로 전환 <<")
    
    if is_review_mode and 'review_files' in locals():
        review_list_path = os.path.join(dataset_dir, 'review_list.txt')
        if not review_files:
            if os.path.exists(review_list_path):
                os.remove(review_list_path)
                print(f"\n모든 검토를 완료하여 '{review_list_path}' 파일을 삭제했습니다.")
        else:
            with open(review_list_path, 'w') as f: f.write('\n'.join(sorted(list(review_files))))
            print(f"\n진행상황 저장 완료. 남은 검토 목록({len(review_files)}개)이 업데이트되었습니다.")

    cv2.destroyAllWindows()
    print("\n라벨링 도구를 종료합니다.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요.")
        exit()

    parser = argparse.ArgumentParser(description="YOLO 형식의 데이터셋을 수동으로 라벨링하는 도구입니다.")
    parser.add_argument('--dataset', type=str, default=None, help="라벨링할 데이터셋의 상대 경로. (예: 'datasets/sample_dataset')")
    parser.add_argument('--start_image', type=str, default=None, help="시작할 이미지 파일 이름. (예: '000123.png')")
    args = parser.parse_args()
    
    main(config, args)