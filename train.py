import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.measure import regionprops
import os
import json
from pathlib import Path
import albumentations as A

class UnlabeledSignatureDetector:
    def _copy_training_files(self, annotation, output_path, split):
        """Copy image and label files to train/val folders for YOLO training"""
        import shutil
        img_src = annotation['image_path']
        img_name = Path(img_src).name
        # Copy image
        img_dst = output_path / 'images' / split / img_name
        try:
            shutil.copy(img_src, img_dst)
        except Exception as e:
            print(f"[Warning] Không thể copy ảnh: {img_src} -> {img_dst}: {e}")
        # Copy label (YOLO format)
        label_name = Path(img_name).stem + '.txt'
        label_src = output_path / 'labels' / label_name
        label_dst_dir = output_path / 'labels' / split
        label_dst_dir.mkdir(parents=True, exist_ok=True)
        label_dst = label_dst_dir / label_name
        if not label_src.exists():
            print(f"[Warning] Label file does not exist, skipping: {label_src}")
            return  
        try:
            shutil.copy(label_src, label_dst)
        except Exception as e:
            print(f"[Warning] Không thể copy label: {label_src} -> {label_dst}: {e}")
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.auto_labels = []
        
    def extract_candidate_regions(self, image_path, method='contour'):
        """Trích xuất các vùng ứng viên có thể chứa chữ ký"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Warning] Không thể load ảnh: {image_path}")
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        candidates = []
        if method == 'contour':
            candidates = self._extract_contour_candidates(gray, img)
        elif method == 'segmentation':
            candidates = self._extract_segmentation_candidates(gray, img)
        elif method == 'sliding_window':
            candidates = self._extract_sliding_window_candidates(gray, img)
        elif method == 'combined':
            # Kết hợp nhiều phương pháp
            c1 = self._extract_contour_candidates(gray, img)
            c2 = self._extract_segmentation_candidates(gray, img)
            candidates = c1 + c2
            candidates = self._remove_duplicate_regions(candidates)
        return candidates
    
    def _extract_contour_candidates(self, gray, original_img):
        """Trích xuất candidates dựa trên contour"""
        # Áp dụng nhiều threshold khác nhau
        candidates = []
        
        # Otsu threshold
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Mean threshold
        mean_val = np.mean(gray)
        _, thresh3 = cv2.threshold(gray, mean_val - 20, 255, cv2.THRESH_BINARY_INV)
        
        for thresh in [thresh1, thresh2, thresh3]:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Lọc vùng quá nhỏ
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Heuristic: chữ ký thường có aspect ratio 1.5-6
                    if 1.2 < aspect_ratio < 8 and w > 40 and h > 15:
                        # Trích xuất ROI
                        roi = original_img[y:y+h, x:x+w]
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'roi': roi,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'method': 'contour'
                        })
        
        return candidates
    
    def _extract_segmentation_candidates(self, gray, original_img):
        """Sử dụng image segmentation để tìm vùng ứng viên"""
        candidates = []
        # SLIC superpixels (fix: set channel_axis=None for grayscale)
        segments = slic(gray, n_segments=100, compactness=10, channel_axis=None)
        for region in regionprops(segments):
            if region.area > 200:
                minr, minc, maxr, maxc = region.bbox
                w, h = maxc - minc, maxr - minr
                aspect_ratio = w / h
                if 1.2 < aspect_ratio < 8 and w > 40 and h > 15:
                    roi = original_img[minr:maxr, minc:maxc]
                    candidates.append({
                        'bbox': (minc, minr, w, h),
                        'roi': roi,
                        'area': region.area,
                        'aspect_ratio': aspect_ratio,
                        'method': 'segmentation'
                    })
        return candidates
    
    def _extract_sliding_window_candidates(self, gray, original_img):
        """Sliding window để tìm vùng có mật độ pixel cao"""
        candidates = []
        h, w = gray.shape
        
        # Các kích thước window khác nhau
        window_sizes = [(80, 40), (120, 60), (160, 80), (200, 100)]
        step = 20
        
        for win_w, win_h in window_sizes:
            for y in range(0, h - win_h, step):
                for x in range(0, w - win_w, step):
                    window = gray[y:y+win_h, x:x+win_w]
                    
                    # Tính mật độ pixel tối (có thể là mực)
                    dark_ratio = np.sum(window < 128) / (win_w * win_h)
                    
                    # Tính variance để đo độ "thú vị" của vùng
                    variance = np.var(window)
                    
                    if dark_ratio > 0.1 and dark_ratio < 0.7 and variance > 500:
                        roi = original_img[y:y+win_h, x:x+win_w]
                        candidates.append({
                            'bbox': (x, y, win_w, win_h),
                            'roi': roi,
                            'dark_ratio': dark_ratio,
                            'variance': variance,
                            'method': 'sliding_window'
                        })
        
        return candidates
    
    def _remove_duplicate_regions(self, candidates, iou_threshold=0.5):
        """Loại bỏ các vùng trùng lặp"""
        if not candidates:
            return []
        
        # Sort by area (giữ vùng lớn hơn)
        candidates = sorted(candidates, key=lambda x: x.get('area', 0), reverse=True)
        
        keep = []
        for i, cand1 in enumerate(candidates):
            should_keep = True
            bbox1 = cand1['bbox']
            
            for cand2 in keep:
                bbox2 = cand2['bbox']
                if self._calculate_iou(bbox1, bbox2) > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(cand1)
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Tính Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / union_area
    
    def extract_features_from_candidates(self, candidates):
        """Trích xuất features từ các vùng ứng viên"""
        features = []
        
        for candidate in candidates:
            roi = candidate['roi']
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Resize về kích thước chuẩn
            resized = cv2.resize(gray_roi, (64, 32))
            
            feature_vector = []
            
            # 1. Đặc trưng hình học
            feature_vector.extend([
                candidate.get('area', 0),
                candidate.get('aspect_ratio', 0),
                candidate['bbox'][2],  # width
                candidate['bbox'][3],  # height
            ])
            
            # 2. Đặc trưng histogram
            hist = cv2.calcHist([resized], [0], None, [16], [0, 256])
            feature_vector.extend(hist.flatten())
            
            # 3. HOG features
            hog_features = hog(resized, orientations=8, pixels_per_cell=(8, 8),
                              cells_per_block=(1, 1), visualize=False)
            feature_vector.extend(hog_features)
            
            # 4. Texture features
            # LBP (Local Binary Pattern) simplified
            feature_vector.extend([
                np.std(resized),
                np.var(resized),
                np.mean(resized),
                np.sum(resized < 128) / resized.size,  # dark pixel ratio
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def cluster_candidates(self, features, method='kmeans', n_clusters=2):
        """Phân cụm các candidates thành chữ ký và không phải chữ ký"""
        if len(features) == 0:
            return []
        
        # Chuẩn hóa features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(features_scaled)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            labels = clusterer.fit_predict(features_scaled)
        
        return labels, scaler
    
    def auto_label_dataset(self, image_folder, output_folder, only_missing=False, annotation_file=None):
        """Tự động gán nhãn cho toàn bộ dataset hoặc chỉ ảnh thiếu annotation nếu only_missing=True"""
        image_folder = Path(image_folder)
        # Support more extensions
        image_paths = list({p.resolve() for p in image_folder.glob('*') if p.suffix.lower() in ['.jpg', '.png']})

        # If only_missing, load annotation file and filter
        if only_missing and annotation_file is not None:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            annotated = set(annotations.keys())
            image_paths = [p for p in image_paths if p.name not in annotated]

        all_candidates = []
        all_features = []
        image_candidate_map = {}

        print(f"Đang xử lý {len(image_paths)} ảnh...")

        # Thu thập tất cả candidates
        for i, img_path in enumerate(image_paths):
            print(f"Xử lý ảnh {i+1}/{len(image_paths)}: {img_path.name}")

            candidates = self.extract_candidate_regions(str(img_path), method='combined')
            features = self.extract_features_from_candidates(candidates)

            if len(candidates) > 0:
                start_idx = len(all_candidates)
                all_candidates.extend(candidates)
                all_features.extend(features)
                end_idx = len(all_candidates)

                image_candidate_map[str(img_path)] = (start_idx, end_idx)

        if len(all_features) == 0:
            print("Không tìm thấy candidates nào!")
            return

        # Phân cụm
        print("Đang phân cụm candidates...")
        labels, scaler = self.cluster_candidates(np.array(all_features))

        # Phân tích clusters để xác định cluster nào là chữ ký
        cluster_stats = {}
        for label in np.unique(labels):
            if label == -1:  # DBSCAN noise
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_candidates = [all_candidates[i] for i in cluster_indices]

            # Tính stats
            areas = [c.get('area', 0) for c in cluster_candidates]
            aspect_ratios = [c.get('aspect_ratio', 0) for c in cluster_candidates]

            cluster_stats[label] = {
                'count': len(cluster_candidates),
                'mean_area': np.mean(areas),
                'mean_aspect_ratio': np.mean(aspect_ratios),
                'candidates': cluster_candidates,
                'indices': cluster_indices
            }

        # Chọn cluster có khả năng là chữ ký cao nhất
        signature_cluster = self._select_signature_cluster(cluster_stats)

        # Tạo nhãn và lưu
        print(f"Cluster {signature_cluster} được chọn làm cluster chữ ký")
        self._save_auto_labels(image_candidate_map, labels, signature_cluster, 
                              all_candidates, output_folder)

        return labels, cluster_stats
    
    def _select_signature_cluster(self, cluster_stats):
        """Chọn cluster có khả năng chứa chữ ký cao nhất"""
        if not cluster_stats:
            return 0
        
        best_cluster = None
        best_score = -1
        
        for label, stats in cluster_stats.items():
            # Scoring heuristic
            score = 0
            
            # Aspect ratio gần với chữ ký thực tế (2-4)
            ideal_aspect = 3.0
            aspect_score = 1 / (1 + abs(stats['mean_aspect_ratio'] - ideal_aspect))
            score += aspect_score * 0.4
            
            # Area trong khoảng hợp lý
            if 500 < stats['mean_area'] < 5000:
                score += 0.3
            
            # Số lượng candidates (không quá ít, không quá nhiều)
            count_score = min(stats['count'] / 50, 1) * 0.3
            score += count_score
            
            if score > best_score:
                best_score = score
                best_cluster = label
        
        return best_cluster
    
    def _save_auto_labels(self, image_candidate_map, labels, signature_cluster, 
                         all_candidates, output_folder):
        """Lưu nhãn tự động"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        annotations = {}
        
        for img_path, (start_idx, end_idx) in image_candidate_map.items():
            img_name = Path(img_path).name
            annotations[img_name] = {
                'image_path': img_path,
                'signatures': [],
                'non_signatures': []
            }
            
            for i in range(start_idx, end_idx):
                candidate = all_candidates[i]
                bbox = candidate['bbox']
                
                annotation = {
                    'bbox': bbox,
                    'confidence': 0.8 if labels[i] == signature_cluster else 0.2
                }
                
                if labels[i] == signature_cluster:
                    annotations[img_name]['signatures'].append(annotation)
                else:
                    annotations[img_name]['non_signatures'].append(annotation)
        
        # Lưu annotations
        with open(output_path / 'auto_annotations.json', 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # Tạo YOLO format labels
        self._create_yolo_labels(annotations, output_path)
        
        print(f"Đã lưu nhãn tự động vào {output_folder}")
    
    def _create_yolo_labels(self, annotations, output_path):
        """Tạo labels theo format YOLO (always create a label file, even if empty)"""
        labels_dir = output_path / 'labels'
        labels_dir.mkdir(exist_ok=True)
        for img_name, data in annotations.items():
            label_name = Path(img_name).stem + '.txt'
            label_path = labels_dir / label_name
            with open(label_path, 'w') as f:
                # Class 0: signature
                if data['signatures']:
                    for sig in data['signatures']:
                        x, y, w, h = sig['bbox']
                        # Convert to YOLO format (normalized)
                        img = cv2.imread(data['image_path'])
                        img_h, img_w = img.shape[:2]
                        x_center = (x + w/2) / img_w
                        y_center = (y + h/2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                # If no signatures, file will be empty (YOLO expects this)
    
    def visualize_auto_labels(self, image_path, annotations_file):
        """Hiển thị kết quả auto-labeling"""
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        img_name = Path(image_path).name
        if img_name not in annotations:
            print(f"Không tìm thấy annotation cho {img_name}")
            return
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        data = annotations[img_name]
        
        # Vẽ signatures (màu xanh lá)
        for sig in data['signatures']:
            x, y, w, h = sig['bbox']
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"SIG {sig['confidence']:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Vẽ non-signatures (màu đỏ)
        for non_sig in data['non_signatures']:
            x, y, w, h = non_sig['bbox']
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img_rgb, f"NO {non_sig['confidence']:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(img_rgb)
        plt.title(f'Auto-labeled: {img_name}')
        plt.axis('off')
        plt.show()
    
    def create_training_dataset(self, annotations_file, output_dir, train_ratio=0.8):
        """Tạo dataset training từ auto-labels"""
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Tạo thư mục con
        (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # Chia train/val
        img_names = list(annotations.keys())
        np.random.shuffle(img_names)
        
        split_idx = int(len(img_names) * train_ratio)
        train_imgs = img_names[:split_idx]
        val_imgs = img_names[split_idx:]
        
        # Copy files
        for img_name in train_imgs:
            self._copy_training_files(annotations[img_name], output_path, 'train')
        
        for img_name in val_imgs:
            self._copy_training_files(annotations[img_name], output_path, 'val')
        
        # Tạo data.yaml cho YOLO
        yaml_content = f"""
train: {output_path}/images/train
val: {output_path}/images/val
nc: 1
names: ['signature']
"""
        with open(output_path / 'data.yaml', 'w') as f:
            f.write(yaml_content)
        
        print(f"Dataset đã được tạo tại {output_dir}")
        print(f"Train: {len(train_imgs)} ảnh, Val: {len(val_imgs)} ảnh")

def main():
    """Demo sử dụng"""
    detector = UnlabeledSignatureDetector()
    
    print("=== AUTO-LABELING SIGNATURE DETECTION ===")
    print("\n1. Trích xuất candidates từ ảnh:")
    print("candidates = detector.extract_candidate_regions('image.jpg', method='combined')")
    
    print("\n2. Auto-label toàn bộ dataset:")
    print("detector.auto_label_dataset('images_folder/', 'output_labels/')")
    
    print("\n3. Visualize kết quả:")
    print("detector.visualize_auto_labels('image.jpg', 'output_labels/auto_annotations.json')")
    
    print("\n4. Tạo training dataset:")
    print("detector.create_training_dataset('auto_annotations.json', 'training_dataset/')")
    
    print("\n=== WORKFLOW HOÀN CHỈNH ===")
    print("1. Chuẩn bị thư mục ảnh chưa có nhãn")
    print("2. Chạy auto-labeling")  
    print("3. Review và chỉnh sửa nhãn thủ công nếu cần")
    print("4. Train model YOLO/CNN với dataset đã được label")
    print("5. Fine-tune và deploy")

if __name__ == "__main__":
    main()
    # 1. Khởi tạo detector
    detector = UnlabeledSignatureDetector()

    # 2. Auto-label toàn bộ dataset
    labels, stats = detector.auto_label_dataset(
        image_folder='CR7/', 
        output_folder='auto_labels/'
    )

    # 3. Kiểm tra kết quả
    detector.visualize_auto_labels(
        './CR7/IMG_4846.png', 
        './auto_labels/auto_annotations.json'
    )

    # 4. Tạo training dataset
    detector.create_training_dataset(
        'auto_labels/auto_annotations.json',
        'training_dataset/'
    )