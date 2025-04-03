import cv2
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt



def preprocess_image(image, use_clahe, clip_limit, tile_grid_size, blur_ksize, sigma):
    """Predspracovanie obrazu: ekvalizácia, redukcia šumu, detekcia hrán."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        gray = clahe.apply(gray)

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), sigma)
    edges = cv2.Canny(blurred, 50, 150)
    return edges, blurred


def detect_circles(image, dp, min_dist, param1, param2, min_radius, max_radius):
    """Detekcia kružníc pomocou Houghovej transformácie."""
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    return circles
def detect_pupil(image):
    """Detekcia zreničky"""
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 80, param1=200, param2=45, minRadius=20, maxRadius=80)
    return filter_pupil(circles, image.shape, 50)

def detect_iris(image, pupil):
    """Detekcia dúhovky"""
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 110, param1=10, param2=40, minRadius=70, maxRadius=106)
    return filter_iris(circles, pupil, image.shape)


def detect_lids(image, iris):
    """Detekcia horného a spodného viečka"""
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 200, param1=30, param2=4, minRadius=300, maxRadius=500)
    top_lid = filter_top_lid(circles, iris, image.shape)
    bottom_lid = filter_bottom_lid(circles, iris, image.shape)
    return top_lid, bottom_lid

def filter_pupil(circles, img_shape, tolerance=50):
    """Filtrovanie zreničky - musí byť približne v strede obrázka a menšia ako dúhovka."""
    if circles is None or len(circles) == 0:
        return None

    center_x, center_y = img_shape[1] // 2, img_shape[0] // 2
    circles = np.int32(np.around(circles[0, :]))  # Zabezpečí správny dátový typ

    # Zrenička je najmenšia kružnica a blízko stredu
    pupil_candidates = [c for c in circles if np.abs(c[0] - center_x) < tolerance and np.abs(c[1] - center_y) < tolerance]

    if not pupil_candidates:
        return None

    pupil = min(pupil_candidates, key=lambda c: c[2])  # Najmenší polomer

    return np.array([[pupil[0], pupil[1], pupil[2]]])


def filter_iris(circles, pupil, img_shape, tolerance=80):
    """Filtrovanie dúhovky - musí byť väčšia ako zrenička a obklopovať ju."""
    if circles is None or pupil is None:
        return None

    center_x, center_y = img_shape[1] // 2, img_shape[0] // 2
    pupil_x, pupil_y, pupil_r = pupil[0]

    # Nepoužívame uint16, aby sme nestratili presnosť
    circles = np.around(circles[0, :]).astype(int)

    # Zvoľme kruhy, ktoré sú väčšie ako zrenička a relatívne blízko jej stredu
    filtered = [c for c in circles if c[2] > pupil_r and abs(c[0] - pupil_x) < tolerance and abs(c[1] - pupil_y) < tolerance]

    if not filtered:
        return None

    # Vyberieme kruh, ktorý je najbližší k zreničke veľkosťou a stredom
    filtered = sorted(filtered, key=lambda c: np.hypot(c[0] - pupil_x, c[1] - pupil_y) + abs(c[2] - pupil_r))

    return np.array([filtered[0]])


def filter_bottom_lid(circles, iris, img_shape):
    """Filtrovanie spodného viečka - kružnica pod zreničkou."""
    if circles is None or iris is None:
        return None

    # Konvertovanie na celocíselné hodnoty
    circles = np.uint16(np.around(circles[0, :]))

    iris_x, iris_y, iris_r = iris[0]

    # Stredová os (x) obrázku
    center_x = img_shape[1] // 2

    # Vybrať kruh pod dúhovkou, ktorý je najbližšie k stredovej osi
    filtered = [c for c in circles if c[1] > iris_y + iris_r // 2 and abs(c[0] - center_x) < 50]

    if not filtered:
        return None

    return np.array([filtered[0]])


def filter_top_lid(circles, iris, img_shape):
    """Filtrovanie horného viečka - kružnica nad zreničkou."""
    if circles is None or iris is None:
        return None

    # Konvertovanie na celocíselné hodnoty
    circles = np.uint16(np.around(circles[0, :]))

    iris_x, iris_y, iris_r = iris[0]

    # Stredová os (y) obrázku
    center_x = img_shape[1] // 2

    # Vybrať kruh, ktorý je nad dúhovkou a najbližšie k stredovej osi
    filtered = [c for c in circles if c[2] > iris_r * 1.5 and abs(c[0] - center_x) < 50]

    if not filtered:
        return None

    return np.array([filtered[0]])


def draw_circles(image, circles, R=0, G=255, B=0):
    """Vykreslenie detegovaných kružníc na obrázok."""
    output = image.copy()

    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles))  # Konverzia na uint16

        # Overenie, že circles má správny tvar
        if circles.ndim == 2 and circles.shape[1] == 3:
            pass  # Tvar je už správny (N, 3)
        elif circles.ndim == 3 and circles.shape[0] == 1:
            circles = circles[0]  # Zbaviť sa zbytočnej dimenzie (1, N, 3) → (N, 3)

        for i, circle in enumerate(circles):
            circle = tuple(circle)  # Konverzia na tuple (x, y, r)

            x, y, r = circle
            color = (R, G, B) if i == 0 else (255, 0, 0)
            cv2.circle(output, (x, y), r, color, 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

    else:
        print("No circles detected")

    return output

def update(_):
    """Callback funkcia na aktualizáciu zobrazenia podľa hodnôt sliderov."""
    padding = cv2.getTrackbarPos("Padding", "Settings")
    blur_ksize = max(cv2.getTrackbarPos("Gaussian Blur", "Settings"),1)
    dp = max(cv2.getTrackbarPos("DP", "Settings"),1) / 10.0
    min_dist = max(cv2.getTrackbarPos("MinDist", "Settings"),1)
    param1 = max(cv2.getTrackbarPos("Param1", "Settings"),1)
    param2 = max(cv2.getTrackbarPos("Param2", "Settings"),1)
    min_radius = max(cv2.getTrackbarPos("MinRadius", "Settings"),1)
    max_radius = max(cv2.getTrackbarPos("MaxRadius", "Settings"),1)

    use_clahe = cv2.getTrackbarPos("CLAHE", "Settings")
    use_hough = cv2.getTrackbarPos("Hough Transform", "Settings")

    image_padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    preprocessed, blurred_img = preprocess_image(image_padded, use_clahe, 2.0, (8, 8), blur_ksize, 1.5)

    if use_hough:
        circles = detect_circles(preprocessed, dp, min_dist, param1, param2, min_radius, max_radius)
        if circles is not None and len(circles) > 0:
            result = draw_circles(image_padded, circles)
        else:
            result = image_padded.copy()
    else:
        result = image_padded.copy()

    cv2.imshow("Result", result)
    cv2.imshow("Preprocessed", preprocessed)
    cv2.imshow("Blurred", blurred_img)

def setup_trackbars():
    """Vytvorenie GUI sliderov."""
    cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Settings", 500, 400)  # Zvýšenie veľkosti okna
    cv2.createTrackbar("Padding", "Settings", 0, 300, update)
    cv2.createTrackbar("Gaussian Blur", "Settings", 1, 50, update)
    cv2.createTrackbar("DP", "Settings", 12, 30, update)  # DP / 10
    cv2.createTrackbar("MinDist", "Settings", 50, 200, update)
    cv2.createTrackbar("Param1", "Settings", 100, 255, update)
    cv2.createTrackbar("Param2", "Settings", 30, 255, update)
    cv2.createTrackbar("MinRadius", "Settings", 20, 500, update)
    cv2.createTrackbar("MaxRadius", "Settings", 100, 500, update)
    cv2.createTrackbar("CLAHE", "Settings", 0, 1, update)  # Checkbox
    cv2.createTrackbar("Hough Transform", "Settings", 0, 1, update)  # Checkbox

def four_circles(image):
    """Detekcia štyroch požadovaných kužníc."""
    image_padded = cv2.copyMakeBorder(image, 110, 110, 110, 110, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    preprocessed_pupil, _ = preprocess_image(image_padded, False, 2.0, (8, 8), 15, 1.5)
    preprocessed_iris, _ = preprocess_image(image_padded, True, 2.0, (8, 8), 1, 1.5)
    preprocessed_lids, _ = preprocess_image(image_padded, True, 2.0, (8, 8), 5, 1.5)
    pupil_circles = detect_pupil(preprocessed_pupil)
    iris_circles = detect_iris(preprocessed_iris, pupil_circles)
    top_lid, bottom_lid = detect_lids(preprocessed_lids, pupil_circles)
    result = draw_circles(image_padded, pupil_circles, R=255, G=255, B=0)
    result = draw_circles(result, iris_circles, R=0, G=255, B=0)
    result = draw_circles(result, top_lid, R=0, G=255, B=255)
    result = draw_circles(result, bottom_lid, R=255, G=0, B=255)
    return result, pupil_circles, iris_circles, top_lid, bottom_lid

def remove_padding(detected_circles, pad_x=110, pad_y=100):
    """Odstráni padding zo súradníc kruhov."""
    adjusted_circles = [[x - pad_x, y - pad_y, r] for x, y, r in detected_circles]
    return adjusted_circles

def compute_iou(circle1, circle2):
    """Vypočíta Intersection-over-Union (IoU) pre dva kruhy."""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # Vzdialenosť medzi stredmi kruhov
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Ak sú kruhy príliš ďaleko od seba, nemajú prienik
    if d >= (r1 + r2):
        return 0.0

    # Ak je jeden kruh úplne vo vnútri druhého
    if d <= abs(r1 - r2):
        return min(r1, r2) ** 2 / max(r1, r2) ** 2

    # Výpočet plôch kruhov
    area1 = np.pi * r1 ** 2
    area2 = np.pi * r2 ** 2

    # Vzorec na výpočet prieniku dvoch kruhov
    part1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    part2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

    intersection_area = part1 + part2 - part3
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area


def evaluate_detection(detected_circles, annotated_circles, iou_threshold=0.75):
    """Vypočíta Precision, Recall a F1-skóre na základe IoU medzi detegovanými a anotovanými kruhmi."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_annotated = set()

    for det in detected_circles:
        best_match = None
        best_iou = 0.0

        for i, ann in enumerate(annotated_circles):
            if i not in matched_annotated:  # Zabezpečíme, že anotovaný kruh bude použitý len raz
                iou = compute_iou(det, ann)
                if iou > best_iou:
                    best_iou = iou
                    best_match = i

        if best_match is not None and best_iou >= iou_threshold:
            true_positives += 1
            matched_annotated.add(best_match)
        else:
            false_positives += 1

    false_negatives = len(annotated_circles) - len(matched_annotated)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    return precision, recall, f1_score


annotated = [[132,166,29],[127,162,103],[179,-192,419],[194,390,301]]
image_path = "duhovky/013/L/S1013L01.jpg"
image = cv2.imread(image_path)
csv_path = "iris_annotation.csv"
setup_trackbars()
update(0)

def load_annotations(csv_path):
    annotations = {}
    with open(csv_path, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 13:
                continue
            image_path = parts[0]
            values = list(map(int, parts[1:]))
            annotations[image_path] = {
                "pupil": tuple(values[0:3]),
                "iris": tuple(values[3:6]),
                "bottom_lid": tuple(values[6:9]),
                "top_lid": tuple(values[9:12])
            }
    return annotations

def hough_circles(img, params):
    """Aplikuje Hough Transformáciu s danými parametrami."""
    return cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT,
        dp=params["dp"], minDist=params["minDist"],
        param1=params["param1"], param2=params["param2"],
        minRadius=params["minRadius"], maxRadius=params["maxRadius"]
    )

def grid_search(image, ground_truth, param_grid):
    best_score = 0
    best_params = None
    results = []

    # Všetky kombinácie parametrov
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

    for params in param_combinations:
        detected = hough_circles(image, params)
        if detected is not None:
            detected = np.uint16(np.around(detected[0, :]))

            precision, recall, f1 = evaluate_detection(detected, ground_truth)

            results.append((params, precision, recall, f1))

            if f1 > best_score:
                best_score = f1
                best_params = params

    return best_params, results


while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC na ukončenie
        break
    if key == ord("c"):
        result, pupil, iris, top_lid, bottom_lid = four_circles(image)
        pupil = np.uint16(np.around(pupil[0, :])).tolist()
        iris = np.uint16(np.around(iris[0, :])).tolist()
        top_lid = np.uint16(np.around(top_lid[0, :])).tolist()
        bottom_lid = np.uint16(np.around(bottom_lid[0, :])).tolist()
        detected = [pupil, iris, top_lid, bottom_lid]
        detected = remove_padding(detected)
        print(f"Detected circles: {detected}")
        print(f"Annotated circles: {annotated}")
        cv2.imshow("Vykreslenie kruznic", result)


        precision, recall, f1_score = evaluate_detection(detected, annotated)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}")


cv2.destroyAllWindows()

param_grid = {
    "dp": [1, 1.5, 2],
    "minDist": [20, 30, 50],
    "param1": [50, 100, 150],
    "param2": [20, 30, 40],
    "minRadius": [10, 20, 30],
    "maxRadius": [100, 150, 200]
}

# Zavolanie grid search na celý dataset
#best_params, results = grid_search(image, ground_truth, param_grid)

# Výpis optimálnych parametrov
#print("Best Parameters:", best_params)

#df = pd.DataFrame(results, columns=["Parameters", "Precision", "Recall", "F1-score"])
#df.to_csv("grid_search_results.csv", index=False)
#print(df)