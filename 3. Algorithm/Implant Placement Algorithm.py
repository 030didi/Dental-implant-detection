import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

def line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c

def angle_bisector(eq1, eq2):
    a1, b1, c1 = eq1
    a2, b2, c2 = eq2
    norm1 = math.hypot(a1, b1)
    norm2 = math.hypot(a2, b2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot calculate angle bisector with zero vector")
        
    a_bis1 = a1 / norm1 + a2 / norm2
    b_bis1 = b1 / norm1 + b2 / norm2
    c_bis1 = c1 / norm1 + c2 / norm2
    a_bis2 = a1 / norm1 - a2 / norm2
    b_bis2 = b1 / norm1 - b2 / norm2
    c_bis2 = c1 / norm1 - c2 / norm2
    return (a_bis1, b_bis1, c_bis1), (a_bis2, b_bis2, c_bis2)

def solve_x(coeffs, y_value):
    a, b, c = coeffs
    if a == 0:
        raise ValueError("Cannot solve x from vertical line")
    return -(b * y_value + c) / a

# Calculate area of polygon on both sides of angle bisector
def polygon_area(pts):
    pts = np.array(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def select_bisector_by_smaller_area(bisectors, A, B, C, D, image_shape):
    h, w = image_shape
    midpoint = ((A[0] + C[0]) // 2, (A[1] + C[1]) // 2)

    areas = []
    for eq in bisectors:
        try:
            # Find two points to draw the angle bisector
            y0 = 0
            y1 = h - 1
            x0 = solve_x(eq, y0)
            x1 = solve_x(eq, y1)
            p0 = (int(x0), y0)
            p1 = (int(x1), y1)

            # Create two triangles or polygons divided by the angle bisector
            side1 = [A, B, p1, p0]
            side2 = [C, D, p1, p0]

            area1 = polygon_area(side1)
            area2 = polygon_area(side2)

            smaller_area = min(area1, area2)
            areas.append((smaller_area, eq))
        except:
            continue

    if not areas:
        raise ValueError("Cannot calculate area to select angle bisector")

    # Return the angle bisector pointing toward the smaller area
    return min(areas, key=lambda x: x[0])[1]

# ========== Image and label file paths ==========
cover_image_path = "./1.png"  # Image 1 (original)
cover_label_path = "./1.txt"  # Label for image 1

crop_image_path = "./obb/1.jpg"  # Image 2 (cropped)
obb_label_path = "./obb/1.txt"  # OBB prediction label on image 2

# ========== Restore cropped coordinates to original image ==========
cover_img = cv2.imread(cover_image_path)
cover_h, cover_w, _ = cover_img.shape

with open(cover_label_path, "r") as f:
    line = f.readline().strip()
    cls, cx, cy, w_, h_= map(float, line.split())
    cx_pix = cx * cover_w
    cy_pix = cy * cover_h
    w_pix = w_ * cover_w
    h_pix = h_ * cover_h
    x_offset = int(cx_pix - w_pix / 2)
    y_offset = int(cy_pix - h_pix / 2)

# ========== Read OBB labels ==========
labels = []
crop_img = cv2.imread(crop_image_path)
h, w, _ = crop_img.shape

try:
    with open(obb_label_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 10:
                label = int(values[0])
                x1, y1, x2, y2, x3, y3, x4, y4 = values[1:9]
                confidence = values[-1]
                corners = [
                    (int(x1 * w) + x_offset, int(y1 * h) + y_offset),
                    (int(x2 * w) + x_offset, int(y2 * h) + y_offset),
                    (int(x3 * w) + x_offset, int(y3 * h) + y_offset),
                    (int(x4 * w) + x_offset, int(y4 * h) + y_offset),
                ]
                labels.append({
                    "label": label,
                    "corners": corners,
                    "confidence": confidence
                })
except FileNotFoundError:
    print("Label file not found")
    exit()

print(f"Loaded {len(labels)} labels")

# ========== Calculate angle bisector ==========
bisectors_found = False
for index in range(1, len(labels), 2):
    line1 = labels[index - 1]["corners"]
    line2 = labels[index]["corners"]
    
    a1, a2 = line1[0]
    b1, b2 = line2[0]
    # Ensure line1's starting point is after line2's starting point
    if a1 > b1:
        line1, line2 = line2, line1

    # Compare line segment slopes
    x2, y2 = line1[1]
    x3, y3 = line1[2]
    x1, y1 = line2[0]
    x4, y4 = line2[3]
    slope_l = (y3 - y2) / (x3 - x2 + 1e-6)
    slope_r = (y4 - y1) / (x4 - x1 + 1e-6)

    if abs(slope_l - slope_r) > 1e-2:
        print(f"Labels {index} and {index+1} lines intersect")
        
        pts_a = np.array(line1)
        pts_b = np.array(line2)
        # Calculate distances for all point pairs and record indices
        distances = []
        for i, pt_a in enumerate(pts_a):
            for j, pt_b in enumerate(pts_b):
                dist = np.linalg.norm(pt_a - pt_b)
                distances.append((dist, i, j))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Get first and second closest
        first_closest = distances[0]
        second_closest = distances[1]

        print("First closest distance:", first_closest)
        print("Second closest distance:", second_closest)
        
        A = line1[first_closest[1]]
        B = line1[second_closest[1]]
        C = line2[first_closest[2]]
        D = line2[second_closest[2]]

        cv2.line(cover_img, A, B, (0, 255, 255), 2)  # Yellow
        cv2.line(cover_img, C, D, (255, 0, 0), 2)    # Blue

        eq1 = line_equation(A, B)
        eq2 = line_equation(C, D)
        bisectors = angle_bisector(eq1, eq2)

        try:
            negative_slope_bisector = select_bisector_by_smaller_area(bisectors, A, B, C, D, cover_img.shape[:2])

            bisectors_found = True
            break
        except ValueError as e:
            print(f"Error: {e}")
            continue

if not bisectors_found:
    print("No valid angle bisector found")
    exit()

# ========== Draw angle bisector ==========
y_min = int(min(A[1], D[1]))
y_max = int(max(B[1], C[1]))
try:
    x_min = solve_x(negative_slope_bisector, y_min)
    x_max = solve_x(negative_slope_bisector, y_max)
    cv2.line(cover_img, (int(x_min), y_min), (int(x_max), y_max), (0, 255, 0), 2)
except ValueError as e:
    print(f"Cannot draw angle bisector: {e}")

# ========== Display and save ==========
plt.imshow(cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

cv2.imwrite("final.jpg", cover_img)