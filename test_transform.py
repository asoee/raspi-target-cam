import cv2
import os
import numpy as np
from perspective import Perspective
import math

# Create output directory
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

# Test with ellipse at angle 60° (similar to real capture)
img = np.zeros((800, 800, 3), dtype=np.uint8)
img.fill(200)
cv2.ellipse(img, (400, 400), (150, 135), 150, 0, 360, (50, 50, 50), -1)

print("Test: Ellipse with angle=60° (similar to real capture)")
print("="*70)

cv2.imwrite(os.path.join(output_dir, 'test_transform_input.png'), img)

p = Perspective()
e, _, _, _, _ = p.experimental_ellipse_detection(img, min_area=1000, min_circularity=0.3)
if e:
    (cx, cy), (a1, a2), ang = e
    print(f"\nDetected: axis1={a1:.1f}, axis2={a2:.1f}, angle={ang:.1f}°")
    
    dimg = img.copy()
    icx = round(cx)
    icy = round(cy)
    cv2.ellipse(dimg, (icx, icy), (round(a1/2), round(a2/2)), ang, 0, 360, (250, 50, 50), 2)
    cv2.imwrite(os.path.join(output_dir, 'test_transform_detected_ellipse.png'), dimg)

    a1e = a1*math.sqrt(2)
    a2e = a2*math.sqrt(2)

    majdx = (a2e/2) * math.cos(math.radians(90-ang))
    majdy = (a2e/2) * math.sin(math.radians(90-ang))
    mindx = (a1e/2) * math.cos(math.radians(ang))
    mindy = (a1e/2) * math.sin(math.radians(ang))

    p1= (round(icx + majdx), round(icy - majdy))
    p2= (round(icx - majdx), round(icy + majdy))
    p3= (round(icx - mindx), round(icy - mindy))
    p4= (round(icx + mindx), round(icy + mindy))

    p3c= (round(icx - majdy), round(icy - majdx))
    p4c= (round(icx + majdy), round(icy + majdx))
    
    cv2.circle(dimg, p1, radius=3, color=(50,250,50), thickness=-1)
    cv2.circle(dimg, p2, radius=3, color=(50,250,50), thickness=-1)
    cv2.circle(dimg, p3, radius=3, color=(50,250,50), thickness=-1)
    cv2.circle(dimg, p4, radius=3, color=(50,250,50), thickness=-1)
    cv2.circle(dimg, p3c, radius=3, color=(50,50,250), thickness=-1)
    cv2.circle(dimg, p4c, radius=3, color=(50,50,250), thickness=-1)

    cv2.imwrite(os.path.join(output_dir, 'test_transform_bounding_box.png'), dimg)
    
    M, r, c = p.calculate_ellipse_to_circle_matrix(e, (800, 800))
    
    p1a = np.array(p1)
    p2a = np.array(p2)
    p3a = np.array(p3)
    p4a = np.array(p4)
    p3ca = np.array(p3c)
    p4ca = np.array(p4c)

    pts1 = np.float32([p1a,p2a,p3a,p4a])
    pts2 = np.float32([p1a,p2a,p3ca,p4ca])
    pm = cv2.getPerspectiveTransform(pts1, pts2)
    print(f"\nfrom points: {pts1}")
    print(f"\nto points: {pts2}")

    print(f"\nMatrix: {pm}")

    
    # Apply and check
    transformed = cv2.warpPerspective(img, pm, (800, 800))

    cv2.imwrite(os.path.join(output_dir, 'test_transform_bounding_transformed.png'), transformed)

    result, _, _, _, _ = p.experimental_ellipse_detection(
        transformed, min_area=1000, min_circularity=0.3
    )
    if result:
        (rcx, rcy), (ra1, ra2), rang = result
        print(f"\nAfter transform: axis1={ra1:.1f}, axis2={ra2:.1f}")
        print(f"Difference: {abs(ra1-ra2):.1f} pixels")
    else:
        print("\nCould not detect in result - too circular (good!)")