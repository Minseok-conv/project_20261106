"""실습 01. 정답 박스 3개를 그립니다. 원본 scene.png는 바꾸지 않습니다."""
from pathlib import Path
import cv2
import pandas as pd
BASE = Path(__file__).resolve().parent
image = cv2.imread(str(BASE / "data/scene.png"))
#cv2.imread()에 이미지 파일 경로를 주면 파일을 읽어서 **NumPy 배열(ndarray)**로 반환
boxes = pd.read_csv(BASE / "data/boxes_gt.csv")
result = image.copy()  # 그릴 배열을 복사하여 원본을 보존합니다.
for row in boxes.itertuples():
    #boxes.itertuples()로 박스 정보를 한 행씩 꺼냅니다.
    cv2.rectangle(
    result,
    (int(row.x1), int(row.y1)),
    (int(row.x2), int(row.y2)),
    (0, 255, 0),
    thickness=2,
    )
    print("class:", row.class_id, "xyxy:", row.x1, row.y1, row.x2, row.y2, "GT confidence: N/A")
cv2.imwrite(str(BASE / "p01_boxes.png"), result)
print("shape:", image.shape, "저장: p01_boxes.png")
