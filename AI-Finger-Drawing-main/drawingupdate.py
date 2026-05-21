import cv2
import mediapipe as mp
import numpy as np
import math
import time
from datetime import datetime

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
h_frame, w_frame, _ = frame.shape
canvas = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)

# --- BIẾN VẼ ---
prev_x, prev_y = 0, 0
drawing_mode = False
fingers_touching = False
draw_color = (0, 0, 255)
brush_size = 5

# --- BIẾN CHỤP ẢNH ---
CAPTURE_IDLE      = 0
CAPTURE_COUNTDOWN = 1
CAPTURE_FLASH     = 2
capture_state      = CAPTURE_IDLE
capture_start_time = 0
CAPTURE_DELAY      = 3
flash_start_time   = 0
FLASH_DURATION     = 0.5

# --- UI ---
colors = [(0,0,255),(0,255,255),(0,255,0),(255,0,0),(255,0,255),(255,255,255)]
color_radius  = 20
color_centers = []
sizes         = [5, 10, 20]
size_centers  = []
button_x1, button_y1 = w_frame//2 - 75, 10
button_x2, button_y2 = w_frame//2 + 75, 60

def dist(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def fingers_are_open(lm):
    return lm[8].y < lm[6].y and lm[4].y < lm[2].y

def is_thumb_index_up(lm):
    """Chỉ ngón cái + ngón trỏ giơ lên, 3 ngón còn lại gập."""
    thumb_up   = lm[4].y  < lm[3].y
    index_up   = lm[8].y  < lm[6].y
    middle_dn  = lm[12].y > lm[10].y
    ring_dn    = lm[16].y > lm[14].y
    pinky_dn   = lm[20].y > lm[18].y
    return thumb_up and index_up and middle_dn and ring_dn and pinky_dn

def get_thumb_index_tips(lm):
    """Trả về tọa độ pixel ngón cái (4) và ngón trỏ (8)."""
    tx, ty = int(lm[4].x*w_frame), int(lm[4].y*h_frame)
    ix, iy = int(lm[8].x*w_frame), int(lm[8].y*h_frame)
    return (tx,ty), (ix,iy)

def unit_vec(a, b, length):
    d = np.array(b,dtype=float) - np.array(a,dtype=float)
    n = np.linalg.norm(d)
    return (d/n*length).astype(int) if n>0 else np.array([0,0])

while True:
    ret, frame = cap.read()
    if not ret: break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result    = hands.process(rgb_frame)

    # --- VẼ UI ---
    cv2.rectangle(frame,(button_x1,button_y1),(button_x2,button_y2),(50,50,50),-1)
    cv2.putText(frame,"CLEAR",(button_x1+20,button_y1+35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    color_centers.clear()
    for i,color in enumerate(colors):
        cx,cy = w_frame-50, 80+i*70
        color_centers.append((cx,cy,color))
        cv2.circle(frame,(cx,cy),color_radius,color,-1 if color==draw_color else 3)
        if color==draw_color:
            cv2.circle(frame,(cx,cy),color_radius+5,(255,255,255),2)

    size_centers.clear()
    for i,size in enumerate(sizes):
        cx,cy = 50, 150+i*100
        size_centers.append((cx,cy,size))
        cv2.circle(frame,(cx,cy),size,draw_color,-1)
        cv2.circle(frame,(cx,cy),30,(200,200,200),2)
        if size==brush_size:
            cv2.circle(frame,(cx,cy),35,(0,255,0),3)

    # --- PHÂN LOẠI TAY ---
    frame_hands = []   # tay giơ ngón cái + trỏ (chụp ảnh)
    draw_hand   = None # tay vẽ bình thường

    if result.multi_hand_landmarks:
        for hl in result.multi_hand_landmarks:
            if is_thumb_index_up(hl.landmark):
                frame_hands.append(get_thumb_index_tips(hl.landmark))
            else:
                draw_hand = hl
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # =========================================================
    # KHUNG CHỤP ẢNH: 2 tay, mỗi tay ngón cái + ngón trỏ
    # Tứ giác nối 4 đầu ngón tay, to nhỏ/nghiêng theo ngón
    # =========================================================
    if len(frame_hands) == 2:
        # Sắp xếp tay trái/phải theo trục X ngón cái
        sorted_h = sorted(frame_hands, key=lambda h: h[0][0])
        L_thumb, L_index = sorted_h[0]   # tay trái
        R_thumb, R_index = sorted_h[1]   # tay phải

        # Tứ giác: cái_trái → cái_phải → trỏ_phải → trỏ_trái
        quad = np.array([L_thumb, R_thumb, R_index, L_index], dtype=np.int32)

        # Bounding box để crop
        xs = quad[:,0]; ys = quad[:,1]
        fx1,fy1 = max(0,int(xs.min())), max(0,int(ys.min()))
        fx2,fy2 = min(w_frame,int(xs.max())), min(h_frame,int(ys.max()))

        # Chấm tròn 4 đầu ngón
        for p in quad:
            cv2.circle(frame,tuple(p),10,(0,255,255),-1)
            cv2.circle(frame,tuple(p),12,(255,255,255),2)

        if capture_state == CAPTURE_IDLE:
            capture_state      = CAPTURE_COUNTDOWN
            capture_start_time = time.time()

        if capture_state == CAPTURE_COUNTDOWN:
            elapsed   = time.time() - capture_start_time
            remaining = CAPTURE_DELAY - elapsed

            if remaining <= 0:
                combined_save = cv2.addWeighted(frame,1.0,canvas,0.8,0)
                cropped = combined_save[fy1:fy2, fx1:fx2]
                ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fn  = f"capture_{ts}.png"
                cv2.imwrite(fn, cropped)
                print(f"✅ Đã lưu: {fn}")
                capture_state    = CAPTURE_FLASH
                flash_start_time = time.time()
            else:
                # Tứ giác nhấp nháy theo ngón tay
                blink  = int(elapsed*4)%2==0
                bcolor = (0,255,0) if blink else (0,200,255)
                cv2.polylines(frame,[quad],isClosed=True,color=bcolor,thickness=3)

                # Góc dày ở 4 đỉnh
                clen = 22
                for i in range(4):
                    p  = quad[i]
                    pp = quad[(i-1)%4]
                    pn = quad[(i+1)%4]
                    v1 = unit_vec(p,pp,clen)
                    v2 = unit_vec(p,pn,clen)
                    cv2.line(frame,tuple(p),tuple(p+v1),bcolor,5)
                    cv2.line(frame,tuple(p),tuple(p+v2),bcolor,5)

                # Đếm ngược ở giữa khung
                mid = ((fx1+fx2)//2, (fy1+fy2)//2)
                txt = str(math.ceil(remaining))
                cv2.putText(frame,txt,(mid[0]-25,mid[1]+25),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),8)
                cv2.putText(frame,txt,(mid[0]-25,mid[1]+25),cv2.FONT_HERSHEY_SIMPLEX,3,bcolor,5)
                cv2.putText(frame,"Giu de chup...",(fx1,fy2+28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        if capture_state == CAPTURE_FLASH:
            cv2.polylines(frame,[quad],isClosed=True,color=(0,255,0),thickness=2)

    else:
        if capture_state == CAPTURE_COUNTDOWN:
            capture_state = CAPTURE_IDLE

    # =========================================================
    # VẼ BÌNH THƯỜNG
    # =========================================================
    if draw_hand and capture_state == CAPTURE_IDLE:
        lm = draw_hand.landmark
        ix,iy = int(lm[8].x*w_frame), int(lm[8].y*h_frame)
        tx,ty = int(lm[4].x*w_frame), int(lm[4].y*h_frame)
        d = dist(ix,iy,tx,ty)

        if d > 50:
            for cx,cy,color in color_centers:
                if dist(ix,iy,cx,cy) < color_radius: draw_color=color
            for cx,cy,size in size_centers:
                if dist(ix,iy,cx,cy) < 30: brush_size=size
            if button_x1<ix<button_x2 and button_y1<iy<button_y2:
                canvas = np.zeros((h_frame,w_frame,3),dtype=np.uint8)

        if d < 40:
            if not fingers_touching and fingers_are_open(lm):
                drawing_mode = not drawing_mode
                if drawing_mode: time.sleep(0.2)
            fingers_touching = True
        else:
            fingers_touching = False

        if drawing_mode:
            cv2.circle(frame,(ix,iy),brush_size,draw_color,-1)
            if prev_x==0 and prev_y==0: prev_x,prev_y=ix,iy
            cv2.line(canvas,(prev_x,prev_y),(ix,iy),draw_color,brush_size*2)
            prev_x,prev_y=ix,iy
        else:
            prev_x,prev_y=0,0

    # =========================================================
    # FLASH SAU KHI CHỤP
    # =========================================================
    if capture_state == CAPTURE_FLASH:
        ef = time.time()-flash_start_time
        if ef < FLASH_DURATION:
            alpha = 1.0 - ef/FLASH_DURATION
            white = np.ones_like(frame)*255
            cv2.addWeighted(white,alpha,frame,1-alpha,0,frame)
            cv2.putText(frame,"Da luu anh!",(w_frame//2-130,h_frame//2),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,180,0),4)
        else:
            capture_state = CAPTURE_IDLE

    combined = cv2.addWeighted(frame,1.0,canvas,0.8,0)
    cv2.imshow("AI Finger Drawing - Advanced", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()