import streamlit as st
import pandas as pd
import time
import re
from PIL import Image, ImageDraw

# ==============================
# CONFIG
# ==============================
RL_FILE = "rl_output.txt"   # Your saved RL output file path
REFRESH_RATE = 1.0          # seconds between lane updates

# ==============================
# Parser for RL Output
# ==============================
def parse_rl_file(file_path):
    """
    Parse lines like:
    Step  495 | Lane 1 | Max_Queue_length : 33 | Green Time: 21s | Queues [33, 27, 24, 23]
    """
    LINE_RE = re.compile(
        r"Step\s+(\d+)\s*\|\s*Lane\s+(\d+)\s*\|\s*Max_Queue_length\s*[:=]\s*(\d+)\s*\|\s*Green\s*Time\s*[:=]\s*(\d+)s\s*\|\s*Queues\s*\[([0-9,\s]+)\]",
        re.IGNORECASE
    )

    steps = []
    import chardet

    # Auto-detect encoding
    rawdata = open(file_path, "rb").read()
    result = chardet.detect(rawdata)
    encoding_used = result["encoding"] or "utf-8"

    with open(file_path, "r", encoding=encoding_used, errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line.strip())
            if m:
                step = int(m.group(1))
                lane = int(m.group(2))
                max_q = int(m.group(3))
                green_t = int(m.group(4))
                queues = [int(x.strip()) for x in m.group(5).split(",") if x.strip().isdigit()]
                steps.append({
                    "step": step,
                    "lane": lane,
                    "max_q": max_q,
                    "green_time": green_t,
                    "queues": queues
                })
    return steps


# ==============================
# Drawing Intersection
# ==============================
def draw_intersection(step_data, w=640, h=520):
    im = Image.new("RGBA", (w, h), (10, 14, 20, 255))
    d = ImageDraw.Draw(im)
    cx, cy = w // 2, h // 2
    road_w = 140

    # Draw roads
    d.rectangle([(cx - road_w//2, 0), (cx + road_w//2, h)], fill=(34, 40, 48))
    d.rectangle([(0, cy - road_w//2), (w, cy + road_w//2)], fill=(34, 40, 48))

    # Lane boxes: N, E, S, W
    box_w, box_h = 150, 90
    boxes = [
        (cx - box_w//2, 20, cx + box_w//2, 20 + box_h),        # North
        (w - 20 - box_w, cy - box_h//2, w - 20, cy + box_h//2),# East
        (cx - box_w//2, h - 20 - box_h, cx + box_w//2, h - 20),# South
        (20, cy - box_h//2, 20 + box_w, cy + box_h//2),        # West
    ]

    lane_labels = ['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4']
    q = step_data["queues"]
    active_lane = step_data["lane"] - 1

    for i, b in enumerate(boxes):
        if i == active_lane:
            d.rectangle(b, fill=(12, 160, 80))  # Green
            outer = [b[0]-4, b[1]-4, b[2]+4, b[3]+4]
            d.rectangle(outer, outline=(12,160,80,180))
        else:
            d.rectangle(b, fill=(120, 20, 20))  # Red

        # Write lane and vehicle info
        d.text((b[0]+10, b[1]+10), lane_labels[i], fill=(255, 255, 255))
        d.text((b[0]+10, b[1]+30), f"Vehicles: {q[i]}", fill=(230, 230, 230))

    # Intersection center
    r = 60
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(24, 28, 36))
    return im


# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="🚦 Smart Adaptive Traffic Signal — RL Visualization", layout="wide")
st.title("🚦 Smart Adaptive Traffic Signal — RL Visualization")


# Load data
steps = parse_rl_file(RL_FILE)

if not steps:
    st.error("❌ No valid RL output found in file. Please ensure rl_output.txt contains the step data.")
    st.stop()

# Display total frames info
st.success(f"✅ Loaded {len(steps)} steps from RL output")

# Layout
col1, col2 = st.columns([0.65, 0.35])
img_box = col1.empty()
info_box = col2.empty()
table_box = col2.empty()
timer_box = st.empty()  # ⏱ Timer displayed below table

# Visualization loop
for step_data in steps:
    green_time = step_data['green_time']
    active_lane = step_data['lane'] - 1

    info_box.markdown(
        f"""
        ### Step {step_data['step']}
        - 🟩 **Active Lane:** {step_data['lane']}
        - 🚗 **Max Queue Length:** {step_data['max_q']}
        - ⏱️ **Green Time:** {step_data['green_time']}s
        - 🧮 **Queue State:** {step_data['queues']}
        """
    )

    df = pd.DataFrame({
        "Lane": ["Lane 1", "Lane 2", "Lane 3", "Lane 4"],
        "Vehicles": step_data["queues"]
    })
    table_box.table(df)

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        remaining = max(0, int(green_time - elapsed))
        img_box.image(draw_intersection(step_data), use_container_width=True)
        timer_box.markdown(f"### ⏱ Countdown: **{remaining}s** remaining")

        if remaining <= 0:
            break
        time.sleep(REFRESH_RATE)
