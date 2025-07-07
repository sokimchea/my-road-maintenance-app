import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import numpy as np
import requests
import matplotlib as mpl
from matplotlib import transforms
import matplotlib.patches as mpatches



# Use Agg backend for better Unicode rendering
mpl.use("agg")

# Load Khmer font
font_path = "FONT/KhmerOSsiemreap.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# Constants
DATA_FILE = "road_maintenance_updated.xlsx"
GOOGLE_SHEET_FILE_ID = "1ESWPe49WlQ1608IH8bACw_XiTiRx4EFTCLvBBW44K_E"
EXCEL_EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_FILE_ID}/export?format=xlsx"

# Refresh button to download from Google Sheets
if st.sidebar.button("🔄 Refresh Data"):
    with st.spinner("Downloading latest Excel data from Google Sheet..."):
        r = requests.get(EXCEL_EXPORT_URL)
        if r.status_code == 200:
            with open(DATA_FILE, 'wb') as f:
                f.write(r.content)
            st.success("✅ Excel downloaded from Google Sheet")
        else:
            st.error("❌ Failed to download Excel. Please check sharing permissions.")

# Load data
if os.path.exists(DATA_FILE):
    df = pd.read_excel(DATA_FILE)
# Remove .0 formatting from Year and Chapter
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).astype(str)
    df["Chapter"] = pd.to_numeric(df["Chapter"], errors="coerce").dropna().astype(int).astype(str)
else:
    st.error("Excel data file not found.")
    st.stop()

# Sidebar Filters
road_ids = df["Road_ID"].dropna().unique()
selected_road = st.sidebar.selectbox("Select Road ID", road_ids)

# Lookup PK Start/End based on selection
road_df = df[df["Road_ID"] == selected_road]
pk_min = float(road_df["PK_Start"].min())
pk_max = float(road_df["PK_End"].max())

start_input = st.sidebar.number_input("PK Start (input)", min_value=0, max_value=int(pk_max), value=int(pk_min),step=100)
end_input = st.sidebar.number_input("PK End (input)", min_value=pk_min, max_value=pk_max, value=pk_max)

years = sorted(df["Year"].dropna().unique(), reverse=True)
selected_years = st.sidebar.multiselect("Select Year(s) to Show", years, default=years)

types = df["Maintenance_Type"].dropna().unique()
selected_types = st.sidebar.multiselect("Select Maintenance Type(s)", types, default=types)

chapters = df["Chapter"].dropna().unique()
selected_chapter = st.sidebar.multiselect("Select Chapter(s)", chapters, default=chapters)

# Layer order (Bottom to Top)
request_years = sorted(df[df["Type"] == "Request"]["Year"].dropna().unique(), reverse=True)
available_layers = [f"Approval {y}" for y in selected_years] + [f"Request {y}" for y in request_years if y in selected_years]
layer_order = st.sidebar.multiselect("Set Layer Order (Bottom to Top)", available_layers, default=available_layers[::-1])

# Color & pattern
st.sidebar.markdown("### Customize Colors")
color_map = {}
default_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#7f8c8d"]
for i, t in enumerate(types):
    color = st.sidebar.color_picker(f"{t} Color", default_colors[i % len(default_colors)])
    color_map[t] = color

# Font size customization
st.sidebar.markdown("### Customize Font Size")
font_size = st.sidebar.slider("Font Size", min_value=6, max_value=20, value=15)
title_font_size = st.sidebar.slider("Title Font Size", min_value=10, max_value=30, value=20)

# Manual Annotation
st.sidebar.markdown("---")
st.sidebar.markdown("### Add Manual Location Label")

manual_labels = []
num_labels = st.sidebar.number_input(
    "How many labels to add?", min_value=0, max_value=10, step=1, value=0
)

for i in range(num_labels):
    with st.sidebar.expander(f"Label #{i+1}"):
        label_text = st.text_input(f"Label Text {i+1}", key=f"label_text_{i}")
        label_pk_start = st.number_input(
            f"Label PK Start {i+1}", min_value=pk_min, max_value=pk_max, key=f"label_pk_start_{i}"
        )
        label_pk_end = st.number_input(
            f"Label PK End {i+1}", min_value=pk_min, max_value=pk_max, key=f"label_pk_end_{i}"
        )
        label_color = st.color_picker(f"Label Color {i+1}", "#000000", key=f"label_color_{i}")
        manual_labels.append((label_text, label_pk_start, label_pk_end, label_color))
# Filter data
filtered = df[
    (df["Road_ID"] == selected_road) &
    (df["PK_End"] >= start_input) &
    (df["PK_Start"] <= end_input) &
    (df["Maintenance_Type"].isin(selected_types)) &
    (df["Chapter"].isin(selected_chapter)) &
    (df["Year"].isin(selected_years))
]

if filtered.empty:
    st.warning("⚠ No data matches your filters. Please adjust selections.")
    st.stop()

# 📊 Preview Chart
st.markdown("### 📊 Preview Chart")

from collections import defaultdict
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

# Step 1: Build y_map, label list, and groupings
y_map, y_labels, y_label_colors = {}, [], {}
subrow_tracker = defaultdict(list)
group_titles = []

row_index = 0
for label in layer_order:
    is_request = label.startswith("Request")
    key_prefix = label.replace(" ", "_")

    if is_request:
        year = label.split()[-1]
        filtered_sub = filtered[
            (filtered["Type"] == "Request") & (filtered["Year"].astype(str) == year)
        ]
        unique_types = sorted(filtered_sub["Maintenance_Type"].dropna().map(str.strip).unique())
        if not unique_types:
            continue

        for m_type in unique_types:
            norm_type = m_type.replace(" ", "_")
            sub_key = f"{key_prefix}_{norm_type}"
            y_map[sub_key] = row_index
            y_labels.append(m_type)
            y_label_colors[row_index] = "green"
            subrow_tracker[label].append(row_index)
            row_index += 1

        group_y = sum(subrow_tracker[label]) / len(subrow_tracker[label])
        group_titles.append((label, group_y))
    else:
        y_key = key_prefix
        y_map[y_key] = row_index
        y_labels.append(label)
        y_label_colors[row_index] = "black"
        subrow_tracker[label].append(row_index)
        row_index += 1

# Step 2: Setup plot
fig_preview, ax = plt.subplots(figsize=(16, 6))
label_positions = {}
pk_label_positions = []

# Step 3: Plot segments with sequence
legend_handles = {}
request_segments = filtered[filtered["Type"] == "Request"].sort_values(by="PK_Start").reset_index()
sequence_number = 1

for idx, seg in filtered.iterrows():
    mtype = str(seg['Maintenance_Type']).strip()
    color = color_map.get(mtype, "gray")
    mtype_key = mtype.replace(" ", "_")

    if seg["Type"] == "Request":
        key = f"Request_{seg['Year']}_{mtype_key}"
    else:
        key = f"Approval_{seg['Year']}".replace(" ", "_")

    if key not in y_map:
        continue

    y = y_map[key]
    clipped_start = max(seg["PK_Start"], start_input)
    clipped_end = min(seg["PK_End"], end_input)
    if clipped_start >= clipped_end:
        continue

    # Draw bar with black border for requests
    edgecolor = "black" if seg["Type"] == "Request" else "none"
    bar = ax.barh(y, width=clipped_end - clipped_start, left=clipped_start,
                  color=color, edgecolor=edgecolor, height=0.6)
    if mtype not in legend_handles:
        legend_handles[mtype] = bar[0]

    # Add sequence number in circle on bar for request
    if seg["Type"] == "Request":
        center_x = (clipped_start + clipped_end) / 2
        ax.text(center_x, y, str(sequence_number),
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            fontproperties=font_prop, zorder=5)
        seg_sequence = sequence_number
        sequence_number += 1
    else:
        seg_sequence = ""

# Step 4: Draw red PK labels with sequence
sequence_number = 1
for _, seg in request_segments.iterrows():
    pk_start, pk_end = seg["PK_Start"], seg["PK_End"]
    label_x = (pk_start + pk_end) / 2
    pk_label = f"({sequence_number}). {int(pk_start//1000)}+{int(pk_start%1000):03d} to {int(pk_end//1000)}+{int(pk_end%1000):03d}"

    pk_offset = 1.0
    for prev_x in pk_label_positions:
        if abs(label_x - prev_x) < 15000:
            pk_offset += 0.35
    pk_label_positions.append(label_x)

    ax.text(label_x, -pk_offset, pk_label,
            ha='center', va='top', fontsize=10, color="darkred",
            fontproperties=font_prop, clip_on=False)

    ax.axvline(pk_start, linestyle="dashed", color=color_map.get(seg["Maintenance_Type"], "gray"), alpha=0.6)
    ax.axvline(pk_end, linestyle="dashed", color=color_map.get(seg["Maintenance_Type"], "gray"), alpha=0.6)

    sequence_number += 1

# Step 5: Manual Labels
for label_text, pk_start, pk_end, label_color in manual_labels:
    if start_input > pk_end or end_input < pk_start:
        continue
    clipped_start = max(pk_start, start_input)
    clipped_end = min(pk_end, end_input)
    if clipped_start >= clipped_end:
        continue
    label_x = (clipped_start + clipped_end) / 2
    ax.axvline(pk_start, linestyle="dashed", color=label_color, linewidth=1.2, alpha=0.8)
    ax.axvline(pk_end, linestyle="dashed", color=label_color, linewidth=1.2, alpha=0.8)
    ax.text(label_x, max(y_map.values()) + 0.5, label_text,
            ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=label_color, backgroundcolor='white', clip_on=False)

# Step 6: Axes formatting
def format_pk(x, pos):
    return f"{int(x // 1000)}+{int(x % 1000):03d}"
ax.xaxis.set_major_formatter(FuncFormatter(format_pk))
ax.set_xlim(start_input, end_input)
ax.set_ylim(-1, max(y_map.values()) + 1)
ax.set_title(f"\n\nRoad: {selected_road}", fontproperties=font_prop, fontsize=title_font_size)
ax.grid(True)

# Step 7: Y labels
ax.set_yticks(list(y_map.values()))
for y_val, label in zip(y_map.values(), y_labels):
    ax.text(start_input - 5000, y_val, label,
            va='center', ha='right', fontsize=font_size,
            fontproperties=font_prop, color=y_label_colors[y_val])

# Step 8: Group vertical labels
for group_label, y_pos in group_titles:
    ax.text(start_input - 40000, y_pos, group_label,
            fontsize=font_size + 1, fontproperties=font_prop,
            color="green", ha='center', va='center', rotation=90,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="green", facecolor="none"),
            clip_on=False)

# Step 9: Request bounding boxes
for label, rows in subrow_tracker.items():
    if not label.startswith("Request") or not rows:
        continue
    top = max(rows) + 0.5
    bottom = min(rows) - 0.5
    ax.hlines([bottom, top], xmin=start_input, xmax=end_input, color='green', linewidth=1.5)
    ax.vlines([start_input, end_input], ymin=bottom, ymax=top, color='green', linewidth=1.5)

# Step 10: Adjust layout and add legend
plt.subplots_adjust(left=0.15, right=0.88)
if legend_handles:
    ax.legend(legend_handles.values(), legend_handles.keys(),
              title="Maintenance Type", loc="upper right")

# Show chart
st.pyplot(fig_preview)

# Summary table
st.markdown("### 📊 Maintenance Summary by Section")

sum_df = filtered.copy()

# Assign sequence number based on PK_Start order within Request type
sum_df["Seq"] = 0
is_request = sum_df["Type"] == "Request"
sum_df.loc[is_request, "Seq"] = sum_df[is_request].sort_values(by="PK_Start").reset_index(drop=True).index + 1

# Format PK label with sequence for Requests
def format_pk_label(row):
    label = f"{int(row['PK_Start']//1000)}+{int(row['PK_Start']%1000):03d} to {int(row['PK_End']//1000)}+{int(row['PK_End']%1000):03d}"
    if row["Type"] == "Request":
        return f"({int(row['Seq'])}). {label}"
    else:
        return label

sum_df["PK_Label"] = sum_df.apply(format_pk_label, axis=1)

# Calculate distance
sum_df["Distance_km"] = ((sum_df["PK_End"] - sum_df["PK_Start"]) / 1000).round(2)
sum_df["Group"] = sum_df["Type"]
sum_df["Maintenance"] = sum_df.apply(lambda row: f"{row['Maintenance_Type']} [{row['Type']}]", axis=1)

# Group and join PKs by maintenance type
sum_grouped = sum_df.groupby(["Group", "Maintenance"]).agg({
    "PK_Label": lambda x: ", ".join(x),
    "Distance_km": "sum"
}).reset_index()

# Rename and format columns
sum_grouped = sum_grouped.rename(columns={
    "Maintenance": "Maintenance [Type]",
    "PK_Label": "PK Range",
    "Distance_km": "Total Distance (km)"
})
sum_grouped["Total Distance (km)"] = sum_grouped["Total Distance (km)"].round(2)

# Show in Streamlit
st.dataframe(sum_grouped)


# ------------------------------------------------------------------
# 📤  Export Chart & Summary → PDF   (single-click download)
# ------------------------------------------------------------------
import io
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

st.markdown("---")

def build_pdf() -> bytes:
    """Generate the PDF and return its bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # ---------- figure & grid ----------
        fig = plt.figure(figsize=(16.5, 11.7))             # A3 landscape
        gs  = GridSpec(2, 1, height_ratios=[3.5, 1])
        fig.subplots_adjust(left=0.1, right=0.9,
                            top=0.9,  bottom=0.1)

        # ----------------------------------------------------------
        # 1️⃣  CHART (ax1)  –– your existing logic (unchanged)
        # ----------------------------------------------------------
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title(f"\n\nRoad: {selected_road}",
                      fontproperties=font_prop,
                      fontsize=title_font_size)
        ax1.set_xlabel("PK (Chainage in km)",
                       fontproperties=font_prop,
                       fontsize=font_size)

        for _, seg in filtered.iterrows():
            key = (f"Request_{seg['Year']}_{seg['Maintenance_Type']}"
                   if seg["Type"] == "Request"
                   else f"Approval_{seg['Year']}").replace(" ", "_")
            if key not in y_map:
                continue
            y  = y_map[key]
            c  = color_map.get(seg["Maintenance_Type"], "gray")
            s  = max(seg["PK_Start"], start_input)
            e  = min(seg["PK_End"],   end_input)
            if s >= e:
                continue
            ax1.barh(y, e-s, left=s, color=c,
                     edgecolor="black" if seg["Type"]=="Request" else "none",
                     height=0.4)
            if seg["Type"] == "Request":
                ax1.axvline(s, linestyle="dashed", color=c, alpha=0.6)
                ax1.axvline(e, linestyle="dashed", color=c, alpha=0.6)

        ax1.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _:
                          f"{int(x//1000)}+{int(x%1000):03d}"))
        ax1.set_xlim(start_input, end_input)
        ax1.set_ylim(-0.5, max(y_map.values())+0.5)
        ax1.set_yticks(list(y_map.values()))
        for y_val, lab in zip(y_map.values(), y_labels):
            ax1.text(start_input-40000, y_val, lab,
                     ha='right', va='center', fontsize=font_size,
                     fontproperties=font_prop,
                     color=y_label_colors[y_val])
        ax1.grid(True)

        # ----------------------------------------------------------
        # 2️⃣  SUMMARY TABLE (ax2)
        # ----------------------------------------------------------
        ax2 = fig.add_subplot(gs[1]); ax2.axis("off")

        def wrap_pk(txt, every=3):
            parts = txt.split(", ")
            return "\n".join(", ".join(parts[i:i+every])
                             for i in range(0, len(parts), every))

        tbl_df = sum_grouped.copy()
        tbl_df["PK Range"] = tbl_df["PK Range"].apply(wrap_pk)
        data      = tbl_df.values.tolist()
        headers   = list(tbl_df.columns)
        col_width = [0.12, 0.20, 0.53, 0.15]

        tbl = ax2.table(cellText=data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0,0,1,1])
        tbl.auto_set_font_size(False)
        for (r, c), cell in tbl.get_celld().items():
            if c < len(col_width):
                cell.set_width(col_width[c])
            cell.set_fontsize(11 if c==2 else 13)
            cell.set_linewidth(0.7)
            cell.set_height(0.15)
            if not r:
                cell.set_facecolor("#003366")
                cell.set_text_props(color="white", weight="bold")
            elif "Request" in str(data[r-1][0]):
                cell.set_facecolor("#e5f5e5")
        tbl.scale(1, 2.0)

        # ---------- add page ----------
        pdf.savefig(fig, bbox_inches='tight')

    buf.seek(0)          # rewind
    return buf.getvalue()

# 👉 single button: when clicked, build_pdf() runs & file downloads
filename = f"{selected_road}_{int(start_input)}_{int(end_input)}.pdf"
st.download_button(
    label="📤 Export Chart & Summary to PDF",
    data=build_pdf,                 # <-- pass the callable
    file_name=filename,
    mime="application/pdf"
)
