import matplotlib.pyplot as plt
import numpy as np
import io
import functools
import urllib.request
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

@functools.lru_cache(maxsize=128)
def get_flag_image(team_name: str):
    """Fetch flag image from FlagCDN and return as PIL Image, cached in memory."""
    from app.shared_theme import get_flag_url
    url = get_flag_url(team_name)
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            img_bytes = response.read()
        img = Image.open(io.BytesIO(img_bytes))
        return img.convert('RGBA')
    except Exception:
        return None

def generate_matchup_png(home_team, away_team, home_elo, away_elo, hw, dr, aw, predicted, confidence):
    """Generate matchup prediction infographic."""
    import matplotlib.patches as patches
    
    fig_w, fig_h = 10, 6
    fig = Figure(figsize=(fig_w, fig_h), facecolor='#0e1117')
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.set_facecolor('#0e1117')
    ax.axis('off')
    
    # Force axes limits to keep rectangle positions stable
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Title (using ax.text for perfect alignment stability)
    ax.text(0.5, 0.92, "MATCH DECISION TELEMETRY", color='#10b981', fontsize=12, fontweight='bold', ha='center', va='center')
    
    # Symmetrical Stacking Layout (All drawn using ax.text in data coordinates)
    # VS Center
    ax.text(0.5, 0.70, "VS", color='#e5c158', fontsize=20, fontweight='bold', ha='center', va='center')
    
    # Home Team Stack (centered at 0.28)
    ax.text(0.28, 0.70, home_team.upper(), color='#38bdf8', fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(0.28, 0.61, f"Elo {home_elo:.0f}", color='#888888', fontsize=12, ha='center', va='center')
    
    # Away Team Stack (centered at 0.72)
    ax.text(0.72, 0.70, away_team.upper(), color='#f87171', fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(0.72, 0.61, f"Elo {away_elo:.0f}", color='#888888', fontsize=12, ha='center', va='center')
    
    # Draw Flags using ax.imshow for pixel-perfect layout stability (no AnnotationBbox drift)
    home_flag = get_flag_image(home_team)
    away_flag = get_flag_image(away_team)
    
    flag_width_x = 0.08
    
    if home_flag:
        home_arr = np.array(home_flag)
        h_px, w_px, _ = home_arr.shape
        aspect = w_px / h_px
        flag_height_y = flag_width_x * (fig_w / fig_h) / aspect
        
        cx, cy = 0.28, 0.81
        extent = [cx - flag_width_x/2, cx + flag_width_x/2, cy - flag_height_y/2, cy + flag_height_y/2]
        ax.imshow(home_arr, extent=extent, zorder=4)
        
    if away_flag:
        away_arr = np.array(away_flag)
        h_px, w_px, _ = away_arr.shape
        aspect = w_px / h_px
        flag_height_y = flag_width_x * (fig_w / fig_h) / aspect
        
        cx, cy = 0.72, 0.81
        extent = [cx - flag_width_x/2, cx + flag_width_x/2, cy - flag_height_y/2, cy + flag_height_y/2]
        ax.imshow(away_arr, extent=extent, zorder=4)
    
    # Draw Probability Bar
    bar_y = 0.45
    bar_height = 0.1
    
    hw_width = hw
    dr_width = dr
    aw_width = aw
    
    rect_hw = patches.Rectangle((0, bar_y), hw_width, bar_height, color='#38bdf8')
    rect_dr = patches.Rectangle((hw_width, bar_y), dr_width, bar_height, color='#e5c158')
    rect_aw = patches.Rectangle((hw_width + dr_width, bar_y), aw_width, bar_height, color='#f87171')
    
    ax.add_patch(rect_hw)
    ax.add_patch(rect_dr)
    ax.add_patch(rect_aw)
    
    # Labels (drawn using ax.text and centered inside the bar)
    if hw_width > 0.08:
        ax.text(hw_width / 2, bar_y + 0.05, f"{hw_width*100:.1f}%\nHOME WIN", color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='center')
    if dr_width > 0.08:
        ax.text(hw_width + dr_width / 2, bar_y + 0.05, f"{dr_width*100:.1f}%\nDRAW", color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='center')
    if aw_width > 0.08:
        ax.text(hw_width + dr_width + aw_width / 2, bar_y + 0.05, f"{aw_width*100:.1f}%\nAWAY WIN", color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='center')
        
    # Decision Banner
    decision_text = f"MODEL DECISION: {predicted.upper()} (CONFIDENCE: {confidence*100:.1f}%)"
    ax.text(0.5, 0.23, decision_text, color='#10b981', fontsize=12, fontweight='bold', ha='center', va='center',
             bbox=dict(facecolor='#111827', edgecolor='#10b981', boxstyle='round,pad=0.8'))
             
    ax.text(0.5, 0.08, "FIFA World Cup Match Outcome Predictor Engine", color='#4b5563', fontsize=8, ha='center', va='center')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=False, facecolor='#0e1117', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def generate_group_standings_png(results, selected_view):
    """Generate standing card tables for selected or all groups."""
    import matplotlib.patches as patches
    from src.group_simulator import WC2026_GROUPS
    
    if selected_view == "ALL GROUPS":
        groups = list(WC2026_GROUPS.keys())
        fig = Figure(figsize=(14, 22), facecolor='#0e1117')
        canvas = FigureCanvas(fig)
        axes = fig.subplots(6, 2).flatten()
    else:
        group_name = selected_view.replace("GROUP ", "")
        groups = [group_name]
        fig = Figure(figsize=(8, 3.66), facecolor='#0e1117')  # aspect ratio similar to card
        canvas = FigureCanvas(fig)
        ax = fig.subplots()
        axes = [ax]
        
    for idx, group_name in enumerate(groups):
        ax = axes[idx]
        ax.set_facecolor('#0e1117')
        ax.axis('off')
        
        # Force axes limits to keep coordinate math exact
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        group_df = results[results["group"] == group_name].sort_values("advance_pct", ascending=False).reset_index(drop=True)
        
        # Group card visual container
        card_rect = patches.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor='#111827', edgecolor='#1f2937', lw=1, zorder=1)
        ax.add_patch(card_rect)
        
        # Text alignment
        ax.text(0.06, 0.86, f"GROUP {group_name}", color='#10b981', fontsize=14, fontweight='bold', va='center', zorder=2)
        ax.text(0.25, 0.86, "Group Stage Standing Telemetry", color='#888888', fontsize=8, fontweight='bold', va='center', zorder=2)
        
        ax.text(0.06, 0.74, "#", color='#888888', fontsize=9, fontweight='bold', va='center', zorder=2)
        ax.text(0.17, 0.74, "Team", color='#888888', fontsize=9, fontweight='bold', va='center', zorder=2)
        ax.text(0.55, 0.74, "Elo", color='#888888', fontsize=9, fontweight='bold', va='center', zorder=2)
        ax.text(0.75, 0.74, "Advance %", color='#888888', fontsize=9, fontweight='bold', va='center', zorder=2)
        
        ax.plot([0.06, 0.94], [0.71, 0.71], color='#1f2937', lw=1, zorder=2)
        
        y_pos = 0.58
        for r, row in group_df.iterrows():
            ax.text(0.06, y_pos, str(r+1), color='#888888', fontsize=10, fontweight='bold', va='center', zorder=2)
            
            # Fetch and draw flag
            flag_img = get_flag_image(row["team"])
            if flag_img:
                flag_arr = np.array(flag_img)
                imagebox = OffsetImage(flag_arr, zoom=0.25)
                ab = AnnotationBbox(imagebox, (0.13, y_pos), frameon=False, pad=0, xycoords='data', zorder=2)
                ax.add_artist(ab)
                
            ax.text(0.17, y_pos, row["team"][:18], color='#ffffff', fontsize=10, fontweight='bold', va='center', zorder=2)
            ax.text(0.55, y_pos, f"{row['elo']:.0f}", color='#10b981', fontsize=10, va='center', zorder=2)
            
            pct = row["advance_pct"]
            pct_color = '#10b981' if pct >= 60 else ('#e5c158' if pct >= 30 else '#f87171')
            ax.text(0.75, y_pos, f"{pct:.1f}%", color=pct_color, fontsize=10, fontweight='bold', va='center', zorder=2)
            
            ax.plot([0.06, 0.94], [y_pos - 0.07, y_pos - 0.07], color='#1f2937', lw=0.5, zorder=2)
            y_pos -= 0.14
            
    if selected_view == "ALL GROUPS":
        for i in range(len(groups), len(axes)):
            axes[i].axis('off')
            
    fig.suptitle("FIFA WORLD CUP 2026 - GROUP STANDINGS PROBABILITIES", color='#10b981', fontsize=16, fontweight='bold', y=0.96)
    fig.text(0.5, 0.02, "Based on 10,000 Monte Carlo Simulation Runs", color='#888888', fontsize=9, ha='center')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=False, facecolor='#0e1117', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def generate_progression_png(results):
    """Generate full 48-team progression leaderboard."""
    leaderboard = results.sort_values(
        ["champion_pct", "final_pct", "sf_pct", "qf_pct", "r16_pct", "r32_pct", "elo"],
        ascending=False
    ).reset_index(drop=True)
    
    fig = Figure(figsize=(18, 14), facecolor='#0e1117')
    canvas = FigureCanvas(fig)
    axes = fig.subplots(1, 2)
    fig.suptitle("TOURNAMENT PROGRESSION LEADERBOARD", color='#10b981', fontsize=18, fontweight='bold', y=0.96)
    
    for col_idx in range(2):
        ax = axes[col_idx]
        ax.set_facecolor('#0e1117')
        ax.axis('off')
        
        # Force axes limits to keep coordinate math exact
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Headers
        ax.text(0.02, 0.94, "Rank", color='#888888', fontsize=10, fontweight='bold', va='center')
        ax.text(0.13, 0.94, "Team (Elo)", color='#888888', fontsize=10, fontweight='bold', va='center')
        ax.text(0.40, 0.94, "R32%", color='#888888', fontsize=10, fontweight='bold', va='center')
        ax.text(0.50, 0.94, "R16%", color='#888888', fontsize=10, fontweight='bold', va='center')
        ax.text(0.60, 0.94, "QF%", color='#888888', fontsize=10, fontweight='bold', va='center')
        ax.text(0.70, 0.94, "SF%", color='#888888', fontsize=10, fontweight='bold', va='center')
        ax.text(0.80, 0.94, "Final%", color='#888888', fontsize=10, fontweight='bold', va='center')
        ax.text(0.90, 0.94, "Champ%", color='#10b981', fontsize=10, fontweight='bold', va='center')
        
        ax.plot([0.02, 0.98], [0.92, 0.92], color='#10b981', lw=1.5)
        
        start_rank = col_idx * 24
        y_pos = 0.88
        for i in range(24):
            rank = start_rank + i
            if rank >= len(leaderboard):
                break
            row = leaderboard.iloc[rank]
            
            ax.text(0.02, y_pos, f"#{rank+1}", color='#ffffff', fontsize=10, fontweight='bold', va='center')
            
            # Fetch and draw flag
            flag_img = get_flag_image(row["team"])
            if flag_img:
                flag_arr = np.array(flag_img)
                imagebox = OffsetImage(flag_arr, zoom=0.25)
                ab = AnnotationBbox(imagebox, (0.09, y_pos), frameon=False, pad=0, xycoords='data', zorder=2)
                ax.add_artist(ab)
                
            ax.text(0.13, y_pos, f"{row['team'][:12].strip()} ({row['elo']:.0f})", color='#ffffff', fontsize=10, va='center')
            ax.text(0.40, y_pos, f"{row['r32_pct']:.0f}%", color='#888888', fontsize=10, va='center')
            ax.text(0.50, y_pos, f"{row['r16_pct']:.0f}%", color='#888888', fontsize=10, va='center')
            ax.text(0.60, y_pos, f"{row['qf_pct']:.0f}%", color='#888888', fontsize=10, va='center')
            ax.text(0.70, y_pos, f"{row['sf_pct']:.0f}%", color='#888888', fontsize=10, va='center')
            ax.text(0.80, y_pos, f"{row['final_pct']:.0f}%", color='#888888', fontsize=10, va='center')
            
            champ_pct = row['champion_pct']
            champ_color = '#10b981' if champ_pct > 0 else '#888888'
            ax.text(0.90, y_pos, f"{champ_pct:.1f}%", color=champ_color, fontsize=10, fontweight='bold', va='center')
            
            ax.plot([0.02, 0.98], [y_pos - 0.0175, y_pos - 0.0175], color='#232b2b', lw=0.5)
            y_pos -= 0.035
            
    fig.text(0.5, 0.02, "Based on 10,000 Monte Carlo Simulation Runs", color='#888888', fontsize=10, ha='center')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=False, facecolor='#0e1117', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def generate_bracket_png(bracket_data):
    """Generate 32-team knockout bracket tree graphic."""
    import matplotlib.patches as patches
    
    fig = Figure(figsize=(20, 12), facecolor='#0e1117')
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.set_facecolor('#0e1117')
    ax.axis('off')
    
    x_coords = [0, 2.2, 4.4, 6.6, 8.8, 11.0]
    col_names = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final", "Champion"]
    
    for c, name in enumerate(col_names):
        ax.text(x_coords[c] + 0.5, 16.5, name.upper(), color='#10b981', fontsize=12, fontweight='bold', ha='center', va='center')
        ax.plot([x_coords[c], x_coords[c] + 1.0], [16.2, 16.2], color='#10b981', lw=1.5)
        
    def draw_match_box(x, y, team_a, score_a, team_b, score_b, winner, is_aet, is_pen, match_label):
        rect = patches.Rectangle((x, y - 0.45), 1.0, 0.9, facecolor='#111827', edgecolor='#374151', lw=1, zorder=3)
        ax.add_patch(rect)
        
        status = " (AET)" if is_aet else (" (PEN)" if is_pen else "")
        ax.text(x + 0.5, y + 0.3, f"{match_label}{status}", color='#888888', fontsize=7, fontweight='bold', ha='center', va='center', zorder=4)
        
        # Draw team A flag
        flag_a = get_flag_image(team_a)
        if flag_a:
            flag_arr_a = np.array(flag_a)
            imagebox_a = OffsetImage(flag_arr_a, zoom=0.2)
            ab_a = AnnotationBbox(imagebox_a, (x + 0.09, y + 0.05), frameon=False, pad=0, xycoords='data', zorder=4)
            ax.add_artist(ab_a)
            
        color_a = '#10b981' if winner == team_a else '#ffffff'
        font_weight_a = 'bold' if winner == team_a else 'normal'
        ax.text(x + 0.17, y + 0.05, team_a[:12], color=color_a, fontsize=8, fontweight=font_weight_a, va='center', zorder=4)
        ax.text(x + 0.92, y + 0.05, str(score_a), color=color_a, fontsize=8, fontweight='bold', ha='right', va='center', zorder=4)
        
        # Draw team B flag
        flag_b = get_flag_image(team_b)
        if flag_b:
            flag_arr_b = np.array(flag_b)
            imagebox_b = OffsetImage(flag_arr_b, zoom=0.2)
            ab_b = AnnotationBbox(imagebox_b, (x + 0.09, y - 0.20), frameon=False, pad=0, xycoords='data', zorder=4)
            ax.add_artist(ab_b)
            
        color_b = '#10b981' if winner == team_b else '#ffffff'
        font_weight_b = 'bold' if winner == team_b else 'normal'
        ax.text(x + 0.17, y - 0.20, team_b[:12], color=color_b, fontsize=8, fontweight=font_weight_b, va='center', zorder=4)
        ax.text(x + 0.92, y - 0.20, str(score_b), color=color_b, fontsize=8, fontweight='bold', ha='right', va='center', zorder=4)
        
    r32_centers = list(range(16))
    r16_centers = [sum(r32_centers[2*i:2*i+2])/2 for i in range(8)]
    qf_centers = [sum(r16_centers[2*i:2*i+2])/2 for i in range(4)]
    sf_centers = [sum(qf_centers[2*i:2*i+2])/2 for i in range(2)]
    final_center = sum(sf_centers)/2
    
    r32_matches = bracket_data["knockouts"]["r32"]
    for i, m in enumerate(r32_matches):
        draw_match_box(x_coords[0], r32_centers[i], m["team_a"], m["score_a"], m["team_b"], m["score_b"], m["winner"], m["extra_time"], m["penalties"], f"R32 Match {i+1}")
        
    r16_matches = bracket_data["knockouts"]["r16"]
    for i, m in enumerate(r16_matches):
        draw_match_box(x_coords[1], r16_centers[i], m["team_a"], m["score_a"], m["team_b"], m["score_b"], m["winner"], m["extra_time"], m["penalties"], f"R16 Match {i+1}")
        for child_idx in [2*i, 2*i+1]:
            child_y = r32_centers[child_idx]
            parent_y = r16_centers[i]
            ax.plot([x_coords[0] + 1.0, x_coords[0] + 1.6, x_coords[0] + 1.6, x_coords[1]], 
                    [child_y, child_y, parent_y, parent_y], color='#374151', lw=1, zorder=1)
            
    qf_matches = bracket_data["knockouts"]["qf"]
    for i, m in enumerate(qf_matches):
        draw_match_box(x_coords[2], qf_centers[i], m["team_a"], m["score_a"], m["team_b"], m["score_b"], m["winner"], m["extra_time"], m["penalties"], f"QF Match {i+1}")
        for child_idx in [2*i, 2*i+1]:
            child_y = r16_centers[child_idx]
            parent_y = qf_centers[i]
            ax.plot([x_coords[1] + 1.0, x_coords[1] + 1.6, x_coords[1] + 1.6, x_coords[2]], 
                    [child_y, child_y, parent_y, parent_y], color='#374151', lw=1, zorder=1)
            
    sf_matches = bracket_data["knockouts"]["sf"]
    for i, m in enumerate(sf_matches):
        draw_match_box(x_coords[3], sf_centers[i], m["team_a"], m["score_a"], m["team_b"], m["score_b"], m["winner"], m["extra_time"], m["penalties"], f"SF Match {i+1}")
        for child_idx in [2*i, 2*i+1]:
            child_y = qf_centers[child_idx]
            parent_y = sf_centers[i]
            ax.plot([x_coords[2] + 1.0, x_coords[2] + 1.6, x_coords[2] + 1.6, x_coords[3]], 
                    [child_y, child_y, parent_y, parent_y], color='#374151', lw=1, zorder=1)
            
    final_m = bracket_data["knockouts"]["final"]
    draw_match_box(x_coords[4], final_center, final_m["team_a"], final_m["score_a"], final_m["team_b"], final_m["score_b"], final_m["winner"], final_m["extra_time"], final_m["penalties"], "World Cup Final")
    for child_idx in [0, 1]:
        child_y = sf_centers[child_idx]
        parent_y = final_center
        ax.plot([x_coords[3] + 1.0, x_coords[3] + 1.6, x_coords[3] + 1.6, x_coords[4]], 
                [child_y, child_y, parent_y, parent_y], color='#374151', lw=1, zorder=1)
        
    champ = final_m["winner"]
    rect = patches.Rectangle((x_coords[5], final_center - 0.5), 1.0, 1.0, facecolor='#10b981', alpha=0.08, edgecolor='#10b981', lw=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x_coords[5] + 0.5, final_center + 0.25, "★ CHAMPION ★", color='#10b981', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)
    
    # Draw champion flag (perfectly centered vertically at final_center)
    flag_champ = get_flag_image(champ)
    if flag_champ:
        flag_arr_champ = np.array(flag_champ)
        imagebox_champ = OffsetImage(flag_arr_champ, zoom=0.35)
        ab_champ = AnnotationBbox(imagebox_champ, (x_coords[5] + 0.5, final_center), frameon=False, pad=0, xycoords='data', zorder=4)
        ax.add_artist(ab_champ)
        
    ax.text(x_coords[5] + 0.5, final_center - 0.25, champ.upper(), color='#10b981', fontsize=12, fontweight='bold', ha='center', va='center', zorder=4)
    
    ax.plot([x_coords[4] + 1.0, x_coords[5]], [final_center, final_center], color='#10b981', lw=1.5, zorder=1)
    
    ax.set_ylim(-1, 17)
    ax.set_xlim(-0.5, 12.5)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=False, facecolor='#0e1117', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()
