import matplotlib.pyplot as plt
import numpy as np
import io

def generate_matchup_png(home_team, away_team, home_elo, away_elo, hw, dr, aw, predicted, confidence):
    """Generate matchup prediction infographic."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.9, "MATCH DECISION TELEMETRY", color='#10b981', fontsize=12, fontweight='bold', ha='center')
    fig.text(0.5, 0.82, f"{home_team.upper()} vs {away_team.upper()}", color='#ffffff', fontsize=20, fontweight='bold', ha='center')
    fig.text(0.5, 0.75, f"Elo {home_elo:.0f}  |  Elo {away_elo:.0f}", color='#888888', fontsize=12, ha='center')
    
    # Draw Probability Bar
    bar_y = 0.45
    bar_height = 0.1
    
    hw_width = hw
    dr_width = dr
    aw_width = aw
    
    rect_hw = plt.Rectangle((0, bar_y), hw_width, bar_height, color='#38bdf8')
    rect_dr = plt.Rectangle((hw_width, bar_y), dr_width, bar_height, color='#e5c158')
    rect_aw = plt.Rectangle((hw_width + dr_width, bar_y), aw_width, bar_height, color='#f87171')
    
    ax.add_patch(rect_hw)
    ax.add_patch(rect_dr)
    ax.add_patch(rect_aw)
    
    # Labels
    if hw_width > 0.08:
        fig.text(hw_width / 2, bar_y + 0.04, f"{hw_width*100:.1f}%\nHOME WIN", color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='center')
    if dr_width > 0.08:
        fig.text(hw_width + dr_width / 2, bar_y + 0.04, f"{dr_width*100:.1f}%\nDRAW", color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='center')
    if aw_width > 0.08:
        fig.text(hw_width + dr_width + aw_width / 2, bar_y + 0.04, f"{aw_width*100:.1f}%\nAWAY WIN", color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='center')
        
    # Decision Banner
    decision_text = f"MODEL DECISION: {predicted.upper()} (CONFIDENCE: {confidence*100:.1f}%)"
    fig.text(0.5, 0.25, decision_text, color='#10b981', fontsize=12, fontweight='bold', ha='center',
             bbox=dict(facecolor='#111827', edgecolor='#10b981', boxstyle='round,pad=0.8'))
             
    fig.text(0.5, 0.08, "FIFA World Cup Match Outcome Predictor Engine", color='#4b5563', fontsize=8, ha='center')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_group_standings_png(results, selected_view):
    """Generate standing card tables for selected or all groups."""
    from src.group_simulator import WC2026_GROUPS
    
    if selected_view == "ALL GROUPS":
        groups = list(WC2026_GROUPS.keys())
        fig, axes = plt.subplots(6, 2, figsize=(14, 22), facecolor='#0e1117')
        axes = axes.flatten()
    else:
        group_name = selected_view.replace("GROUP ", "")
        groups = [group_name]
        fig, ax = plt.subplots(figsize=(8, 550/150), facecolor='#0e1117')  # aspect ratio similar to card
        axes = [ax]
        
    for idx, group_name in enumerate(groups):
        ax = axes[idx]
        ax.set_facecolor('#0e1117')
        ax.axis('off')
        
        group_df = results[results["group"] == group_name].sort_values("advance_pct", ascending=False).reset_index(drop=True)
        
        # Group card visual container
        card_rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor='#111827', edgecolor='#1f2937', lw=1, zorder=1)
        ax.add_patch(card_rect)
        
        # Text alignment
        ax.text(0.06, 0.86, f"GROUP {group_name}", color='#10b981', fontsize=14, fontweight='bold', zorder=2)
        ax.text(0.25, 0.86, "Group Stage Standing Telemetry", color='#888888', fontsize=8, fontweight='bold', zorder=2)
        
        ax.text(0.06, 0.74, "#", color='#888888', fontsize=9, fontweight='bold', zorder=2)
        ax.text(0.12, 0.74, "Team", color='#888888', fontsize=9, fontweight='bold', zorder=2)
        ax.text(0.55, 0.74, "Elo", color='#888888', fontsize=9, fontweight='bold', zorder=2)
        ax.text(0.75, 0.74, "Advance %", color='#888888', fontsize=9, fontweight='bold', zorder=2)
        
        ax.plot([0.06, 0.94], [0.71, 0.71], color='#1f2937', lw=1, zorder=2)
        
        y_pos = 0.58
        for r, row in group_df.iterrows():
            ax.text(0.06, y_pos, str(r+1), color='#888888', fontsize=10, fontweight='bold', zorder=2)
            ax.text(0.12, y_pos, row["team"][:18], color='#ffffff', fontsize=10, fontweight='bold', zorder=2)
            ax.text(0.55, y_pos, f"{row['elo']:.0f}", color='#10b981', fontsize=10, zorder=2)
            
            pct = row["advance_pct"]
            pct_color = '#10b981' if pct >= 60 else ('#e5c158' if pct >= 30 else '#f87171')
            ax.text(0.75, y_pos, f"{pct:.1f}%", color=pct_color, fontsize=10, fontweight='bold', zorder=2)
            
            ax.plot([0.06, 0.94], [y_pos - 0.05, y_pos - 0.05], color='#1f2937', lw=0.5, zorder=2)
            y_pos -= 0.14
            
    if selected_view == "ALL GROUPS":
        for i in range(len(groups), len(axes)):
            axes[i].axis('off')
            
    fig.suptitle("FIFA WORLD CUP 2026 - GROUP STANDINGS PROBABILITIES", color='#10b981', fontsize=16, fontweight='bold', y=0.96)
    fig.text(0.5, 0.02, "Based on 10,000 Monte Carlo Simulation Runs", color='#888888', fontsize=9, ha='center')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_progression_png(results):
    """Generate full 48-team progression leaderboard."""
    leaderboard = results.sort_values(
        ["champion_pct", "final_pct", "sf_pct", "qf_pct", "r16_pct", "r32_pct", "elo"],
        ascending=False
    ).reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 14), facecolor='#0e1117')
    fig.suptitle("TOURNAMENT PROGRESSION LEADERBOARD", color='#10b981', fontsize=18, fontweight='bold', y=0.96)
    
    for col_idx in range(2):
        ax = axes[col_idx]
        ax.set_facecolor('#0e1117')
        ax.axis('off')
        
        # Headers
        ax.text(0.02, 0.94, "Rank", color='#888888', fontsize=10, fontweight='bold')
        ax.text(0.08, 0.94, "Team (Elo)", color='#888888', fontsize=10, fontweight='bold')
        ax.text(0.40, 0.94, "R32%", color='#888888', fontsize=10, fontweight='bold')
        ax.text(0.50, 0.94, "R16%", color='#888888', fontsize=10, fontweight='bold')
        ax.text(0.60, 0.94, "QF%", color='#888888', fontsize=10, fontweight='bold')
        ax.text(0.70, 0.94, "SF%", color='#888888', fontsize=10, fontweight='bold')
        ax.text(0.80, 0.94, "Final%", color='#888888', fontsize=10, fontweight='bold')
        ax.text(0.90, 0.94, "Champ%", color='#10b981', fontsize=10, fontweight='bold')
        
        ax.plot([0.02, 0.98], [0.92, 0.92], color='#10b981', lw=1.5)
        
        start_rank = col_idx * 24
        y_pos = 0.88
        for i in range(24):
            rank = start_rank + i
            if rank >= len(leaderboard):
                break
            row = leaderboard.iloc[rank]
            
            ax.text(0.02, y_pos, f"#{rank+1}", color='#ffffff', fontsize=10, fontweight='bold')
            ax.text(0.08, y_pos, f"{row['team'][:16]} ({row['elo']:.0f})", color='#ffffff', fontsize=10)
            ax.text(0.40, y_pos, f"{row['r32_pct']:.0f}%", color='#888888', fontsize=10)
            ax.text(0.50, y_pos, f"{row['r16_pct']:.0f}%", color='#888888', fontsize=10)
            ax.text(0.60, y_pos, f"{row['qf_pct']:.0f}%", color='#888888', fontsize=10)
            ax.text(0.70, y_pos, f"{row['sf_pct']:.0f}%", color='#888888', fontsize=10)
            ax.text(0.80, y_pos, f"{row['final_pct']:.0f}%", color='#888888', fontsize=10)
            
            champ_pct = row['champion_pct']
            champ_color = '#10b981' if champ_pct > 0 else '#888888'
            ax.text(0.90, y_pos, f"{champ_pct:.1f}%", color=champ_color, fontsize=10, fontweight='bold')
            
            ax.plot([0.02, 0.98], [y_pos - 0.015, y_pos - 0.015], color='#232b2b', lw=0.5)
            y_pos -= 0.035
            
    fig.text(0.5, 0.02, "Calculated across 10,000 Monte Carlo Simulation Runs", color='#888888', fontsize=10, ha='center')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_bracket_png(bracket_data):
    """Generate 32-team knockout bracket tree graphic."""
    fig, ax = plt.subplots(figsize=(20, 12), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    ax.axis('off')
    
    x_coords = [0, 2.2, 4.4, 6.6, 8.8, 11.0]
    col_names = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final", "Champion"]
    
    for c, name in enumerate(col_names):
        ax.text(x_coords[c] + 0.5, 16.5, name.upper(), color='#10b981', fontsize=12, fontweight='bold', ha='center')
        ax.plot([x_coords[c], x_coords[c] + 1.0], [16.2, 16.2], color='#10b981', lw=1.5)
        
    def draw_match_box(x, y, team_a, score_a, team_b, score_b, winner, is_aet, is_pen, match_label):
        rect = plt.Rectangle((x, y - 0.45), 1.0, 0.9, facecolor='#111827', edgecolor='#374151', lw=1, zorder=3)
        ax.add_patch(rect)
        
        status = " (AET)" if is_aet else (" (PEN)" if is_pen else "")
        ax.text(x + 0.5, y + 0.3, f"{match_label}{status}", color='#888888', fontsize=7, fontweight='bold', ha='center', zorder=4)
        
        color_a = '#10b981' if winner == team_a else '#ffffff'
        font_weight_a = 'bold' if winner == team_a else 'normal'
        ax.text(x + 0.08, y + 0.04, team_a[:14], color=color_a, fontsize=8, fontweight=font_weight_a, zorder=4)
        ax.text(x + 0.92, y + 0.04, str(score_a), color=color_a, fontsize=8, fontweight='bold', ha='right', zorder=4)
        
        color_b = '#10b981' if winner == team_b else '#ffffff'
        font_weight_b = 'bold' if winner == team_b else 'normal'
        ax.text(x + 0.08, y - 0.22, team_b[:14], color=color_b, fontsize=8, fontweight=font_weight_b, zorder=4)
        ax.text(x + 0.92, y - 0.22, str(score_b), color=color_b, fontsize=8, fontweight='bold', ha='right', zorder=4)
        
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
    rect = plt.Rectangle((x_coords[5], final_center - 0.5), 1.0, 1.0, facecolor='#10b981', alpha=0.08, edgecolor='#10b981', lw=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x_coords[5] + 0.5, final_center + 0.25, "★ CHAMPION ★", color='#10b981', fontsize=9, fontweight='bold', ha='center', zorder=4)
    ax.text(x_coords[5] + 0.5, final_center - 0.15, champ.upper(), color='#10b981', fontsize=12, fontweight='bold', ha='center', zorder=4)
    
    ax.plot([x_coords[4] + 1.0, x_coords[5]], [final_center, final_center], color='#10b981', lw=1.5, zorder=1)
    
    ax.set_ylim(-1, 17)
    ax.set_xlim(-0.5, 12.5)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
