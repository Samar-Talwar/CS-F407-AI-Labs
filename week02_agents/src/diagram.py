# CS F407 Lab, Week 2 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Generate block diagram of the Goal-Based Agent Architecture."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def generate_agent_diagram(output_path: Path | str) -> None:
    """Draw a clean, publication-ready block diagram of the Goal-Based Agent architecture.

    Args:
        output_path: Path where the resulting PNG will be saved.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    bg_color = "#F8FAFC"
    border_color = "#334155"
    agent_bg = "#EFF6FF"
    env_bg = "#FEF3C7"
    box_blue = "#DBEAFE"
    box_green = "#DCFCE7"
    box_purple = "#F3E8FF"
    box_orange = "#FFEDD5"
    text_color = "#0F172A"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Title
    ax.text(
        5.5,
        6.6,
        "Goal-Based Agent Architecture (Warehouse Navigation)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )

    # Outer Boxes: Environment and Agent
    # Environment Box (Bottom)
    env_rect = patches.FancyBboxPatch(
        (0.8, 0.5),
        9.4,
        1.5,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor=env_bg,
        edgecolor=border_color,
        linewidth=1.5,
    )
    ax.add_patch(env_rect)
    ax.text(
        1.2,
        1.75,
        "ENVIRONMENT (Warehouse 2D Grid)",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#92400E",
    )
    ax.text(
        5.5,
        1.1,
        "Grid Layout: Shelving Units (Walls '#'), Free Space ('.'), Start ('S'), Goal ('G')\n"
        "State space: Discrete grid coordinates (r, c) | Deterministic & Fully Observable",
        ha="center",
        va="center",
        fontsize=9,
        color=text_color,
    )

    # Agent Box (Top)
    agent_rect = patches.FancyBboxPatch(
        (0.8, 2.4),
        9.4,
        3.8,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor=agent_bg,
        edgecolor=border_color,
        linewidth=1.5,
    )
    ax.add_patch(agent_rect)
    ax.text(
        1.2,
        5.9,
        "GOAL-BASED AGENT",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1E40AF",
    )

    # Internal Component Boxes inside Agent
    # 1. Sensors (Percept)
    sens_rect = patches.FancyBboxPatch(
        (1.2, 2.7),
        2.2,
        0.9,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        facecolor=box_blue,
        edgecolor="#3B82F6",
        linewidth=1.2,
    )
    ax.add_patch(sens_rect)
    ax.text(
        2.3,
        3.15,
        "Sensors / Percept\n(Current Pos (r, c))",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=text_color,
    )

    # 2. State Maintenance
    state_rect = patches.FancyBboxPatch(
        (1.2, 4.3),
        2.2,
        1.1,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        facecolor=box_purple,
        edgecolor="#8B5CF6",
        linewidth=1.2,
    )
    ax.add_patch(state_rect)
    ax.text(
        2.3,
        4.85,
        "Agent State\n• Pos (r, c)\n• Environment Map\n• Explored Set",
        ha="center",
        va="center",
        fontsize=8.5,
        color=text_color,
    )

    # 3. Goal Component
    goal_rect = patches.FancyBboxPatch(
        (4.4, 4.3),
        2.2,
        1.1,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        facecolor=box_green,
        edgecolor="#10B981",
        linewidth=1.2,
    )
    ax.add_patch(goal_rect)
    ax.text(
        5.5,
        4.85,
        "Goal Specification\n• Target: Goal 'G'\n• GoalTest(state)\n  is_goal == True",
        ha="center",
        va="center",
        fontsize=8.5,
        color=text_color,
    )

    # 4. Decision Component (Search Planner)
    dec_rect = patches.FancyBboxPatch(
        (4.4, 2.7),
        2.2,
        1.2,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        facecolor=box_orange,
        edgecolor="#F97316",
        linewidth=1.4,
    )
    ax.add_patch(dec_rect)
    ax.text(
        5.5,
        3.3,
        "Decision Component\n(Search Planner)\n• BFS / Queue\n• Path Reconstruction",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=text_color,
    )

    # 5. Actions / Actuators
    act_rect = patches.FancyBboxPatch(
        (7.6, 2.7),
        2.2,
        1.2,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        facecolor=box_blue,
        edgecolor="#3B82F6",
        linewidth=1.2,
    )
    ax.add_patch(act_rect)
    ax.text(
        8.7,
        3.3,
        "Actuators / Actions\n• Up, Down, Left, Right\n• Wall-collision check\n• Execute step",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=text_color,
    )

    # Arrows indicating information flow
    arrow_props = dict(
        arrowstyle="-|>",
        color="#1E293B",
        lw=1.5,
        mutation_scale=15,
    )

    # Env to Sensors
    ax.annotate(
        "",
        xy=(2.3, 2.7),
        xytext=(2.3, 2.0),
        arrowprops=arrow_props,
    )
    ax.text(2.4, 2.35, "Percepts", fontsize=8, color="#475569")

    # Sensors to State
    ax.annotate(
        "",
        xy=(2.3, 4.3),
        xytext=(2.3, 3.6),
        arrowprops=arrow_props,
    )
    ax.text(2.4, 3.95, "Update", fontsize=8, color="#475569")

    # State to Decision Component
    ax.annotate(
        "",
        xy=(4.4, 3.5),
        xytext=(3.4, 4.4),
        arrowprops=arrow_props,
    )
    ax.text(3.4, 3.8, "Current State", fontsize=8, color="#475569")

    # Goal to Decision Component
    ax.annotate(
        "",
        xy=(5.5, 3.9),
        xytext=(5.5, 4.3),
        arrowprops=arrow_props,
    )
    ax.text(5.6, 4.05, "Objective", fontsize=8, color="#475569")

    # Decision Component to Actuators
    ax.annotate(
        "",
        xy=(7.6, 3.3),
        xytext=(6.6, 3.3),
        arrowprops=arrow_props,
    )
    ax.text(6.7, 3.45, "Action Plan", fontsize=8, color="#475569")

    # Actuators to Environment
    ax.annotate(
        "",
        xy=(8.7, 2.0),
        xytext=(8.7, 2.7),
        arrowprops=arrow_props,
    )
    ax.text(8.8, 2.35, "Action (Move)", fontsize=8, color="#475569")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
