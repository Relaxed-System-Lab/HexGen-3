#!/usr/bin/env python3
"""
Visualize an analyzer.json workflow as a directed graph.

Creates a clustered graph (one cluster per phase, e.g. "prefill", "decode") where each
component (q_proj, k_proj, etc.) is a node annotated with OPs / AI / time. Nodes are
sized by OPs and colored by the reported 'bound' (compute vs memory).

Primary renderer: graphviz (recommended). Fallback: networkx + matplotlib.

Usage:
  python tools/visualize_analyzer.py \
    --input .local/output/analyzer.json \
    --output .local/output/analyzer_workflow --format png

Dependencies (install if missing): graphviz, networkx, matplotlib
Note: The `graphviz` system binary must be installed to render with the graphviz backend.
"""

import argparse
import json
import math
import os
import sys

try:
    import graphviz
    GRAPHVIZ_PY_AVAILABLE = True
except Exception:
    GRAPHVIZ_PY_AVAILABLE = False

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False


def human_format(num):
    """Return compact human-readable number (e.g., 1.3M, 2.5G)."""
    try:
        num = float(num)
    except Exception:
        return str(num)
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%3.1f%s" % (num, "P")


COLOR_MAP = {
    "compute": "#8dd3c7",  # teal
    "memory": "#fdb462",   # orange
}


def build_graphviz(data, outpath, fmt="png", engine="dot"):
    g = graphviz.Digraph(name="LLM_workflow", format=fmt, engine=engine)
    g.attr(rankdir='LR')

    # Create clusters for each top-level phase
    for phase, comps in data.items():
        cluster_name = f"cluster_{phase}"
        with g.subgraph(name=cluster_name) as c:
            c.attr(style='filled', color='lightgrey')
            c.attr(label=phase)
            # Add nodes
            for comp, metrics in comps.items():
                node_id = f"{phase}__{comp}"
                ops = metrics.get("OPs")
                ai = metrics.get("arithmetic_intensity")
                itime = metrics.get("inference_time")
                bound = metrics.get("bound", "")
                label = f"{comp}\\nOPs: {human_format(ops)}\\nAI: {ai:.2f}\\nTime: {itime:.6g}s"
                size = max(0.35, math.log10(ops + 1) / 6 if ops and ops > 0 else 0.35)
                fillcolor = COLOR_MAP.get(bound, "#dddddd")
                c.node(node_id, label=label, style='filled', fillcolor=fillcolor, shape='box', width=str(size), fontsize='10')

    # Add cross-phase edges for components that exist in multiple phases (e.g., q_proj)
    phases = list(data.keys())
    for i in range(len(phases) - 1):
        p_from = phases[i]
        for p_to in phases[i+1:]:
            # connect same-named components from earlier phase to later phase
            for comp in data[p_from].keys():
                if comp in data[p_to]:
                    g.edge(f"{p_from}__{comp}", f"{p_to}__{comp}")

    # Render
    outdir = os.path.dirname(outpath)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # graphviz.render wants filename without extension for some usages; render via .render
    rendered_path = g.render(filename=outpath, cleanup=True)
    return rendered_path


def build_networkx(data, outpath, fmt="png"):
    if not NX_AVAILABLE:
        raise RuntimeError("networkx/matplotlib not available for fallback drawing")

    G = nx.DiGraph()
    # add nodes with attributes
    for phase, comps in data.items():
        for comp, metrics in comps.items():
            node_id = f"{phase}__{comp}"
            G.add_node(node_id, phase=phase, comp=comp, metrics=metrics)

    # add cross-phase edges
    phases = list(data.keys())
    for i in range(len(phases) - 1):
        p_from = phases[i]
        for p_to in phases[i+1:]:
            for comp in data[p_from].keys():
                if comp in data[p_to]:
                    G.add_edge(f"{p_from}__{comp}", f"{p_to}__{comp}")

    # layout
    pos = nx.spring_layout(G, seed=42)

    # node sizes and colors
    ops_list = [n[1].get('metrics', {}).get('OPs', 1) for n in G.nodes(data=True)]
    min_ops = min(ops_list) if ops_list else 1
    max_ops = max(ops_list) if ops_list else 1
    sizes = [300 + 3000 * (math.log10((ops or 1) + 1) - math.log10(min_ops + 1)) / (math.log10(max_ops + 1) - math.log10(min_ops + 1) + 1e-9) for ops in ops_list]

    colors = []
    for _, d in G.nodes(data=True):
        bound = d.get('metrics', {}).get('bound', '')
        colors.append(COLOR_MAP.get(bound, '#cccccc'))

    # labels
    labels = {}
    for n, d in G.nodes(data=True):
        m = d.get('metrics', {})
        labels[n] = f"{d.get('comp')}\nOPs:{human_format(m.get('OPs'))}\nAI:{m.get('arithmetic_intensity',0):.2f}\n{m.get('inference_time',0):.6g}s"

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=12, edge_color='#888888')
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, linewidths=0.5, edgecolors='k')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.axis('off')

    outfile = f"{outpath}.{fmt}" if not outpath.endswith(f".{fmt}") else outpath
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    return outfile


def load_analyzer_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    # Expecting top-level keys like 'decode' and 'prefill'; if the file contains a wrapper (like 'total_results') we try to use that
    if not any(isinstance(v, dict) for v in obj.values()):
        # probably wrapped; try common keys
        for candidate in ['decode', 'prefill', 'total_results']:
            if candidate in obj:
                # if total_results contains nested decode/prefill
                if isinstance(obj[candidate], dict) and any(isinstance(v, dict) for v in obj[candidate].values()):
                    return obj[candidate]
        # fallback: return whole object
        return obj
    return obj


def main():
    parser = argparse.ArgumentParser(description='Visualize analyzer.json as workflow graph')
    parser.add_argument('--input', '-i', default='.local/output/analyzer.json', help='Path to analyzer.json')
    parser.add_argument('--output', '-o', default='.local/output/analyzer_workflow', help='Output path (no extension) or with extension')
    parser.add_argument('--format', '-f', default='png', help='Output image format (png, pdf, svg)')
    parser.add_argument('--engine', default='dot', help='graphviz layout engine (dot, neato, sfdp, twopi, circo)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("\nThis script visualizes roofline analyzer results.")
        print("To generate the analyzer.json file, run:")
        print("  python cli/run_analyzer.py")
        print("\nFor scheduling experiment results, use:")
        print("  python cli/visualization/plot_results.py")
        sys.exit(2)

    data = load_analyzer_json(args.input)

    # prefer graphviz
    if GRAPHVIZ_PY_AVAILABLE:
        try:
            print("Rendering with graphviz...")
            out = build_graphviz(data, args.output, fmt=args.format, engine=args.engine)
            print(f"Wrote: {out}")
            return
        except Exception as e:
            print(f"Graphviz render failed: {e}")

    # fallback to networkx + matplotlib
    try:
        print("Falling back to networkx+matplotlib rendering...")
        out = build_networkx(data, args.output, fmt=args.format)
        print(f"Wrote: {out}")
        return
    except Exception as e:
        print(f"Fallback rendering failed: {e}")
        print("To enable nicer output, install the 'graphviz' Python package and the Graphviz system binary, or ensure 'networkx' and 'matplotlib' are installed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
