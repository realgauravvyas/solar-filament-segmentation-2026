"""Write every number of the report from the result files, so the text cannot drift from the data.

    python report/make_numbers.py --public 0.40

Reads ``results/metrics.json`` (stage 8), ``results/selection.json`` (stage 5),
``results/ablation.json`` (stage 9) and ``results/threshold_by_group.json`` (stage 11); writes ``report/numbers.tex``,
``report/table_main.tex``, ``report/table_ablation.tex`` and ``report/table_sites.tex``.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from filaseg import data  # noqa: E402
from filaseg.candidates import FEATURE_NAMES  # noqa: E402


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--public", required=True, help="public leaderboard score of the final submission")
    parser.add_argument("--inference-minutes", default="60", help="measured inference time on the test set")
    parser.add_argument("--inference-gpu", default="one Tesla T4", help="hardware of that measurement")
    parser.add_argument("--detectors", default=(
        "a YOLO11m-seg at $1536^2$ pixels (80 epochs) and a YOLO11s-seg at $1280^2$ pixels (60 epochs)"
    ))
    args = parser.parse_args()

    results = REPO / "results"
    metrics = json.load(open(results / "metrics.json"))
    selection = json.load(open(results / "selection.json"))
    ablation = json.load(open(results / "ablation.json"))
    by_group = json.load(open(results / "threshold_by_group.json"))["groups"]
    overall, human = metrics["overall"], metrics["annotator_vs_annotator"]
    report = selection["report"]

    observations = data.load_observations()
    n_views = sum(len(o.views) for o in observations)
    n_drawings = sum(len(v.polygons) for o in observations for v in o.views)
    n_multi = sum(1 for o in observations if len(o.views) > 1)

    full_pq = report["scorer"]["pq"]
    numbers = "".join([
        macro("pqAll", f"{overall['pq']:.3f}"),
        macro("pqAllLow", f"{overall['ci95']['pq'][0]:.3f}"),
        macro("pqAllHigh", f"{overall['ci95']['pq'][1]:.3f}"),
        macro("sqAll", f"{overall['sq']:.2f}"),
        macro("pqHuman", f"{human['pq']:.3f}"),
        macro("thirdOne", f"{metrics['predicting_the_third_annotator']['one_annotator']['pq']:.2f}"),
        macro("thirdTwo", f"{metrics['predicting_the_third_annotator']['two_annotators_agree']['pq']:.2f}"),
        macro("thirdMethod", f"{metrics['predicting_the_third_annotator']['method']['pq']:.2f}"),
        macro("thirdFrames", str(metrics['predicting_the_third_annotator']['method']['frames'])),
        macro("dqHuman", f"{human['dq']:.2f}"),
        macro("sqHuman", f"{human['sq']:.2f}"),
        macro("pqPublic", args.public),
        macro("groupMin", f"{min(r['pq'] for r in metrics['by_annotation_group'].values()):.2f}"),
        macro("groupMax", f"{max(r['pq'] for r in metrics['by_annotation_group'].values()):.2f}"),
        macro("nGroups", str(len(metrics['by_annotation_group']))),
        macro("siteMin", f"{min(r['pq'] for r in metrics['by_site'].values()):.2f}"),
        macro("siteMax", f"{max(r['pq'] for r in metrics['by_site'].values()):.2f}"),
        macro("inferenceMinutes", args.inference_minutes),
        macro("inferenceGpu", args.inference_gpu),
        macro("pqOracle", f"{report['oracle']['pq']:.2f}"),
        macro("tauChosen", f"{selection['value_threshold']:.2f}"),
        macro("tauGroupMin", f"{min(g['best_threshold'] for g in by_group.values()):.2f}"),
        macro("tauGroupMax", f"{max(g['best_threshold'] for g in by_group.values()):.2f}"),
        macro("tauGroupLoss", f"{max(g['loss'] for g in by_group.values()):.3f}"),
        macro("nFrames", str(len(observations))),
        macro("nViews", f"{n_views:,}".replace(",", "\\,")),
        macro("nDrawings", f"{n_drawings:,}".replace(",", "\\,")),
        macro("nMulti", str(n_multi)),
        macro("nFeatures", str(len(FEATURE_NAMES))),
        macro("detectorList", args.detectors),
        macro("ablationShapeGain", f"{ablation['claim_fused']['pq'] - ablation['detector_masks']['pq']:.2f}"),
        macro("missedShare", f"{100 * (1 - report['candidate_recall']):.0f}\\,\\%"),
        macro("outlineMinus", f"{ablation['outline_as_predicted']['pq'] - ablation['outline_minus_1px']['pq']:.2f}"),
        macro("outlinePlus", f"{ablation['outline_as_predicted']['pq'] - ablation['outline_plus_1px']['pq']:.2f}"),
        macro("ablationText", (
            f"Connected components of the U-Net output, with no detector, reach "
            f"{ablation['unet_components']['pq']:.3f}, and the detector masks, with no U-Net, "
            f"{ablation['detector_masks']['pq']:.3f}; U-Net pixels grouped by one detector reach "
            f"{ablation['claim_one_view']['pq']:.3f}. Fusing two detectors and four mirror views adds "
            f"{ablation['claim_fused']['pq'] - ablation['claim_one_view']['pq']:.3f}, and the three "
            f"probability cuts with the learned selector add a further "
            f"{full_pq - ablation['claim_fused']['pq']:.3f}."
        )),
    ])
    (REPO / "report" / "numbers.tex").write_text(numbers, encoding="utf-8")

    def row(label: str, r: dict) -> str:
        low, high = r["ci95"]["pq"]
        return (f"        {label} & {r['frames']} & {r['pq']:.3f} [{low:.3f}, {high:.3f}] & "
                f"{r['dq']:.3f} & {r['sq']:.3f} \\\\\n")

    table = (
        "    \\begin{tabular}{lrccc}\n        \\toprule\n"
        "        Frames & $n$ & PQ [95\\,\\% interval] & DQ & SQ \\\\\n        \\midrule\n"
    )
    table += row("All", overall)
    for k, r in metrics["by_number_of_annotators"].items():
        table += row(f"{k} annotator{'s' if k != '1' else ''}", r)
    table += "        \\midrule\n" + row("Annotator vs.\\ annotators", human)
    table += "        \\bottomrule\n    \\end{tabular}\n"
    (REPO / "report" / "table_main.tex").write_text(table, encoding="utf-8")

    sites = {"Bh": "Big Bear", "Ch": "Cerro Tololo", "Lh": "Learmonth", "Mh": "Mauna Loa",
             "Th": "El Teide", "Uh": "Udaipur"}
    site_rows = "".join(
        f"        {sites.get(code, code)} & {r['frames']} & {r['pq']:.3f} [{r['ci95']['pq'][0]:.3f}, "
        f"{r['ci95']['pq'][1]:.3f}] \\\\\n"
        for code, r in metrics["by_site"].items()
    )
    (REPO / "report" / "table_sites.tex").write_text(
        "    \\begin{tabular}{lrc}\n        \\toprule\n"
        "        GONG site & $n$ & PQ [95\\,\\% interval] \\\\\n        \\midrule\n"
        + site_rows + "        \\bottomrule\n    \\end{tabular}\n",
        encoding="utf-8",
    )

    lines = [
        ("U-Net connected components (no detector)", ablation["unet_components"]["pq"]),
        ("Detector masks (no U-Net)", ablation["detector_masks"]["pq"]),
        ("U-Net pixels in detections, one detector, one view", ablation["claim_one_view"]["pq"]),
        ("\\quad + second detector and mirror views", ablation["claim_fused"]["pq"]),
        ("\\quad + three probability cuts and learned selector (full)", full_pq),
        ("Oracle selection of the same candidates", report["oracle"]["pq"]),
    ]
    (REPO / "report" / "table_ablation.tex").write_text(
        "    \\begin{tabular}{lc}\n        \\toprule\n        Variant & PQ \\\\\n        \\midrule\n"
        + "".join(f"        {label} & {pq:.3f} \\\\\n" for label, pq in lines)
        + "        \\bottomrule\n    \\end{tabular}\n",
        encoding="utf-8",
    )
    print("wrote numbers.tex, table_main.tex, table_ablation.tex, table_sites.tex")


if __name__ == "__main__":
    main()
