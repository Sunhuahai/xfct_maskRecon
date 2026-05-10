from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _top_screening(rows: list[dict], limit: int = 5) -> list[dict]:
    default_rows = [r for r in rows if r.get("angle_set") == "phase0_default"] or rows
    return sorted(default_rows, key=lambda r: _float(r, "ranking_score"), reverse=True)[:limit]


def _summarize_protocol(rows: list[dict]) -> str:
    if not rows:
        return "Protocol comparison has not been run yet."
    lines = [
        "| protocol | best valid single DL | best valid multi DL | best valid multi run | interpretation |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for protocol in sorted(set(r["protocol"] for r in rows)):
        subset = [r for r in rows if r["protocol"] == protocol]
        valid = [
            r
            for r in subset
            if str(r.get("detection_limit_invalid", "False")).lower() != "true"
            and np.isfinite(_float(r, "detection_limit_mgml"))
            and _float(r, "detection_limit_mgml") > 0.0
        ]
        singles = [r for r in valid if r.get("family") == "single_pinhole"]
        multis = [r for r in valid if r.get("family") != "single_pinhole"]
        best_single = min(singles, key=lambda r: _float(r, "detection_limit_mgml")) if singles else None
        best_multi = min(multis, key=lambda r: _float(r, "detection_limit_mgml")) if multis else None
        interp = "no valid DL comparison"
        if best_single and best_multi:
            if protocol == "equal_detected_counts":
                interp = (
                    "coding/inverse quality improved"
                    if _float(best_multi, "detection_limit_mgml") < _float(best_single, "detection_limit_mgml")
                    else "no equal-count coding advantage"
                )
            else:
                interp = (
                    "multi-hole lower valid DL"
                    if _float(best_multi, "detection_limit_mgml") < _float(best_single, "detection_limit_mgml")
                    else "multi-hole not lower valid DL"
                )
        lines.append(
            "| {protocol} | {single:.4f} | {multi:.4f} | {run} | {interp} |".format(
                protocol=protocol,
                single=_float(best_single or {}, "detection_limit_mgml"),
                multi=_float(best_multi or {}, "detection_limit_mgml"),
                run=(best_multi or {}).get("run", "NA"),
                interp=interp,
            )
        )
    return "\n".join(lines)


def _summarize_robustness(rows: list[dict]) -> str:
    if not rows:
        return "Pose sensitivity has not been run yet."
    nominal = {
        r["candidate_id"]: r
        for r in rows
        if r.get("perturbation") == "nominal"
    }
    lines = ["| candidate | mean | max | note |", "| --- | ---: | ---: | --- |"]
    for candidate_id in sorted(nominal):
        deltas = []
        for row in rows:
            if row.get("candidate_id") != candidate_id or row.get("perturbation") == "nominal":
                continue
            deltas.append(abs(_float(row, "detection_limit_mgml") - _float(nominal[candidate_id], "detection_limit_mgml")))
        if deltas:
            lines.append(
                f"| {candidate_id} | {float(np.mean(deltas)):.4f} | {float(np.max(deltas)):.4f} | mean/max absolute DL change |"
            )
    return "\n".join(lines)


def _improvement_status(screening: list[dict], mbir: list[dict], protocol: list[dict], robustness: list[dict], validation_status: str) -> str:
    provisional = "provisional; validation failed" if validation_status != "PASS" else "validated"
    lines = ["| question | status |", "| --- | --- |"]
    if protocol:
        eq_time = [r for r in protocol if r.get("protocol") == "equal_acquisition_time"]
        trad5 = [r for r in eq_time if r.get("run") == "traditional_5"]
        multis = [r for r in eq_time if r.get("family") != "single_pinhole"]
        if trad5 and multis:
            trad_counts = float(np.mean([_float(r, "total_detected_counts") for r in trad5]))
            best_multi_counts = max(float(np.mean([_float(r, "total_detected_counts") for r in multis if r["run"] == run])) for run in sorted(set(r["run"] for r in multis)))
            lines.append(("| raw throughput | " f"{provisional}; best multi/single count ratio about {best_multi_counts / max(trad_counts, 1e-12):.2f} in quick equal-time runs. |"))
        else:
            lines.append("| raw throughput | not assessed. |")
    else:
        lines.append("| raw throughput | not assessed. |")
    if screening:
        best = _top_screening(screening, limit=1)[0]
        lines.append(
            "| Fisher/task detectability | "
            f"{provisional}; top screened candidate `{best['candidate_id']}` has task Fisher d2={_float(best, 'task_fisher_d2_mean'):.3e}, "
            "but quick screening did not establish a validated grid-vs-new improvement. |"
        )
    else:
        lines.append("| Fisher/task detectability | not assessed. |")
    valid_protocol = [
        r
        for r in protocol
        if str(r.get("detection_limit_invalid", "False")).lower() != "true"
        and np.isfinite(_float(r, "detection_limit_mgml"))
        and _float(r, "detection_limit_mgml") > 0.0
    ]
    if valid_protocol:
        best_single = min((r for r in valid_protocol if r.get("family") == "single_pinhole"), key=lambda r: _float(r, "detection_limit_mgml"), default=None)
        best_multi = min((r for r in valid_protocol if r.get("family") != "single_pinhole"), key=lambda r: _float(r, "detection_limit_mgml"), default=None)
        if best_single and best_multi:
            dl_status = "improves" if _float(best_multi, "detection_limit_mgml") < _float(best_single, "detection_limit_mgml") else "does not improve"
            lines.append(f"| reconstruction DL | {provisional}; best valid multi {dl_status} over best valid single in quick rows. |")
        else:
            lines.append("| reconstruction DL | no valid single-vs-multi DL comparison. |")
    else:
        lines.append("| reconstruction DL | no valid DL comparison; CNR fits are invalid/nonsensical in quick rows. |")
    if protocol:
        best_bias = min((abs(_float(r, "roi_bias")) for r in protocol if r.get("family") != "single_pinhole"), default=np.nan)
        single_bias = min((abs(_float(r, "roi_bias")) for r in protocol if r.get("family") == "single_pinhole"), default=np.nan)
        lines.append(f"| ROI bias | {provisional}; best abs multi bias={best_bias:.4f}, best abs single bias={single_bias:.4f} in quick protocol rows. |")
    else:
        lines.append("| ROI bias | not assessed. |")
    if mbir:
        best_fit = min(mbir, key=lambda r: _float(r, "final_deviance"))
        best_resid = min(mbir, key=lambda r: _float(r, "residual_structure_score"))
        lines.append(
            "| projection fit | "
            f"{provisional}; lowest raw-MBIR deviance `{best_fit['candidate_id']}`, lowest residual structure `{best_resid['candidate_id']}`. |"
        )
    else:
        lines.append("| projection fit | not assessed. |")
    if robustness:
        nominal = {r["candidate_id"]: r for r in robustness if r.get("perturbation") == "nominal"}
        scores = {}
        for candidate_id, base in nominal.items():
            vals = [
                abs(_float(r, "detection_limit_mgml") - _float(base, "detection_limit_mgml"))
                for r in robustness
                if r.get("candidate_id") == candidate_id and r.get("perturbation") != "nominal"
            ]
            if vals:
                scores[candidate_id] = float(np.mean(vals))
        if scores:
            best_robust = min(scores, key=scores.get)
            lines.append(f"| robustness | {provisional}; lowest mean DL perturbation is `{best_robust}` ({scores[best_robust]:.4f} mg/ml). |")
        else:
            lines.append("| robustness | not assessed. |")
    else:
        lines.append("| robustness | not assessed. |")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> str:
    validation = _read_json(PROJECT_ROOT / "results" / "forward_model_validation" / "validation_summary.json")
    screening = _read_csv(PROJECT_ROOT / "results" / "mask_design" / "candidate_screening.csv")
    top_candidates = _read_json(PROJECT_ROOT / "results" / "mask_design" / "top_candidates.json")
    mbir = _read_csv(PROJECT_ROOT / "results" / "poisson_mbir_mask_recon" / "poisson_mbir_summary.csv")
    protocol = _read_csv(PROJECT_ROOT / "results" / "protocol_comparison" / "protocol_summary.csv")
    robustness = _read_csv(PROJECT_ROOT / "results" / "mask_pose_sensitivity" / "mask_pose_sensitivity.csv")

    validation_status = (validation or {}).get("overall_status", "NOT_RUN")
    padding_result = None
    if validation:
        for test in validation.get("tests", []):
            if test.get("name") == "detector_padding":
                padding_result = test
                break
    padding_text = "not assessed"
    if padding_result:
        padding_text = (
            f"{padding_result.get('status')}; virtual fraction="
            f"{padding_result.get('virtual_fraction', 'NA')}"
        )

    top_screen = _top_screening(screening)
    best = top_screen[0] if top_screen else None
    best_geometry = "No screened candidate available."
    if best:
        best_geometry = (
            f"`{best['candidate_id']}` ({best['family']}), holes={best['hole_count']}, "
            f"diameter={best['hole_diameter_mm']} mm, min distance={best['min_distance_mm']} mm, "
            f"score={_float(best, 'ranking_score'):.4f}."
        )
    if top_candidates and top_candidates.get("validation_warning"):
        best_geometry += f" Warning: {top_candidates['validation_warning']}"

    interpretation = "Protocol comparison has not been run."
    if protocol:
        equal_count = [r for r in protocol if r.get("protocol") == "equal_detected_counts"]
        valid_equal_count = [
            r
            for r in equal_count
            if str(r.get("detection_limit_invalid", "False")).lower() != "true"
            and np.isfinite(_float(r, "detection_limit_mgml"))
            and _float(r, "detection_limit_mgml") > 0.0
        ]
        singles = [r for r in valid_equal_count if r.get("family") == "single_pinhole"]
        multis = [r for r in valid_equal_count if r.get("family") != "single_pinhole"]
        if singles and multis:
            best_single = min(singles, key=lambda r: _float(r, "detection_limit_mgml"))
            best_multi = min(multis, key=lambda r: _float(r, "detection_limit_mgml"))
            if _float(best_multi, "detection_limit_mgml") < _float(best_single, "detection_limit_mgml"):
                interpretation = (
                    "In the available equal-detected-count runs, the best multi-hole candidate beats the best single-pinhole "
                    "baseline, which would indicate coding/inverse-problem benefit if forward validation passes."
                )
            else:
                interpretation = (
                    "In the available equal-detected-count runs, the best multi-hole candidate does not beat the best "
                    "single-pinhole baseline. Any equal-time/equal-dose gain should be interpreted mainly as throughput."
                )
        else:
            interpretation = (
                "The quick equal-detected-count runs do not provide a reliable valid-DL comparison because the CNR fits "
                "are invalid or nonsensical for the relevant rows. Treat these protocol results as smoke-test outputs only."
            )

    if validation_status != "PASS":
        recommendation = (
            "Fix forward-model support first. The current evidence should not be used to choose hardware: regenerate the "
            "mask matrix with physical 80-column clipping before padding, or regenerate projection data for a true "
            "160-column detector. After that, rerun screening and protocol comparison."
        )
    elif best and best.get("family") in {"blue_noise", "sparse_random", "ring", "ring_two"}:
        recommendation = (
            "Move the best sparse/ring candidate to a finite-aperture PMMA validation run, then run the final 50-seed "
            "protocol comparison before committing to fabrication."
        )
    else:
        recommendation = (
            "No non-grid candidate has a defensible validated advantage yet; keep the current grid only as a baseline and "
            "prioritize detector-size, pitch, and hole-count redesign."
        )

    lines = [
        "# Next-Stage Multi-Pinhole XFCT Experiment Report",
        "",
        "## Repository Changes and Commands",
        "",
        "Added scripts/modules:",
        "",
        "- `src/mask_xfct_model.py`: depth-dependent matrix-free mask operator and task phantoms.",
        "- `experiments/validate_mask_forward_consistency.py`: forward consistency validation.",
        "- `scripts/generate_mask_candidates.py`: sparse mask candidate generator.",
        "- `experiments/screen_mask_candidates.py`: task-based candidate screening.",
        "- `recon/poisson_tv_pdhg.py`: raw-domain Poisson-TV MBIR solver.",
        "- `experiments/run_poisson_mbir_mask_recon.py`: raw mask-domain MBIR experiment.",
        "- `experiments/run_mask_protocol_comparison.py`: equal time/dose/count comparison.",
        "- `experiments/run_mask_pose_sensitivity.py`: pose/geometry robustness study.",
        "- `experiments/make_next_experiments_report.py`: this report generator.",
        "",
        "Primary commands:",
        "",
        "```bash",
        "conda run -n xfct python experiments/run_effect_comparison.py --quick",
        "conda run -n xfct python experiments/validate_mask_forward_consistency.py --quick",
        "conda run -n xfct python scripts/generate_mask_candidates.py --quick",
        "conda run -n xfct python experiments/screen_mask_candidates.py --quick --candidate-limit 20",
        "conda run -n xfct python experiments/run_poisson_mbir_mask_recon.py --quick",
        "conda run -n xfct python experiments/run_mask_protocol_comparison.py --quick --num-seeds 3",
        "conda run -n xfct python experiments/run_mask_pose_sensitivity.py --quick",
        "conda run -n xfct python experiments/make_next_experiments_report.py",
        "```",
        "",
        "## Forward-Model Validation",
        "",
        f"Overall validation status: **{validation_status}**.",
        f"80x80 to 80x160 padding result: **{padding_text}**.",
        "",
        (validation or {}).get("stop_condition_message", "Validation has not been run."),
        "",
        "## Candidate Mask Ranking",
        "",
        best_geometry,
        "",
        "| rank | candidate | family | score | truncation | overlap | Fisher d2 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top_screen, start=1):
        lines.append(
            f"| {idx} | {row['candidate_id']} | {row['family']} | {_float(row, 'ranking_score'):.4f} | "
            f"{_float(row, 'global_truncation_physical_mean'):.4f} | {_float(row, 'overlap_max'):.4f} | "
            f"{_float(row, 'task_fisher_d2_mean'):.4e} |"
        )
    if not top_screen:
        lines.append("| NA | not run | NA | NA | NA | NA | NA |")
    lines.extend(
        [
            "",
            "## Raw Poisson MBIR",
            "",
            "| candidate | family | counts | DL | R2 | deviance | residual structure |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in mbir[:10]:
        lines.append(
            f"| {row['candidate_id']} | {row['family']} | {_float(row, 'raw_total_counts'):.3e} | "
            f"{_float(row, 'detection_limit_mgml'):.4f} | {_float(row, 'roi_r_squared'):.4f} | "
            f"{_float(row, 'final_deviance'):.3e} | {_float(row, 'residual_structure_score'):.4f} |"
        )
    if not mbir:
        lines.append("| not run | NA | NA | NA | NA | NA | NA |")
    lines.extend(
        [
            "",
            "## Fair Protocols",
            "",
            _summarize_protocol(protocol),
            "",
            "Interpretation:",
            "",
            interpretation,
            "",
            "## Improvement Status",
            "",
            _improvement_status(screening, mbir, protocol, robustness, validation_status),
            "",
            "## Pose Robustness",
            "",
            _summarize_robustness(robustness),
            "",
            "## Failure Modes",
            "",
            "- FOV truncation: reported as physical-detector truncation before padding and remains a central risk.",
            "- Hole overlap: reported by isolated-hole footprint inner products; high overlap is penalized.",
            "- Forward mismatch: validation explicitly checks physical 80-column support versus padded 160-column rows.",
            "- Poor conditioning: approximated with weighted mutual coherence and task Fisher/CRLB metrics.",
            "- Regularization bias: reported through ROI bias, CNR slope/intercept/R2, and invalid DL flags.",
            "- Pose sensitivity: reported as DL, ROI bias, deviance, and residual-structure change under mask/geometry perturbations.",
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
            "Scientific rule: if multi-hole wins only under equal time/dose but loses under equal detected counts, treat the benefit as throughput rather than coding information.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the next-stage XFCT multi-pinhole experiment report.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "next_experiments_report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(args), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
