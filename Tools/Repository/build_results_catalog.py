"""Build study-level summaries from the retained, read-only evaluation records."""
import argparse
import csv
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("MR", "b-minFDE6", "minADE1", "minADE6", "minFDE1", "minFDE6")
STUDIES = (
    ("SEAM/State_Space_Integration", "SEAM State-Space Integration",
     "Residual agent-history refinement and future-sequence decoder replacement.",
     "Results/SEAM_AV2_MAMBA_3RUN_20260824-202936", "80 epochs; global batch 32; two GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.",
     "launch_lab3_seam_3run.sh", "publish_seam_suite_snapshot.sh"),
    ("SEAM/Context_Attention_Ablation", "SEAM Context and Attention Ablation",
     "Independent uncertainty-aware context, relative geometry and QKNorm interventions.",
     "Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512", "20 epochs; global batch 48; three GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.",
     "launch_lab3_seam_20epoch_4test_max_resources.sh", "publish_seam_20epoch_snapshot.sh"),
    ("SEAM/Combined_Extension", "SEAM Combined Extension",
     "Uncertainty, geometry and QKNorm composed with a future-head Mamba module.",
     "Results/SEAM_AV2_80EPOCH_COMBINED_20260906-144118", "80 epochs configured; global batch 32; two GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.",
     "launch_combined.sh", "publish_combined.py"),
    ("SHARP/Attention_Operators", "SHARP Attention Operators",
     "Multi-head attention, query-key normalisation, head mixing and their combination.",
     "Main_Results/Attention_Experiments", "80 epochs; global batch 24; three GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5.",
     "setup_lab3_attention_ablation.py", "Codes/publish_lab3_attention_snapshot.sh"),
    ("SHARP/Architecture_Ablation", "SHARP Architectural Ablation",
     "Nine independent memory, context, pooling, geometry and decoder interventions against a shared control.",
     "Runs/SHARP_AV2_20EPOCH_10TEST_20260805-031733", "20 epochs; global batch 32; four GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5; 13 warm-up epochs.",
     "launch_lab2_10test_suite_v5.sh", "publish_lab2_10test_snapshot.sh"),
    ("SHARP/State_Space_Integration", "SHARP State-Space Integration",
     "Scene-token recurrence and residual temporal-agent recurrence, retained as placement experiments.",
     "Main_Results/Mamba_Encoder_Addition_Results", "The reported placement runs use 80 epochs and global batch 32 on four GPUs. Earlier setup versions also remain available.",
     "setup_lab2_sharp_mamba_article80.py", None),
    ("SHARP/Architecture_Integration", "SHARP Architecture Integration",
     "A reference reproduction followed by composed QKNorm, uncertainty and relative-geometry modifications.",
     "Results/SHARP_FINAL_3RUN_20260825-174605", "80 epochs; global batch 32; four GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5; 13 warm-up epochs.",
     "launch_final_3run_suite.sh", "publish_final_artifacts.sh"),
)


def read(path):
    return (ROOT / path).read_text(encoding="utf-8-sig")


def md_rows(path):
    rows = []
    for line in read(path).splitlines():
        if line.startswith("| ") and not line.startswith("|---"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def number(value):
    return float(re.search(r"[+-]?\d+\.\d+", value).group())


def record(label, selected, vector, source, basis):
    return dict(variant=label, selected_minADE6=selected, **vector,
                source=source, evaluation_basis=basis)


def collect(study):
    key, title, desc, result, protocol, launcher, publisher = study
    base = "Studies/" + key + "/"
    r = base + result
    records, timing = [], []
    caveat = ""
    if key == "SEAM/State_Space_Integration":
        table = md_rows(r + "/Summary.md")
        best = [row for row in table if len(row) == 5 and row[1] == "complete"]
        metrics = [row for row in table if len(row) == 8 and row[1].isdigit()]
        for chosen, vector in zip(best, metrics):
            values = dict(zip(("MR", "minADE1", "minADE6", "minFDE1", "minFDE6", "b-minFDE6"), map(float, vector[2:])))
            records.append(record(chosen[0], number(chosen[4]), values, r + "/Summary.md", "checkpoint selection; rounded final log vector separate"))
        assert len(records) == 3
        caveat = "The best-checkpoint metric and rounded final log vector are separate observations. The captured summary labels the final validation as epoch 81; that label is retained, not interpreted as an 82-epoch training schedule."
        insight = "Future-head replacement reduces selected minADE6 from 0.662859 to 0.648480 (-2.17%). Agent-history addition gives 0.664814 (+0.29%). The result supports sequence modelling at the future-coordinate head in this study, rather than a universal benefit from adding recurrence."
    elif key == "SEAM/Context_Attention_Ablation":
        labels = {"baseline": "SEAM reference", "uncertainty_target_context": "Uncertainty-aware context", "relative_geometry_bias": "Relative geometry", "qknorm": "QKNorm"}
        for row in csv.DictReader(io.StringIO(read(r + "/Comparison.csv"))):
            assert row["status"] == "complete"
            records.append(record(labels[row["variant"]], float(row["best_saved_minADE6"]), {m: float(row[m]) for m in METRICS}, r + "/Comparison.csv", "saved checkpoint selection; final vector separate for geometry"))
        assert len(records) == 4
        caveat = "For relative geometry, selected minADE6 is 0.717261 while the final vector contains 0.718735. These are not the same checkpoint. The earlier batch-32 attempt is retained separately and is not pooled with this batch-48 study."
        timing_source = r + "/Timing.csv"
        timing_text = read(timing_source)
        timing = list(csv.DictReader(io.StringIO(timing_text)))
        for item in timing:
            item["variant"] = labels[item["variant"]]
        insight = "All three independent interventions improve selected minADE6 against the matched control: uncertainty -2.60%, geometry -1.43% and QKNorm -3.27%. QKNorm achieves the lowest selected error, 0.703845. These independent gains do not establish that their composition will be additive."
    elif key == "SHARP/Architecture_Ablation":
        for row in md_rows(r + "/Summary.md"):
            if len(row) == 12 and row[1] == "Completed":
                vector = dict(zip(METRICS, map(float, [*row[5:9], *row[10:12]])))
                records.append(record(row[0], vector["minADE6"], vector, r + "/Summary.md", "published suite summary; selection and other metrics may differ"))
        assert len(records) == 10
        caveat = "The captured summary derives selected minADE6 from checkpoint filenames and uses retained or nearest validation records for other metrics. Treat the complete vector as reported evidence, not a newly re-evaluated selected checkpoint."
        insight = "Relative geometry gives the lowest selected minADE6 (0.749679, -0.35%), followed by uncertainty-aware context (0.749961, -0.32%). The remaining interventions do not improve this primary error. Endpoint refinement improves minFDE1 despite its higher minADE6, illustrating that trajectory-average and endpoint criteria need not move together."
    elif key == "SHARP/Attention_Operators":
        source = r + "/Runs/SHARP_ATTENTION_ABLATION_20260717-123916/Summary.md"
        for row in md_rows(source):
            if len(row) == 11 and row[1] == "Completed":
                vector = dict(zip(METRICS, map(float, row[5:])))
                records.append(record(row[0], float(row[3]), vector, source, "MHA final vector; alternatives selected evaluations"))
        assert len(records) == 4
        caveat = "The MHA vector is final-epoch validation, whereas the alternatives have retained selected-checkpoint evaluations. The primary comparison therefore uses the saved selected minADE6 for every operator; do not attribute all secondary-metric changes to a fully matched checkpoint evaluation."
        insight = "QKNorm improves selected minADE6 from 0.673460 to 0.669777 (-0.55%). Talking-Heads alone gives +0.23%, and QKNorm with head mixing gives +0.15%. The combined operator has the lowest MR in the retained vectors, but not the lowest trajectory-average error."
    elif key == "SHARP/Architecture_Integration":
        for slug, label in (("01_official_sharp_baseline", "SHARP reference"), ("02_qknorm_uncertainty_geometry", "SHARP + QKNorm + uncertainty + geometry")):
            values = json.loads(read(r + "/" + slug + "/metrics.json"))
            assert values["evaluation_devices"] == 1
            records.append(record(label, values["minADE6"], {m: values[m] for m in METRICS}, r + "/" + slug + "/metrics.json", "single-device selected-checkpoint evaluation"))
            t = json.loads(read(r + "/" + slug + "/RUN_TIMING.json"))
            timing.append({"variant": label, "training_duration_seconds": t["training_duration_seconds"], "source": r + "/" + slug + "/RUN_TIMING.json"})
        caveat = "Two final selected-checkpoint evaluations are available. The residual-Mamba extension has implementation code but no committed final metrics in this evidence set. The older Current_Progress snapshot predates the second completed evaluation and is not the current comparison."
        insight = "The combined model improves MR, b-minFDE6, minFDE1 and minFDE6. Selected minADE6 changes from 0.679282 to 0.680589 (+0.19%), so the composition does not improve every accuracy criterion. MR changes from 0.155955 to 0.155515 (-0.28%). Independent screening gains are therefore not assumed to sum when mechanisms are composed."
    elif key == "SHARP/State_Space_Integration":
        source = r + "/FINAL_METRICS.txt"
        values = {m: float(re.search(r"^" + re.escape(m) + r"=([0-9.eE+-]+)", read(source), re.M).group(1)) for m in METRICS}
        records.append(record("Scene-token Mamba", values["minADE6"], values, source, "selected-checkpoint evaluation"))
        source2 = r + "/Temporal_Agent_Mamba/Snapshots/20260803-045226/BEST_CHECKPOINT.txt"
        selected = re.search(r"minADE6[_=: ]+([0-9.]+)", read(source2))
        assert selected, source2
        records.append(record("Residual temporal-agent Mamba", float(selected.group(1).rstrip(".")), {}, source2, "selected-checkpoint filename only"))
        caveat = "Only the scene-token placement has a full-precision final vector in this comparison. No secondary metrics are invented for the temporal-agent checkpoint. These historical placement runs are not merged with the independent architecture-integration control."
        insight = "Moving recurrence to the ordered agent history reduces selected minADE6 from 0.680362 to 0.673736 (-0.97%) relative to the scene-token placement. The result motivates preserving temporal order and a small residual pathway; it is not evidence that arbitrary recurrence improves every SHARP configuration."
    else:
        caveat = "The latest committed snapshot records early training and no final selected-checkpoint evaluation. The code is available for reproducibility, but this extension is not included in completed-result rankings. No separate SEAM + QKNorm + uncertainty + geometry run without Mamba is recorded."
        insight = "This implementation combines all three context/attention mechanisms with future-head Mamba. Accuracy claims require its own completed evaluation; they cannot be inferred from independent ablations."
    return records, timing, caveat, insight


def table(records):
    lines = ["| Variant | Selected minADE6 | Change | MR | b-minFDE6 | minADE1 | Vector minADE6 | minFDE1 | minFDE6 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    baseline = records[0]["selected_minADE6"]
    for row in records:
        values = [f"{row[m]:.6f}" if m in row else "-" for m in METRICS]
        delta = 100 * (row["selected_minADE6"] / baseline - 1)
        lines.append("| " + " | ".join([row["variant"], f'{row["selected_minADE6"]:.6f}', f"{delta:+.2f}%", *values]) + " |")
    lines += ["", "All errors are lower-is-better. Change is relative to the first row's selected minADE6. A dash means that a comparable numerical record is not supplied. Selection values and complete metric vectors are explicitly distinguished below.", ""]
    return "\n".join(lines)


def outputs():
    files, registry = {}, []
    for study in STUDIES:
        key, title, desc, result, protocol, launcher, publisher = study
        base = "Studies/" + key
        records, timing, caveat, insight = collect(study)
        lines = [f"# {title}: Results", "", desc, "", "## Comparison", ""]
        lines += [table(records)] if records else ["No final metric vector is available in the committed evidence.", ""]
        lines += ["## Interpretation", "", insight, "", "## Protocol and Evidence", "", protocol, "", caveat, ""]
        for source in dict.fromkeys(row["source"] for row in records):
            rel = source[len(base) + 1:]
            lines.append(f"- [Recorded source]({rel})")
        lines += [f"- [Original run records]({result}/)", "- [Implementation and reproduction](README.md)", ""]
        if timing:
            lines += ["## Training Time", "", "Timing is reported separately from forecasting error. See the source records for the retained duration definition.", ""]
            if "training_duration_seconds" in timing[0]:
                lines += ["| Variant | Duration (h:mm:ss) |", "|---|---:|"]
                for row in timing:
                    seconds = row["training_duration_seconds"]
                    lines.append(f'| {row["variant"]} | {seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d} |')
            else:
                lines += ["| Variant | Duration (h:mm:ss) |", "|---|---:|"]
                for row in timing:
                    lines.append(f'| {row["variant"]} | {row["active_training_time"]} |')
                lines += ["", f"[Recorded timing table]({result}/Timing.csv)"]
            lines.append("")
        files[base + "/Summary.md"] = "\n".join(lines)
        nav = [f"# {title}", "", desc, "", "## Recorded Finding", "", insight, "", caveat, "", "## Navigation", "", "- [Results and interpretation](Summary.md)", f"- [Original result records]({result}/)", "- [Setup and operational details](SETUP.md)", f"- [Preparation or launch entry point]({launcher})"]
        if publisher:
            nav.append(f"- [Artifact publisher]({publisher})")
        nav += ["- [Modification definitions](../../../Results/Info.md)", "- [Repository reproduction guide](../../../Documentation/Reproduction.md)", "", "## Shared Controls", "", protocol, "", "Numerical run records and timestamped snapshots are preserved as captured. Historical filenames identify the actual run; the study title identifies its scientific purpose. Earlier launch versions remain available for tracing source evolution, not as additional experiments.", ""]
        files[base + "/README.md"] = "\n".join(nav)
        dest = "Results/Studies/" + key
        files[dest + "/Summary.md"] = f"# {title}\n\n[Study summary and source evidence](../../../../{base}/Summary.md).\n\n{insight}\n\n{caveat}\n"
        if records:
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=["variant", "selected_minADE6", *METRICS, "source", "evaluation_basis"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
            files[dest + "/Metrics.csv"] = stream.getvalue()
        registry.append((title, base, len(records), insight))
    lines = ["# Experiment Registry", "", "This catalogue is organised by scientific study. Counts refer to evaluated configurations represented in the linked records, not setup attempts, snapshots or the number of scripts. No result is inferred from an implementation alone.", "", "| Study | Evaluated configurations | Evidence |", "|---|---:|---|"]
    for title, base, count, insight in registry:
        lines.append(f"| {title} | {count} | [Summary](../../{base}/Summary.md) |")
    lines += ["", "## Separate Historical Records", "", "Earlier SHARP AV1, nuScenes and shorter baseline summaries are retained in the [historical consolidated report](../../Documentation/Archive/2026-08-25/Main/SHARP_AV2_Main_Results.md). Its capture date and split/mode differences remain part of the record. They are not inserted as controls into the current matched AV2 studies.", "", "The earlier single-GPU work is held separately in [Thesis_A4000](https://github.com/madvidd/Thesis_A4000); this repository does not contain all of that external archive. No cross-dataset or cross-budget ranking is constructed.", "", "## Snapshot Precedence", "", "SEAM state-space integration has three completed variants; SEAM context/attention ablation has four; SHARP attention has four; the SHARP architectural ablation has ten. The SHARP integration suite has two committed final evaluations, superseding its older progress snapshot. Its residual-Mamba extension and the SEAM combined extension have no committed final evaluation in this catalogue.", "", "The [file inventory](../../Documentation/Repository_Audit/Original_File_Inventory.json) and [migration map](../../Documentation/Repository_Audit/Path_Migration.csv) retain the original path and content identity of every pre-organisation file.", ""]
    files["Results/Main/All_Run_Registry.md"] = "\n".join(lines)
    files["Results/Main/SHARP_AV2_Main_Results.md"] = "# SHARP Results Index\n\nResults are separated by matched study rather than combined into a single cross-protocol ranking.\n\n" + "\n".join(f"- [{title}](../../{base}/Summary.md)" for title, base, _, _ in registry if "/SHARP/" in base) + "\n\n[Earlier AV1, nuScenes and baseline records](../../Documentation/Archive/2026-08-25/Main/SHARP_AV2_Main_Results.md) are historical comparisons, not replacement controls for these studies.\n"
    files["Results/README.md"] = "# Experimental Results\n\nCurated summaries are regenerated from retained source records. Each model is compared with the control from its own study; selected checkpoint metrics and final log vectors are distinguished.\n\n" + "\n".join(f"- [{title}](../{base}/Summary.md): {insight}" for title, base, _, insight in registry) + "\n\n## Supporting Files\n\n- [Complete study registry](Main/All_Run_Registry.md)\n- [Modification guide](Info.md)\n- [Study-specific CSV tables](Studies/)\n- [Evidence and filename conventions](../Documentation/Repository_Guide.md)\n\n`Metrics.csv` files are derived comparisons with explicit source paths, not replacements for raw evaluations. Run `python Tools/Repository/build_results_catalog.py --check` from the repository root to check that the maintained summaries agree with the retained records.\n"
    files["Results/README.md"] += "\nThe [research figure catalogue](Figures/README.md) and [visual gallery](Figures/Gallery.md) link every current numbered figure and the graphical abstract to SVG, vector PDF and PNG files, with separate architecture, metric and training-time groups.\n"
    for model in ("SEAM", "SHARP"):
        files[f"Studies/{model}/README.md"] = f"# {model} Studies\n\n" + "\n".join(f"- [{title}]({base.split('/')[-1]}/README.md)" for title, base, _, _ in registry if f"/{model}/" in base) + "\n\nEach study retains its own training controls, source provenance and numerical evidence.\n"
    files["Studies/README.md"] = "# Streaming Forecasting Studies\n\n- [SEAM studies](SEAM/README.md): endpoint-aware context, attention and selective recurrence.\n- [SHARP studies](SHARP/README.md): short-window attention, architectural ablations and mechanism integration.\n\n[Experiment registry](../Results/Main/All_Run_Registry.md) | [Modification guide](../Results/Info.md)\n"
    files["Results/Studies/README.md"] = "# Study-Specific Comparison Data\n\n" + "\n".join(f"- [{title}]({base.removeprefix('Studies/')}/Summary.md)" for title, base, _, _ in registry) + "\n"
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    differences = []
    for relative, content in outputs().items():
        path = ROOT / relative
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                differences.append(relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if differences:
        raise SystemExit("Catalogue differs: " + ", ".join(differences))
    print("RESULTS_CATALOGUE_VERIFIED" if args.check else "RESULTS_CATALOGUE_WRITTEN")


if __name__ == "__main__":
    main()
