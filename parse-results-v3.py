# %%
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_rows", 500)

root = "result-clean"
results = []

for dataset in os.listdir(root):
    if not os.path.isdir(os.path.join(root, dataset)):
        continue

    for c in os.listdir(os.path.join(root, dataset)):
        if c.startswith("Exp"):
            continue

        for exp in os.listdir(os.path.join(root, dataset, c)):
            exp_num = int(exp[3:])

            try:
                for file in os.listdir(os.path.join(root, dataset, c, exp)):
                    if "metric_values" in file and file.endswith(".json"):
                        with open(os.path.join(root, dataset, c, exp, file), "r") as f:
                            metrics = json.load(f)

                            metrics["Dataset"] = dataset
                            metrics["Experiment"] = exp_num
                            # metrics['c'] = 0.5 if 20 > (exp_num % 20) >= 10 else 0.02
                            metrics["c"] = float(c)

                            results.append(metrics)

                if os.path.exists(os.path.join(root, dataset, c, exp, "external")):
                    for method in os.listdir(
                        os.path.join(root, dataset, c, exp, "external")
                    ):
                        for file in os.listdir(
                            os.path.join(root, dataset, c, exp, "external", occ_method)
                        ):
                            if "metric_values" in file and file.endswith(".json"):
                                with open(
                                    os.path.join(
                                        root,
                                        dataset,
                                        c,
                                        exp,
                                        "external",
                                        method,
                                        file,
                                    ),
                                    "r",
                                ) as f:
                                    metrics = json.load(f)

                                    metrics["Dataset"] = dataset
                                    metrics["Experiment"] = exp_num
                                    # metrics['c'] = 0.5 if 20 > (exp_num % 20) >= 10 else 0.02
                                    metrics["c"] = float(c)

                                    results.append(metrics)

                if os.path.exists(os.path.join(root, dataset, c, exp, "occ")):
                    for occ_method in os.listdir(
                        os.path.join(root, dataset, c, exp, "occ")
                    ):
                        for file in os.listdir(
                            os.path.join(root, dataset, c, exp, "occ", occ_method)
                        ):
                            if "metric_values" in file and file.endswith(".json"):
                                with open(
                                    os.path.join(
                                        root,
                                        dataset,
                                        c,
                                        exp,
                                        "occ",
                                        occ_method,
                                        file,
                                    ),
                                    "r",
                                ) as f:
                                    metrics = json.load(f)

                                    metrics["Dataset"] = dataset
                                    metrics["Experiment"] = exp_num
                                    # metrics['c'] = 0.5 if 20 > (exp_num % 20) >= 10 else 0.02
                                    metrics["c"] = float(c)

                                    results.append(metrics)
            except:
                continue
        # break

results_df = pd.DataFrame.from_records(results)

results_df = results_df[~results_df.Method.str.contains("SRuleOnly")]

results_df["BaseMethod"] = "VAE-PU"
results_df["LabelShiftMethod"] = np.where(
    results_df.Method.str.contains("augmented-label-shift"),
    "Augmented",
    np.where(
        results_df.Method.str.contains("EM-label-shift"),
        "EM",
        np.where(
            results_df.Method.str.contains("simple-label-shift"),
            "Simple",
            "None",
        ),
    ),
)
results_df["OCC"] = np.where(
    results_df.Method.str.contains("A\^3"),
    "A^3",
    np.where(
        results_df.Method.str.contains("IsolationForest"),
        "IForest",
        np.where(results_df.Method.str.contains("OddsRatio"), "Bayes", "None"),
    ),
)

# results_df.Method = np.where(
#     results_df.Method == "A^3",
#     r"$A^3$",
#     results_df.Method,
# )
# results_df.Method = np.where(
#     results_df.Method == "EM",
#     "SAR-EM",
#     results_df.Method,
# )
# results_df.Method = np.where(
#     results_df.Method == "No OCC",
#     r"VP",
#     results_df.Method,
# )
# results_df.Method = results_df.Method.str.replace(
#     "-no S info", " -no S info", regex=False
# )
# results_df.Method = results_df.Method.str.replace("-e200-lr1e-4", "", regex=False)
# results_df.Method = results_df.Method.str.replace(" +S rule", "+S", regex=False)
# results_df.Method = results_df.Method.str.replace("SRuleOnly", "VP", regex=False)
# results_df.Method = results_df.Method.str.replace(
#     "OddsRatio-PUprop", "VP-B", regex=False
# )
# results_df.Method = results_df.Method.str.replace("$A^3$", "VP-$A^3$", regex=False)
# results_df.Method = results_df.Method.str.replace(
#     "IsolationForest", "VP-IF", regex=False
# )
results_df

# %%
os.makedirs("label_shift_metrics", exist_ok=True)

for metric in ["Accuracy", "Precision", "Recall", "F1 score", "Balanced accuracy"]:
    pivot = results_df.pivot_table(
        values=metric,
        index=["c", "Dataset", "Label shift \\pi"],
        columns=["BaseMethod", "LabelShiftMethod", "OCC"],
    )
    pivot
    # results_df.pivot_table(values='Balanced accuracy', index=['c', "Dataset"], columns=["BaseMethod", "Balancing", "OCC"])
    max_pivot = pivot.applymap(lambda a: f"{a * 100:.1f}") + np.where(
        pivot.eq(pivot.max(axis=1), axis=0), "*", ""
    )
    max_pivot.to_csv(os.path.join("label_shift_metrics", f"{metric}.csv"))

# %%
for dataset, rename in [
    ("CIFAR_CarTruck_red_val", "CIFAR CarTruck"),
    ("CIFAR_MachineAnimal_red_val", "CIFAR MachineAnimal"),
    ("STL_MachineAnimal_red_val", "STL MachineAnimal"),
    ("MNIST_35_bold_val", "MNIST 3v5"),
    ("MNIST_evenodd_bold_val", "MNIST OvE"),
]:
    results_df.loc[
        np.isin(
            results_df.Dataset,
            [dataset],
        )
        & np.isin(
            results_df.Method,
            [
                "VP-$A^3$",
                "VP-IF",
            ],
        ),
        "Dataset",
    ] = rename


# %%
def process_results(
    min_exp,
    max_exp,
    methods_filter,
    dataset_order,
    multicolumn=False,
    scaling=0.63,
):
    filtered_df = results_df

    for dataset, name in [
        ("CIFAR CarTruck", "CIFAR CT"),
        ("CIFAR MachineAnimal", "CIFAR MA"),
        ("STL MachineAnimal", "STL MA"),
        ("MNIST 3v5", "MNIST 3v5"),
        ("MNIST OvE", "MNIST OvE"),
        ("CDC-Diabetes", "CDC Diabetes"),
    ]:
        filtered_df.Dataset = np.where(
            filtered_df.Dataset == dataset, name, filtered_df.Dataset
        )

    # scar_datasets = [
    #     dataset for dataset in filtered_df.Dataset.unique() if "SCAR" in dataset
    # ]
    filtered_df = filtered_df.loc[np.isin(filtered_df.Dataset, dataset_order)]

    if min_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment >= min_exp]
    if max_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment < max_exp]
    if methods_filter is not None:
        filtered_df = filtered_df.loc[np.isin(filtered_df.Method, methods_filter)]

    filtered_df["Method"] = pd.Categorical(filtered_df["Method"], methods_filter)
    filtered_df["Dataset"] = pd.Categorical(filtered_df["Dataset"], dataset_order)

    for metric in ["Accuracy", "Precision", "Recall", "F1 score"]:
        processed_results = (
            filtered_df.pivot_table(
                values=metric,
                index=["c", "Method"],
                columns="Dataset",
                aggfunc=pd.DataFrame.mean,
            ).round(4)
            * 100
        )
        processed_results_sem = (
            filtered_df.pivot_table(
                values=metric,
                index=["c", "Method"],
                columns="Dataset",
                aggfunc=pd.DataFrame.sem,
            ).round(4)
            * 100
        )

        os.makedirs(os.path.join("processed_results", "Metrics"), exist_ok=True)
        os.makedirs(os.path.join("processed_results", "_all_tables"), exist_ok=True)
        processed_results.to_csv(
            os.path.join("processed_results", "Metrics", f"{metric}.csv")
        )

        # # PREPARE RESULT TABLES

        processed_results.columns.name = None

        def highlight_max(df, value_df):
            is_max = value_df.groupby(level=0).transform("max").eq(value_df)

            # max_df = pd.DataFrame(df, index=df.index, columns=df.columns)
            # max_df = max_df.applymap(lambda a: f'{a:.2f}')
            max_df = pd.DataFrame(
                np.where(is_max == True, "\\textbf{" + df + "}", df),
                index=df.index,
                columns=df.columns,
            )
            return max_df

        processed_results_text = (
            processed_results.applymap(lambda a: f"{a:.2f}")
            + " $\pm$ "
            + processed_results_sem.applymap(lambda a: f"{a:.2f}")
        )
        processed_results = highlight_max(processed_results_text, processed_results)

        include_caption = True
        include_label = True

        latex_table = processed_results.to_latex(
            index=True,
            escape=False,
            multirow=True,
            # caption=f"{metric} values per dataset." if include_caption else None,
            caption=(
                f"{metric} values -- VAE-PU-Bayes (traditional PU setting)"
                if include_caption
                else None
            ),
            label="tab:" + metric.replace(" ", "_") if include_label else None,
            position=(
                None
                if not include_label and not include_caption
                else "tbp" if not multicolumn else "btp"
            ),
        )
        cline_start = len(processed_results.index.names)
        cline_end = cline_start + len(processed_results.columns)

        # add full rule before baseline
        # latex_table = re.sub(r'(\\\\.*?\n)(.*?)Baseline', r'\1\\midrule \n\2Baseline', latex_table)

        # add mid rule after LBE or EM
        # latex_table = re.sub(r'(LBE.*? \\\\)', r'\1 \\cline{' \
        #     + str(cline_start) + '-' + str(cline_end) + \
        # '}', latex_table)
        latex_table = re.sub(
            r"(SAR-EM.*? \\\\)",
            r"\1 \\cline{" + str(cline_start) + "-" + str(cline_end) + "}",
            latex_table,
        )
        # latex_table = re.sub(r'(EM.*? \\\\)', r'\1 \\cline{' \
        #     + str(cline_start) + '-' + str(cline_end) + \
        # '}', latex_table)
        # latex_table = re.sub(r'(Baseline.*? \\\\)', r'\1 \\cmidrule{' \
        #     + str(cline_start) + '-' + str(cline_end) + \
        # '}', latex_table)

        # merge headers
        def merge_headers(latex_table):
            table_lines = latex_table.split("\n")
            tabular_start = 0
            tabular_end = len(table_lines) - 3

            if include_caption or include_label:
                tabular_start += 3
                tabular_end -= 1
            if include_caption and include_label:
                tabular_start += 1

            def process_line(l):
                return [
                    "\\textbf{" + name.replace("\\", "").strip() + "}"
                    for name in l.split("&")
                    if name.replace("\\", "").strip() != ""
                ]

            header_line, index_line = (
                table_lines[tabular_start + 2],
                table_lines[tabular_start + 3],
            )
            headers = process_line(header_line)
            index_names = process_line(index_line)

            new_headers = index_names + headers
            new_headers[-1] += " \\\\"
            new_headers = " & ".join(new_headers)

            table_lines.remove(header_line)
            table_lines.remove(index_line)
            table_lines.insert(tabular_start + 2, new_headers)

            table_lines = [
                "\t" + l if i > tabular_start and i < tabular_end else l
                for i, l in enumerate(table_lines)
            ]
            if include_caption or include_label:
                table_start = 0
                table_end = len(table_lines) - 2
                table_lines = [
                    "\t" + l if i > table_start and i < table_end else l
                    for i, l in enumerate(table_lines)
                ]

            # insert scaling
            table_lines.insert(tabular_end + 1, "}")
            table_lines.insert(tabular_start, "\scalebox{" + f"{scaling:.2f}" + "}{")
            # insert scaling

            return "\n".join(table_lines)

        latex_table = merge_headers(latex_table)

        if multicolumn:
            latex_table = latex_table.replace("{table}", "{table*}")
        latex_table = latex_table.replace(
            "\\centering",
            "\\centering \\scriptsize \\renewcommand{\\arraystretch}{1.2}",
        )

        with open(
            os.path.join("processed_results", "Metrics", f"{metric}.tex"), "w"
        ) as f:
            f.write(latex_table)
        with open(
            os.path.join("processed_results", "_all_tables", f"{metric}.tex"),
            "w",
        ) as f:
            f.write(latex_table)

        print(f"{metric} df")
        display(processed_results)


### ---------------------------------------------------------

min_exp, max_exp = 0, 10
methods_filter = [
    "VP",
    r"VP-$A^3$",
    "VP-IF",
    "VP-B",
]
dataset_filter = "MNIST 3v5"
grouping_cols = ["c", "Method"]
dataset_order = [
    "MNIST 3v5",
    "MNIST OvE",
    "CIFAR CT",
    "CIFAR MA",
    "STL MA",
    "CDC Diabetes",
]
multicolumn = False

process_results(
    min_exp,
    max_exp,
    methods_filter,
    dataset_order,
    multicolumn=multicolumn,
)


# %%
