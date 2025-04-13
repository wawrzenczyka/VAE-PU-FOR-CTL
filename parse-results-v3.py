# %%
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_rows", 500)

root = "results-2025-04-09/result"
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

# results_df = results_df[~results_df.Method.str.contains("SRuleOnly")]

results_df["BaseMethod"] = "VAE-PU"
results_df["OCC"] = np.where(
    results_df.Method.str.contains("A\^3"),
    "A^3",
    np.where(
        results_df.Method.str.contains("IsolationForest"),
        "IForest",
        np.where(results_df.Method.str.contains("OddsRatio"), "Bayes", "None"),
    ),
)

results_df["\\alpha"] = np.where(
    results_df.Method.str.contains("FOR-CTL"),
    results_df.Method.str.split("-").apply(lambda x: x[-1] if re.match('[0-9]+\.[0-9]+', x[-1]) else None),
    'Normal OCC',
)

results_df["Dataset"] = results_df["Dataset"].str.replace("MachineAnimal", "VA")
results_df["Dataset"] = results_df["Dataset"].str.replace("CarTruck", "CT")

results_df.to_csv("full_results.csv", index=False)

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

# %%
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

results_df = pd.read_csv("full_results.csv")
results_df

results_df = results_df[~(results_df.OCC == "None")]

# %%
for metric in ["Accuracy", 'F1 score']:
    pivot_df = results_df.pivot_table(
        values=metric,
        index=["Dataset", "c"],
        columns=["OCC", '\\alpha'],
        aggfunc=np.nanmean,
        dropna=False
    )

    pivot_df.round(3).to_csv(f"{metric}.csv")
    pivot_df.round(3).to_latex(f"{metric}.tex")
    display(pivot_df.round(3))

# %%
results_df['Actual to expected example ratio'] = results_df['Actual n_PU'] / results_df['Expected n_PU']
pivot_df = results_df.loc[~(results_df["\\alpha"] == 'Normal OCC')].pivot_table(
    values=['Actual to expected example ratio'],
    index=["Dataset", "c"],
    columns=["OCC", '\\alpha'],
    aggfunc=np.nanmean,
    dropna=False
)

pivot_df.round(3).to_csv(f"n_PU.csv")
pivot_df.round(3).to_latex(f"n_PU.tex")
display(pivot_df.round(3))

# %%
for metric in ["U-Balanced accuracy", "U-Accuracy"]:
    results_df["Label shift label"] = np.where(
        results_df["Label shift \\pi"] == "None",
        "no shift",
        "$\\widetilde{\\pi}="
        + results_df["Label shift \\pi"].astype(str).str.slice(0, 3)
        + "$",
    )

    # Calculate the maximum accuracy value for each group (["Dataset", "Label shift label", "c", "Experiment"])
    max_accuracy_df = results_df.groupby(
        ["Dataset", "Label shift label", "c", "Experiment"]
    )[metric].transform("max")

    # Calculate the accuracy difference (current accuracy - maximum accuracy in the group)
    results_df[f"{metric} Accuracy Difference"] = max_accuracy_df - results_df[metric]

    # Compute the mean of the accuracy differences for each combination of ["Dataset", "Label shift label", 'Label shift method']
    mean_accuracy_diff_df = (
        results_df.groupby(["Dataset", "Label shift method"])[
            f"{metric} Accuracy Difference"
        ]
        .mean()
        .reset_index()
    )

    # Pivot the DataFrame as specified, with 'Label shift method' as the index and ["Dataset", "Label shift label"] as columns
    pivot_df = mean_accuracy_diff_df.pivot(
        values=f"{metric} Accuracy Difference",
        columns="Dataset",
        index="Label shift method",
    )

    # Calculate the mean accuracy difference across all columns for each 'Label shift method' and add it as the last column
    pivot_df["Mean Difference"] = pivot_df.mean(axis=1)

    # Display the final DataFrame with the new column showing the mean accuracy difference
    display(pivot_df)

    # Save the results to CSV
    pivot_df.round(3).to_csv(f"mean_accuracy_diff_{metric}.csv")

# %%
os.makedirs("label_shift_metrics", exist_ok=True)

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 score",
    "Balanced accuracy",
    "U-Accuracy",
    "U-Precision",
    "U-Recall",
    "U-F1 score",
    "U-Balanced accuracy",
]:
    pivot = results_df.pivot_table(
        values=metric,
        index=["Dataset", "Label shift \\pi", "c"],
        columns=["BaseMethod", "OCC", "Label shift method"],
        dropna=False,
    )
    pivot
    # results_df.pivot_table(values='Balanced accuracy', index=['c', "Dataset"], columns=["BaseMethod", "Balancing", "OCC"])
    max_pivot = pivot.applymap(lambda a: f"{a * 100:.1f}") + np.where(
        pivot.eq(pivot.max(axis=1), axis=0), "*", ""
    )
    max_pivot.to_csv(os.path.join("label_shift_metrics", f"{metric}.csv"))

# %%
os.makedirs("label_shift_metrics_condensed", exist_ok=True)

condensed_results_df = results_df.loc[
    np.isin(
        results_df["Label shift method"],
        ["Augmented label shift", "Cutoff label shift", "EM label shift"],
    )
]
condensed_results_df["Label shift method"] = condensed_results_df[
    "Label shift method"
].str.replace(" label shift", "")

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 score",
    "Balanced accuracy",
    "U-Accuracy",
    "U-Precision",
    "U-Recall",
    "U-F1 score",
    "U-Balanced accuracy",
]:
    pivot = condensed_results_df.pivot_table(
        values=metric,
        index=["Dataset", "Label shift \\pi", "c"],
        columns=["OCC", "Label shift method"],
    )
    pivot
    # results_df.pivot_table(values='Balanced accuracy', index=['c', "Dataset"], columns=["BaseMethod", "Balancing", "OCC"])
    max_pivot = pivot.applymap(lambda a: f"{a * 100:.1f}") + np.where(
        pivot.eq(pivot.max(axis=1), axis=0), "*", ""
    )
    max_pivot.to_csv(os.path.join("label_shift_metrics_condensed", f"{metric}.csv"))

# %%
direct_estimation_error = (
    results_df["Immediate \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
direct_estimation_error = direct_estimation_error.dropna()
direct_error = np.sqrt((direct_estimation_error**2).mean())

em_estimation_error = (
    results_df["EM \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
em_estimation_error = em_estimation_error.dropna()
em_error = np.sqrt((em_estimation_error**2).mean())

direct_error, em_error

# %%
os.makedirs("pi_metrics", exist_ok=True)

results_df["Direct \\pi~ estimation error"] = (
    results_df["Immediate \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
results_df["EM \\pi~ estimation error"] = (
    results_df["EM \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
results_df["Direct 1 / \\pi~ estimation error"] = (
    1 / results_df["Immediate \\pi estimation"]
    - 1 / results_df["True label shift \\pi"]
).abs()
results_df["EM 1 / \\pi~ estimation error"] = (
    1 / results_df["EM \\pi estimation"] - 1 / results_df["True label shift \\pi"]
).abs()

for metric in [
    "Direct \\pi~ estimation error",
    "EM \\pi~ estimation error",
    "Direct 1 / \\pi~ estimation error",
    "EM 1 / \\pi~ estimation error",
]:
    pivot = results_df.pivot_table(
        values=metric,
        index=["Dataset", "Label shift \\pi", "c"],
        columns=["BaseMethod", "OCC", "Label shift method"],
    )
    # display(pivot)

    metric_file = metric.replace("1 / \\pi", "pi inverse").replace("\\pi", "pi")
    pivot.to_csv(os.path.join("pi_metrics", f"{metric_file}.csv"))

# %%
import altair as alt

from save_chart import save_chart

alt.data_transformers.enable("vegafusion")
# alt.renderers.disable("jupyter")

chart = (
    alt.Chart(
        results_df[["Label shift \\pi", "U-Balanced accuracy", "Dataset", "c"]][
            :5000
        ].rename(columns={"Label shift \\pi": "pi~"})
    )
    .mark_line()
    .encode(
        x=alt.X("pi~:N"),
        y=alt.Y("U-Balanced accuracy"),
        row=alt.Facet("Dataset"),
        column=alt.Facet("c"),
    )
)
chart

# %%
results_df.to_csv("test.csv")
