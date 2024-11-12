# %%
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_rows", 500)

root = "result"
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

results_df["Label shift \\pi"] = results_df["Label shift \\pi"].fillna("None")
results_df = results_df[
    (results_df["Dataset"] != "CDC-Diabetes")
    & (~results_df["Dataset"].str.contains("SCAR"))
]
# results_df = results_df[(results_df["c"] != 0.02)]
results_df = results_df[results_df["OCC"] == "Bayes"]

results_df["Immediate \\pi~ estimation error"] = (
    results_df["Immediate \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
results_df["EM \\pi~ estimation error"] = (
    results_df["EM \\pi estimation"] - results_df["True label shift \\pi"]
).abs()

pi_mse_df = pd.concat(
    [
        results_df.pivot_table(
            values=metric,
            index=["Dataset", "Label shift \\pi", "c"],
            columns=["BaseMethod", "OCC", "Label shift method"],
            aggfunc=lambda x: np.sqrt(np.mean(x**2)),
        )
        .set_axis(["Value"], axis=1)
        .assign(Metric=metric)
        for metric in ["Immediate \\pi~ estimation error", "EM \\pi~ estimation error"]
    ],
    axis=0,
)
pi_mse_df = pi_mse_df.reset_index(drop=False)
pi_errors = (
    pi_mse_df.groupby(["Dataset", "Label shift \\pi", "Metric"])
    .Value.agg(lambda x: np.sqrt(np.sum(x**2 / len(x))))
    .reset_index(drop=False)
)

results_df = results_df[
    np.isin(
        results_df["Label shift method"],
        ["Augmented label shift", "Cutoff label shift", "EM label shift"],
    )
]

results_df["Label shift method"] = np.where(
    results_df["Label shift method"] == "Augmented label shift",
    "ALS",
    results_df["Label shift method"],
)
results_df["Label shift method"] = np.where(
    results_df["Label shift method"] == "Cutoff label shift",
    "CLS",
    results_df["Label shift method"],
)
results_df["Label shift method"] = np.where(
    results_df["Label shift method"] == "EM label shift",
    "CLS-EM",
    results_df["Label shift method"],
)

results_df

# %%
sns.set_theme()

for metric in ["U-Balanced accuracy", "U-Accuracy"]:
    results_df["Label shift label"] = np.where(
        results_df["Label shift \\pi"] == "None",
        "no shift",
        "$\\widetilde{\\pi}="
        + results_df["Label shift \\pi"].astype(str).str.slice(0, 3)
        + "$",
    )

    facet = sns.FacetGrid(
        results_df, row="Dataset", col="Label shift label", height=2.2
    )
    facet.map_dataframe(
        sns.lineplot,
        x="c",
        y=metric,
        hue="Label shift method",
        style="Label shift method",
        markers=True,
        errorbar="se",
    )
    facet.set_titles("{row_name}, {col_name}")
    facet.add_legend()

    sns.move_legend(
        facet,
        "lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        title=None,
        frameon=False,
    )

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{metric}.pdf", bbox_inches="tight")
    plt.savefig(f"plots/{metric}.png", dpi=300, bbox_inches="tight")

    plt.show()

# %%
sns.set_theme()

pi_errors["$\\widetilde{\\pi}$"] = np.where(
    pi_errors["Label shift \\pi"] == "None",
    "No shift",
    pi_errors["Label shift \\pi"].astype(str).str.slice(0, 3),
)

pi_errors["Estimation method"] = np.where(
    pi_errors["Metric"].str.contains("EM"), "EM", "Immediate estimator"
)
pi_errors["Error"] = pi_errors["Value"]

facet = sns.FacetGrid(pi_errors, col="Dataset", height=2.6, col_wrap=3)
facet.map_dataframe(
    sns.lineplot,
    x="$\\widetilde{\\pi}$",
    y="Error",
    hue="Estimation method",
    style="Estimation method",
    markers=True,
)
facet.set_titles("{col_name}")
facet.add_legend()

sns.move_legend(
    facet,
    "lower center",
    bbox_to_anchor=(0.69, 0.24),
    ncol=1,
    title=None,
    frameon=False,
)

for ax in facet.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/pi-errors.pdf", bbox_inches="tight")
plt.savefig("plots/pi-errors.png", dpi=300, bbox_inches="tight")

plt.show()

# %%
for metric in ["U-Balanced accuracy", "U-Accuracy"]:
    results_df["Label shift label"] = np.where(
        results_df["Label shift \\pi"] == "None",
        "no shift",
        "$\\widetilde{\\pi}="
        + results_df["Label shift \\pi"].astype(str).str.slice(0, 3)
        + "$",
    )

    # First, calculate the rank of U-Accuracy values within each subgroup defined by ["Dataset", "Label shift label", 'c', 'Experiment']
    results_df[f"{metric} Rank"] = results_df.groupby(
        ["Dataset", "Label shift label", "c", "Experiment"]
    )[metric].rank(ascending=False)
    # Then, compute the average rank for each combination of ["Dataset", "Label shift label", 'Label shift method']
    average_rank_df = (
        results_df.groupby(["Dataset", "Label shift method"])[f"{metric} Rank"]
        .mean()
        .reset_index()
    )
    # Pivot the DataFrame as specified, with 'Label shift method' as the index and ["Dataset", "Label shift label"] as columns
    pivot_df = average_rank_df.pivot(
        values=f"{metric} Rank", columns="Dataset", index="Label shift method"
    )

    # Calculate the mean rank across all columns for each 'Label shift method' and add it as the last column
    pivot_df["Mean Rank"] = pivot_df.mean(axis=1)

    # Display the final DataFrame
    display(pivot_df)

    pivot_df.round(2).to_csv(f"ranks_{metric}.csv")

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
immediate_estimation_error = (
    results_df["Immediate \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
immediate_estimation_error = immediate_estimation_error.dropna()
immediate_error = np.sqrt((immediate_estimation_error**2).mean())

em_estimation_error = (
    results_df["EM \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
em_estimation_error = em_estimation_error.dropna()
em_error = np.sqrt((em_estimation_error**2).mean())

immediate_error, em_error

# %%
os.makedirs("pi_metrics", exist_ok=True)

results_df["Immediate \\pi~ estimation error"] = (
    results_df["Immediate \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
results_df["EM \\pi~ estimation error"] = (
    results_df["EM \\pi estimation"] - results_df["True label shift \\pi"]
).abs()
results_df["Immediate 1 / \\pi~ estimation error"] = (
    1 / results_df["Immediate \\pi estimation"]
    - 1 / results_df["True label shift \\pi"]
).abs()
results_df["EM 1 / \\pi~ estimation error"] = (
    1 / results_df["EM \\pi estimation"] - 1 / results_df["True label shift \\pi"]
).abs()

for metric in [
    "Immediate \\pi~ estimation error",
    "EM \\pi~ estimation error",
    "Immediate 1 / \\pi~ estimation error",
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
