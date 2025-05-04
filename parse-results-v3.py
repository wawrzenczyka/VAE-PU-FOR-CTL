# %%
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

pd.set_option("display.max_rows", 500)

root = "results-2025-05-04/result"
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

# results_df = results_df[~(results_df.OCC == "None")]

# Add normal VAE-PU results to IForest results
results_df.loc[results_df['OCC'] == "None", "\\alpha"] = 'VAE-PU'
# results_df[results_df.OCC == "None", "OCC"] = 'IForest'

# %%
filtered_df = results_df[results_df["OCC"] == 'IForest']

for metric in ["Accuracy", 'F1 score']:
    # Create a pivot table with both mean and sem
    mean_df = filtered_df.pivot_table(
        values=metric,
        index=["Dataset", "c"],
        # columns=["OCC", '\\alpha'],
        columns=['\\alpha'],
        aggfunc=np.nanmean,
        dropna=False
    )
    
    # Calculate standard error of the mean (SEM)
    sem_df = filtered_df.pivot_table(
        values=metric,
        index=["Dataset", "c"],
        # columns=["OCC", '\\alpha'],
        columns=['\\alpha'],
        aggfunc=lambda x: scipy.stats.sem(x),
        dropna=False
    )
    
    # Format as "mean ± SEM" with exactly 3 decimal places
    mean_formatted = mean_df.applymap(lambda x: f"{x:.3f}")
    sem_formatted = sem_df.applymap(lambda x: f"{x:.3f}")
    formatted_df = mean_formatted + " \pm " + sem_formatted
    
    # Save regular CSV output
    formatted_df.to_csv(f"{metric}.csv")
    
    # Create a LaTeX version with bold formatting for best alpha value and italic for runner-up
    max_mask = pd.DataFrame(False, index=mean_df.index, columns=mean_df.columns)
    runner_up_mask = pd.DataFrame(False, index=mean_df.index, columns=mean_df.columns)
    
    # Find max and runner-up values for each row
    for idx in mean_df.index:
        row_values = mean_df.loc[idx].sort_values(ascending=False)
        if len(row_values) >= 1:
            # Mark highest value
            max_col = row_values.index[0]
            max_mask.at[idx, max_col] = True
            
            # Mark second highest value (runner-up)
            if len(row_values) >= 2:
                runner_up_col = row_values.index[1]
                runner_up_mask.at[idx, runner_up_col] = True
    
    # Apply formatting for LaTeX
    latex_df = formatted_df.copy()
  
    # First wrap ALL values in math mode
    for idx in latex_df.index:
        for col in latex_df.columns:
            latex_df.at[idx, col] = f"${{{latex_df.at[idx, col]}}}$"
    
    # # Apply italic formatting for runner-up (do this second) - using mathit
    # for idx, row in runner_up_mask.iterrows():
    #     for col, is_runner_up in row.items():
    #         if is_runner_up:
    #             latex_df.at[idx, col] = f"$\\mathit{{{formatted_df.at[idx, col]}}}$"
    
    # Apply bold formatting for highest (do this last to override if there's a tie) - using mathbf
    for idx, row in max_mask.iterrows():
        for col, is_max in row.items():
            if is_max:
                latex_df.at[idx, col] = f"$\\mathbf{{{formatted_df.at[idx, col]}}}$"
    
    # Format column headers to use bold math
    latex_header_map = {}
    for col in latex_df.columns:
        # Check if the column name contains math symbols, is a number, or is 'c'
        if '\\' in str(col) or str(col).replace('.', '', 1).isdigit() or str(col) == 'c':
            latex_header_map[col] = f"$\\bm{{{col}}}$"
        else:
            latex_header_map[col] = f"\\textbf{{{col}}}"

    # Apply the header mapping to rename the columns
    latex_df = latex_df.rename(columns=latex_header_map)

    # Format index names to use bold math/text
    index_name_map = {}
    for name in latex_df.index.names:
        # Check if the column name contains math symbols, is a number, or is 'c'
        if '\\' in str(name) or str(name).replace('.', '', 1).isdigit() or str(name) == 'c':
            index_name_map[name] = f"$\\bm{{{name}}}$"
        elif name:
            index_name_map[name] = f"\\textbf{{{name}}}"

    # Apply the index name mapping
    latex_df.index.names = [index_name_map.get(name, name) for name in latex_df.index.names]
    
    # Format column names (MultiIndex level names) to use bold math/text
    column_name_map = {}
    for name in latex_df.columns.names:
        # Check if the column name contains math symbols, is a number, or is 'c'
        if '\\' in str(name) or str(name).replace('.', '', 1).isdigit() or str(name) == 'c':
            column_name_map[name] = f"$\\bm{{{name}}}$"
        elif name:
            column_name_map[name] = f"\\textbf{{{name}}}"
    
    # Apply the column name mapping
    latex_df.columns.names = [column_name_map.get(name, name) for name in latex_df.columns.names]

    # Save the formatted LaTeX output with bold headers and index names
    latex_output = latex_df.to_latex(
        escape=False, 
        multirow=True,
        column_format='ll|llll|l'  # Specify the desired column format
    )

    # Add a note about requiring the bm package
    latex_output = "% Requires \\usepackage{bm} in preamble\n" + latex_output

    latex_output = latex_output.replace("\\cline{1-7}", "\\midrule")

    # Write to file
    with open(f"{metric}.tex", "w") as f:
        f.write(latex_output)
    
    # Display the formatted results
    display(latex_df)

# %%
# for metric in ["Accuracy", 'F1 score']:
#     # Create a pivot table with both mean and sem
#     mean_df = results_df.pivot_table(
#         values=metric,
#         index=["Dataset", "c"],
#         columns=["OCC", '\\alpha'],
#         aggfunc=np.nanmean,
#         dropna=False
#     )
    
#     # Calculate standard error of the mean (SEM)
#     sem_df = results_df.pivot_table(
#         values=metric,
#         index=["Dataset", "c"],
#         columns=["OCC", '\\alpha'],
#         aggfunc=lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))),
#         dropna=False
#     )
    
#     # Format as "mean ± SEM" with exactly 3 decimal places
#     mean_formatted = mean_df.applymap(lambda x: f"{x:.3f}")
#     sem_formatted = sem_df.applymap(lambda x: f"{x:.3f}")
#     formatted_df = mean_formatted + " ± " + sem_formatted
    
#     # Save regular CSV output
#     formatted_df.to_csv(f"{metric}.csv")
    
#     # Create a LaTeX version with bold formatting for best alpha value per OCC method
#     max_mask = pd.DataFrame(False, index=mean_df.index, columns=mean_df.columns)
    
#     # Group by OCC and find max values across different alphas
#     for occ in mean_df.columns.get_level_values('OCC').unique():
#         occ_cols = [col for col in mean_df.columns if col[0] == occ]
#         max_per_row = mean_df[occ_cols].max(axis=1)
        
#         for col in occ_cols:
#             max_mask[col] = mean_df[col].eq(max_per_row)
    
#     # Apply bold formatting for LaTeX
#     latex_df = formatted_df.copy()
#     for idx, row in max_mask.iterrows():
#         for col, is_max in row.items():
#             if is_max:
#                 latex_df.at[idx, col] = f"\\textbf{{{latex_df.at[idx, col]}}}"
    
#     # Save the formatted LaTeX output
#     latex_df.to_latex(f"{metric}.tex", escape=False)
    
#     # Display the formatted results (without bold formatting for display)
#     display(latex_df)

# %%
filtered_df['Actual to expected example ratio'] = filtered_df['Actual n_PU'] / results_df['Expected n_PU']
pivot_df = filtered_df.loc[~(filtered_df["\\alpha"] == 'Normal OCC')].pivot_table(
    values=['Actual to expected example ratio'],
    index=["Dataset", "c"],
    # columns=["OCC", '\\alpha'],
    columns=['\\alpha'],
    aggfunc=np.nanmean,
    dropna=False
)

# Save the formatted LaTeX output with bold headers and index names
latex_output = pivot_df.round(3).to_latex(
    escape=False, 
    multirow=True,
    column_format='ll|llll'  # Specify the desired column format
)

# Add a note about requiring the bm package
latex_output = "% Requires \\usepackage{bm} in preamble\n" + latex_output

latex_output = latex_output.replace("\\cline{1-6}", "\\midrule")

pivot_df.round(3).to_csv(f"n_PU.csv")
# Write to file
with open(f"n_PU.tex", "w") as f:
    f.write(latex_output)

display(pivot_df.round(3))

# %%
