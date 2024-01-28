import os
import pandas as pd

def export_metrics(metadata_list, metadata_list_full, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # convert the dictionary in a pandas data frame
    df = pd.DataFrame.from_dict(metadata_list, orient='columns')
    if "Language" in df.columns:
        df_lang = df.copy()
        df_lang = df_lang.groupby('Language').mean().reset_index(level=0)
        df_lang = df_lang.fillna(0)
        # round 4 decimals points 
        df_lang = df_lang.round(4)
        # sort by language name
        df_lang = df_lang.sort_values('Language')

        # add stds
        df_lang_with_std = df_lang.copy()
        df_std = df.groupby('Language').std().fillna(0).round(2).reset_index(level=0)
        df_lang_with_std["UTMOS"] = df_lang_with_std["UTMOS"].astype('str') + " ± " + df_std["UTMOS"].astype('str')

        # add mean line with std
        df_lang_with_std.loc['mean'] = df_lang.mean().round(4)
        # add language std to the mean
        df_lang_with_std.loc['mean', 'UTMOS']  = str(df_lang_with_std.loc['mean']["UTMOS"]) + " ± " + df_lang["UTMOS"].std().round(2).astype('str')

        markdown_table = df_lang_with_std.to_markdown()
        print(markdown_table)
        if out_path is not None:
            markdown_table_path=out_path.replace(".csv", "_markdown_table.txt")
            with open(markdown_table_path, "w") as text_file:
                text_file.write(markdown_table)
            print("Language Markdown results Table saved at:", markdown_table_path)
            df_lang_with_std.to_csv(out_path, index=True, sep=',', encoding='utf-8')
            print("CSV with Language Ranking saved at:", out_path)

    df_debug_full = pd.DataFrame.from_dict(metadata_list_full, orient='columns')
    out_path = os.path.join(os.path.dirname(out_path), "metrics_utterance_level.csv")
    df_debug_full.to_csv(out_path, index=True, sep=',', encoding='utf-8')
    print("CSV with all samples metrics saved at:", out_path)
