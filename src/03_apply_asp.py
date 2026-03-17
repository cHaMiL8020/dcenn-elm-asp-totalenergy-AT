import pandas as pd
import clingo
import os
import holidays

def run_asp_safety_check(df_subset, rules_path):
    print("Initializing ASP Clingo Solver...")
    # Initialize Clingo control object to find all models
    ctl = clingo.Control(["0"]) 
    ctl.load(rules_path)

    print("Translating neural network predictions into logical facts...")
    years = pd.to_datetime(df_subset['timestamp']).dt.year.unique().tolist()
    at_holidays = holidays.Austria(years=years)

    facts = ""
    # We use enumerate to create a sequential time index (T) for the solver
    for idx, row in enumerate(df_subset.itertuples()):
        t = idx
        # ASP natively prefers integers, so we cast the floats
        pred_val = int(row.predicted_generation)
        actual_val = int(row.power_generation)
        cglo_val = int(row.cglo)

        # Generate standard logic programming facts
        facts += f"pred({t}, {pred_val}).\n"
        facts += f"actual({t}, {actual_val}).\n"
        facts += f"cglo({t}, {cglo_val}).\n"

        # Mark Austrian public holidays as a logical fact
        ts_date = pd.to_datetime(row.timestamp).date()
        if ts_date in at_holidays:
            facts += f"holiday({t}).\n"

    # Add facts to the knowledge base and ground the rules
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])

    print("Solving for logical anomalies...")
    anomalies = []
    
    # Callback function to extract flagged anomalies from the solver
    def on_model(m):
        for symbol in m.symbols(atoms=True):
            if symbol.name == "anomaly":
                time_idx = symbol.arguments[0].number
                reason = symbol.arguments[1].string
                anomalies.append({'time_idx': time_idx, 'reason': reason})

    ctl.solve(on_model=on_model)
    return pd.DataFrame(anomalies)

def main():
    # Load predictions
    preds_path = "data/predictions_2024.csv"
    rules_path = "rules/grid_rules.lp"
    
    if not os.path.exists(preds_path):
        print(f"Error: {preds_path} not found. Run phase 2 first.")
        return

    df = pd.read_csv(preds_path)
    
    # Note: Running the ASP solver on a massive dataset (35,000+ rows) at once 
    # can consume high memory. We will test it on the first month (approx 2976 rows).
    df_test_window = df.head(2976).copy()
    
    # Run the neuro-symbolic check
    anomaly_df = run_asp_safety_check(df_test_window, rules_path)
    
    if anomaly_df.empty:
        print("\nSUCCESS: No physical or logical anomalies found in the AI predictions for this window.")
    else:
        # Map the original timestamps back to the anomalies
        anomaly_df = anomaly_df.merge(
            df_test_window[['timestamp', 'power_generation', 'predicted_generation']].reset_index(),
            left_on='time_idx', 
            right_on='index'
        ).drop(columns=['index', 'time_idx'])
        
        print(f"\n[ALERT] ASP Solver flagged {len(anomaly_df)} logical violations!")
        print("Here is a sample of the anomalies:")
        print(anomaly_df.head(10).to_string(index=False))
        
        # Save anomalies for analysis
        anomaly_df.to_csv("data/flagged_anomalies.csv", index=False)
        print("\nFull anomaly report saved to data/flagged_anomalies.csv")

if __name__ == "__main__":
    main()