import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot HSMC scaling data")
    parser.add_argument("csv_file", type=str, help="Path to the benchmark CSV file")
    parser.add_argument("-o", "--output", type=str, default="scaling_curve.png", help="Output image path")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df = df.sort_values(by="particles")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df["particles"], df["throughput"] / 1e6, marker='o', color='crimson', linewidth=2)
    plt.title("Simulation Throughput")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Million Steps / Second")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(df["particles"], df["elapsed"], marker='s', color='royalblue', linewidth=2)
    plt.title("Total Runtime Scaling")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Chart successfully saved to: {args.output}")

if __name__ == "__main__":
    main()
