import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_student_plots():
    print("--- [START] Generating Student Diagnostic Visualizations ---")
    
    # Load dataset
    csv_path = os.path.join("data", "student_performance.csv")
    df = pd.read_csv(csv_path)
    
    # 1. Matplotlib Scatter Plot: Visually identifying pass/fail clusters
    # We use plt.scatter() in order to plot individual student coordinates comparing midterm_score vs assignment_score
    plt.figure(figsize=(6, 4))
    colors = df["pass_target"].map({1: "green", 0: "red"})
    plt.scatter(df["midterm_score"], df["assignment_score"], c=colors, s=100, alpha=0.8)
    plt.title("Student Performance: Midterm vs Assignment Scores\n(Green = Pass, Red = Fail)")
    plt.xlabel("Midterm Score")
    plt.ylabel("Assignment Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save chart to verify layout in artifact checks
    plt.savefig("student_scatter_plot.png")
    plt.close()
    print("[Success] Scatter plot saved as student_scatter_plot.png")
    
    # 2. Seaborn Heatmap: Identifying correlation patterns between features
    # We use sns.heatmap() in order to convert the numerical correlations matrix into a quick visual color gradient
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Student Attributes")
    
    plt.savefig("student_correlation_heatmap.png")
    plt.close()
    print("[Success] Correlation heatmap saved as student_correlation_heatmap.png")
    print("--- [SUCCESS] Visualization Task Complete ---")

if __name__ == "__main__":
    generate_student_plots()
