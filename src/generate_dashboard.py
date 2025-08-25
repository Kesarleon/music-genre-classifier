import os
import base64

def get_image_as_base64(path):
    """Reads an image file and returns it as a base64 encoded string."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def generate_dashboard(metrics_dir="metrics", output_filename="dashboard.html"):
    """
    Generates an HTML dashboard from the model evaluation metrics.
    """
    # File paths
    report_path = os.path.join(metrics_dir, "classification_report.txt")
    report_smote_path = os.path.join(metrics_dir, "classification_report_smote.txt")
    cm_path = os.path.join(metrics_dir, "confusion_matrix.png")
    cm_smote_path = os.path.join(metrics_dir, "confusion_matrix_smote.png")

    # Read classification reports
    try:
        with open(report_path, "r") as f:
            report_base = f.read()
    except FileNotFoundError:
        report_base = "Classification report not found."

    try:
        with open(report_smote_path, "r") as f:
            report_smote = f.read()
    except FileNotFoundError:
        report_smote = "SMOTE classification report not found."

    # Encode images
    cm_base64 = get_image_as_base64(cm_path)
    cm_smote_base64 = get_image_as_base64(cm_smote_path)

    # HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Performance Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }}
            h1 {{ text-align: center; color: #333; }}
            .container {{ display: flex; justify-content: space-around; flex-wrap: wrap; }}
            .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; width: 45%; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); background-color: #fff;}}
            h2 {{ border-bottom: 2px solid #eee; padding-bottom: 8px; color: #444; }}
            pre {{ background-color: #fdfdfd; border: 1px solid #eee; padding: 1em; white-space: pre-wrap; font-size: 0.9em; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>Music Genre Classifier - Performance Dashboard</h1>
        <div class="container">
            <div class="card">
                <h2>Model without SMOTE</h2>
                <h3>Classification Report</h3>
                <pre>{report_base}</pre>
                <h3>Confusion Matrix</h3>
                {f'<img src="data:image/png;base64,{cm_base64}">' if cm_base64 else '<p>Image not found.</p>'}
            </div>
            <div class="card">
                <h2>Model with SMOTE</h2>
                <h3>Classification Report</h3>
                <pre>{report_smote}</pre>
                <h3>Confusion Matrix</h3>
                {f'<img src="data:image/png;base64,{cm_smote_base64}">' if cm_smote_base64 else '<p>Image not found.</p>'}
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_filename, "w") as f:
        f.write(html_content)

    print(f"Dashboard generated and saved to {output_filename}")

if __name__ == "__main__":
    # To test this script, we would need the metrics files.
    # The orchestration script will ensure they exist before calling this.
    if not os.path.exists("metrics"):
        print("Metrics directory not found. Please run the evaluation script first.")
    else:
        generate_dashboard()
