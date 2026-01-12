from Bio import SeqIO
import json
import os

def read_fasta(fasta_path):
    """
    Read a single sequence from a FASTA file (error if more than one).
    
    Args:
        fasta_path: Path to the FASTA file
    
    Returns:
        str: DNA sequence as string
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty, has multiple sequences, or is malformed
    """
    # Check if file exists
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    # Check if file is empty
    if os.path.getsize(fasta_path) == 0:
        raise ValueError(f"FASTA file is empty: {fasta_path}")
    
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except Exception as e:
        raise ValueError(f"Failed to parse FASTA file '{fasta_path}': {str(e)}")
    
    if len(records) == 0:
        raise ValueError(f"No valid sequences found in FASTA file: {fasta_path}")
    if len(records) > 1:
        raise ValueError(f"FASTA file contains {len(records)} sequences, but only one is allowed: {fasta_path}")
    
    record = records[0]
    return str(record.seq)


def write_output(results, output_path):
    """
    Write prediction results to a JSON file
    results: dict or list of dicts
    """
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/result.json", "w") as f:
        json.dump(results, f, indent=4)

def generate_html_report(results, output_dir, mode="rg"):
    """
    Generates an HTML report for the predictions.
    
    Args:
        results: Dictionary containing the results.
        output_dir: Directory where the report will be saved.
        mode: 'rg' for Region mode, 'pr' for Promoter mode.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stress Predictor Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            .summary-box {{ background-color: #f8f9fa; border: 1px solid #ddd; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .score {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .stress {{ color: #e74c3c; font-weight: bold; }}
            .non-stress {{ color: #27ae60; }}
            .visualization {{ margin-top: 20px; text-align: center; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .slice-container {{ border: 1px solid #eee; padding: 15px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        </style>
    </head>
    <body>
        <h1>Stress Predictor Analysis Report</h1>
        <p>Generated on: {os.path.basename(output_dir)}</p>
    """

    if mode == "rg":
        final_score = results.get("final_score", 0.0)
        html_content += f"""
        <div class="summary-box">
            <h2>Region Analysis Summary</h2>
            <p>Average Stress Confidence Score: <span class="score">{final_score:.4f}</span></p>
        </div>
        
        <div class="visualization">
            <h3>Stress Distribution Visualization</h3>
            <img src="visualization.png" alt="Stress visualization">
            <div style="margin-top: 15px; border:1px solid #ddd; padding:10px; border-radius:5px; background-color:#fff; display:inline-block; text-align:left;">
                <h4 style="margin-top:0;">Legend:</h4>
                <ul style="list-style:none; padding-left:0; margin:0;">
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:15px; background-color:green; margin-right:8px; vertical-align:middle;"></span><strong>High Confidence Stress (>0.8)</strong>: Strong evidence of stress-related sequence.</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:15px; background-color:yellow; border:1px solid #ccc; margin-right:8px; vertical-align:middle;"></span><strong>Uncertain (0.4-0.6)</strong>: Ambiguous region, model is unsure.</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:15px; background-color:red; margin-right:8px; vertical-align:middle;"></span><strong>Non-Stress (<0.2)</strong>: Likely normal/non-stress sequence.</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:3px; background-color:blue; margin-right:8px; vertical-align:middle;"></span><strong>Blue Line</strong>: Automatically detected contiguous stress region.</li>
                </ul>
            </div>
        </div>

        <div class="regions-table">
            <h3>Identified Stress Regions</h3>
            <table>
                <thead>
                    <tr>
                        <th>Start (bp)</th>
                        <th>End (bp)</th>
                        <th>Length (bp)</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
    """
        regions = results.get("regions", [])
        if not regions:
             html_content += "<tr><td colspan='4'>No significant stress regions detected.</td></tr>"
        else:
             for reg in regions:
                 html_content += f"""
                    <tr>
                        <td>{reg['start']}</td>
                        <td>{reg['end']}</td>
                        <td>{reg['length']}</td>
                        <td><span class="stress">{reg['avg_prob']:.4f}</span></td>
                    </tr>
                 """
        
        html_content += """
                </tbody>
            </table>
        </div>

        <h3>Window-based Predictions</h3>
        <table>
            <thead>
                <tr>
                    <th>Window ID</th>
                    <th>Prediction</th>
                    <th>Confidence Score</th>
                    <th>Sequence Snippet</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Sort keys to ensure order (1, 2, 3...)
        keys = [k for k in results.keys() if k.isdigit()]
        keys.sort(key=int)
        
        for k in keys:
            v = results[k]
            label = "Stress (1)" if str(v['label_seq']) == "1" else "Non-Stress (0)"
            cls = "stress" if str(v['label_seq']) == "1" else "non-stress"
            seq_snippet = v['sequence'][:20] + "..." if len(v['sequence']) > 20 else v['sequence']
            
            html_content += f"""
                <tr>
                    <td>{k}</td>
                    <td class="{cls}">{label}</td>
                    <td>{v['score']:.4f}</td>
                    <td style="font-family: monospace;">{seq_snippet}</td>
                </tr>
            """
            
        html_content += """
            </tbody>
        </table>
        """

    elif mode == "pr":
        html_content += """
        <div class="summary-box">
            <h2>Promoter Analysis Summary</h2>
            <p>Analysis performed on multiple sequence slices.</p>
        </div>
        """
        
        # Sort slices
        slices = sorted([k for k in results.keys() if k.startswith("slice_")], key=lambda x: int(x.split('_')[1]))
        
        for slice_key in slices:
            slice_data = results[slice_key]
            slice_score = slice_data.get("final_score", 0.0)
            
            # Find the corresponding image file
            # In promoter_stress_classification loop:
            # save_path = f"slice{slice_id+1}_stride{stride}.png"
            # We don't have stride here easily, but we can look for png files in output dir starting with slice name?
            # Or pass stride into this function?
            # Simpler: List png files in directory that match the slice pattern.
            
            # Extract slice number
            slice_num = slice_key.split('_')[1]
            img_file = f"slice{slice_num}_*.png" 
            # Note: Wildcard won't work in HTML src. We need to find the actual filename.
            
            actual_img = ""
            for f in os.listdir(output_dir):
                if f.startswith(f"slice{slice_num}_") and f.endswith(".png"):
                    actual_img = f
                    break
            
            html_content += f"""
            <div class="slice-container">
                <h3>{slice_key.replace('_', ' ').title()} (Score: <span class="score" style="font-size: 18px;">{slice_score:.4f}</span>)</h3>
                
                <div class="visualization">
                    <img src="{actual_img}" alt="Visualization for {slice_key}">
                </div>
                
                <details>
                    <summary style="cursor: pointer; color: #2980b9; font-weight: bold; margin: 10px 0;">Show Detailed Table</summary>
                    <table>
                        <thead>
                            <tr>
                                <th>Window</th>
                                <th>Prediction</th>
                                <th>Confidence</th>
                                <th>Sequence</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            w_keys = [k for k in slice_data.keys() if k.isdigit()]
            w_keys.sort(key=int)
            
            for k in w_keys:
                v = slice_data[k]
                label = "Stress (1)" if str(v['label_seq']) == "1" else "Non-Stress (0)"
                cls = "stress" if str(v['label_seq']) == "1" else "non-stress"
                seq_snippet = v['sequence'][:20] + "..." if len(v['sequence']) > 20 else v['sequence']
                
                html_content += f"""
                            <tr>
                                <td>{k}</td>
                                <td class="{cls}">{label}</td>
                                <td>{v['score']:.4f}</td>
                                <td style="font-family: monospace;">{seq_snippet}</td>
                            </tr>
                """
            
            html_content += """
                        </tbody>
                    </table>
                </details>
            </div>
            """

    html_content += """
    </body>
    </html>
    """
    
    with open(f"{output_dir}/report.html", "w") as f:
        f.write(html_content)

