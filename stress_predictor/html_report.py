import os

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
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; text-align: center; }}
            h2 {{ color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }}
            h3 {{ color: #7f8c8d; }}
            .summary-box {{ background-color: white; border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .score {{ font-size: 24px; font-weight: bold; color: #c0392b; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #34495e; color: white; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .stress {{ color: #c0392b; font-weight: bold; }}
            .non-stress {{ color: #27ae60; }}
            .visualization {{ margin-top: 20px; text-align: center; background: white; padding: 15px; border-radius: 8px; border: 1px solid #eee; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .slice-container {{ background: white; border: 1px solid #e0e0e0; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
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
            <div style="margin-top: 15px; border:1px solid #ddd; padding:15px; border-radius:8px; background-color:#f8f9fa; display:inline-block; text-align:left; width: 80%;">
                <h4 style="margin-top:0;">Chart Legend:</h4>
                <ul style="list-style:none; padding-left:0; margin:0;">
                    <li style="margin-bottom:5px;"><strong style="color: #333;">Y-Axis (Probability)</strong>: 0.0 = Non-Stress, 1.0 = High Stress Confidence.</li>
                    <li style="margin-bottom:5px;"><strong style="color: #333;">X-Axis (Position)</strong>: Base pair location within the sequence.</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:15px; background-color:green; margin-right:8px; vertical-align:middle;"></span><strong>Green Dots</strong>: High likelihood of stress motif (>0.8).</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:15px; background-color:yellow; border:1px solid #ccc; margin-right:8px; vertical-align:middle;"></span><strong>Yellow Dots</strong>: Uncertain region (0.4-0.6).</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:15px; background-color:red; margin-right:8px; vertical-align:middle;"></span><strong>Red Dots</strong>: Non-stress background (<0.2).</li>
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
        """

    elif mode == "pr":
        html_content += """
        <div class="summary-box">
            <h2>Promoter Analysis Summary (Promoter Scan Mode)</h2>
            <div style="background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
                <h4 style="margin-top:0;">Chart Legend:</h4>
                 <ul style="list-style:none; padding-left:0; margin:0;">
                    <li style="margin-bottom:5px;"><strong style="color: #333;">Y-Axis (Probability)</strong>: 0.0 = Non-Stress, 1.0 = High Stress Confidence.</li>
                    <li style="margin-bottom:5px;"><strong style="color: #333;">X-Axis (Position)</strong>: Base pair location within this specific slice.</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:15px; background-color:green; margin-right:8px; vertical-align:middle;"></span><strong>Green Dots</strong>: High likelihood of stress motif.</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:15px; background-color:red; margin-right:8px; vertical-align:middle;"></span><strong>Red Dots</strong>: Non-stress background.</li>
                    <li style="margin-bottom:5px;"><span style="display:inline-block; width:15px; height:3px; background-color:blue; margin-right:8px; vertical-align:middle;"></span><strong>Blue Bar</strong>: Detected significant stress region.</li>
                </ul>
            </div>
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
                <h3>{slice_key.replace('_', ' ').title()} (Avg Score: <span class="score" style="font-size: 18px;">{slice_score:.4f}</span>)</h3>
                
                <div class="visualization">
                    <img src="{actual_img}" alt="Visualization for {slice_key}">
                </div>
                
                <div class="regions-table" style="margin-top: 20px;">
                    <h4>Detected Stress Regions in this Slice</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>Start (bp)</th>
                                <th>End (bp)</th>
                                <th>Length (bp)</th>
                                <th>Avg Probability</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            regions = slice_data.get("regions", [])
            if not regions:
                 html_content += "<tr><td colspan='4'>No significant stress regions found in this slice.</td></tr>"
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
            </div>
            """

    html_content += """
    </body>
    </html>
    """
    
    with open(f"{output_dir}/report.html", "w") as f:
        f.write(html_content)
