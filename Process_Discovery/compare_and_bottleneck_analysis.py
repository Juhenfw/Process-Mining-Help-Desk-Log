import os
import json
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter

warnings.filterwarnings('ignore')

class ProcessMiningPaper:
    """Simplified Process Mining Framework for Academic Paper"""
    
    def __init__(self, csv_path, output_dir='output_paper'):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.event_log = None
        self.models = {}  # Store models: {'inductive': (net, im, fm), ...}
        self.metrics = {}  # Store all metrics
        self.report = []
        
        # Create directories
        for subdir in ['visualizations', 'models', 'reports', 'data']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        print(f"[INIT] Output directory: {output_dir}")
    
    def log(self, msg):
        """Log message"""
        print(msg)
        self.report.append(msg)
    
    # ==================== STEP 1: LOAD DATA ====================
    def load_and_preprocess(self):
        """Load CSV and convert to PM4Py event log"""
        self.log("\n" + "="*60)
        self.log("STEP 1: LOADING & PREPROCESSING DATA")
        self.log("="*60)
        
        try:
            # Read CSV
            df = pd.read_csv(self.csv_path)
            self.log(f"✓ Loaded {len(df)} events from CSV")
            
            # Preprocess
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(['Case ID', 'timestamp'])
            df = df.dropna(subset=['Case ID', 'Activity', 'timestamp'])
            
            # Rename to PM4Py standard
            df_renamed = df.rename(columns={
                'Case ID': 'case:concept:name',
                'Activity': 'concept:name',
                'timestamp': 'time:timestamp'
            })
            
            # Convert to event log
            self.event_log = pm4py.convert_to_event_log(df_renamed)
            
            self.log(f"✓ Event log created:")
            self.log(f"  - Cases: {len(self.event_log)}")
            self.log(f"  - Events: {sum(len(trace) for trace in self.event_log)}")
            self.log(f"  - Variants: {len(pm4py.get_variants(self.event_log))}")
            
            # Export to XES for DISCO
            xes_path = os.path.join(self.output_dir, 'models', 'eventlog_for_disco.xes')
            pm4py.write_xes(self.event_log, xes_path)
            self.log(f"✓ Exported XES for DISCO: {xes_path}")
            
            return True
            
        except Exception as e:
            self.log(f"✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== STEP 2: DISCOVER MODELS ====================
    def discover_all_models(self):
        """Discover process models using multiple algorithms"""
        self.log("\n" + "="*60)
        self.log("STEP 2: PROCESS DISCOVERY (MULTI-ALGORITHM)")
        self.log("="*60)
        
        try:
            # 1. Inductive Miner (Main algorithm for paper)
            self.log("\n[1/3] Discovering with Inductive Miner...")
            tree = inductive_miner.apply(self.event_log)
            net, im, fm = pt_converter.apply(tree)
            self.models['inductive'] = (net, im, fm, tree)
            self.log(f"  ✓ Places: {len(net.places)}, Transitions: {len(net.transitions)}")
            
            # 2. Heuristics Miner (For comparison)
            self.log("\n[2/3] Discovering with Heuristics Miner...")
            heu_net, heu_im, heu_fm = heuristics_miner.apply(
                self.event_log,
                parameters={
                    heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5,
                    heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.65
                }
            )
            self.models['heuristics'] = (heu_net, heu_im, heu_fm, None)
            self.log(f"  ✓ Places: {len(heu_net.places)}, Transitions: {len(heu_net.transitions)}")
            
            # 3. Alpha Miner (Baseline for comparison)
            self.log("\n[3/3] Discovering with Alpha Miner...")
            alpha_net, alpha_im, alpha_fm = alpha_miner.apply(self.event_log)
            self.models['alpha'] = (alpha_net, alpha_im, alpha_fm, None)
            self.log(f"  ✓ Places: {len(alpha_net.places)}, Transitions: {len(alpha_net.transitions)}")
            
            self.log(f"\n✓ Discovered 3 models successfully")
            return True
            
        except Exception as e:
            self.log(f"✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== STEP 3: CONFORMANCE CHECKING ====================
    def conformance_checking(self):
        """Advanced conformance checking with token replay"""
        self.log("\n" + "="*60)
        self.log("STEP 3: CONFORMANCE CHECKING")
        self.log("="*60)
        
        for algo_name, (net, im, fm, tree) in self.models.items():
            self.log(f"\n[{algo_name.upper()}] Conformance Analysis:")
            
            try:
                # Token Replay (Fitness)
                replay_result = token_replay.apply(self.event_log, net, im, fm)

                # Extract fitness safely
                if isinstance(replay_result, dict):
                    fitness = replay_result.get('average_trace_fitness',
                                                replay_result.get('log_fitness', 0.0))
                elif isinstance(replay_result, (int, float)):
                    fitness = float(replay_result)
                else:
                    # fallback kalau list/tipe lain
                    fitness = 0.0

                self.log(f"  Token Replay Fitness: {fitness:.4f}")
                
                # Try alignments (optional, may be slow)
                try:
                    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
                    self.log(f"  Computing alignments...")
                    aligned_traces = alignments.apply(self.event_log, net, im, fm)
                    
                    # Calculate alignment-based metrics
                    total_cost = 0
                    total_moves = 0
                    for trace_align in aligned_traces:
                        if isinstance(trace_align, dict):
                            total_cost += trace_align.get('cost', 0)
                            total_moves += len(trace_align.get('alignment', []))
                    
                    avg_cost = total_cost / len(aligned_traces) if aligned_traces else 0
                    alignment_fitness = 1 - (total_cost / max(1, total_moves))
                    
                    self.log(f"  Alignment Fitness: {alignment_fitness:.4f}")
                    self.log(f"  Average Alignment Cost: {avg_cost:.2f}")
                    
                    # Store alignment metrics
                    if algo_name not in self.metrics:
                        self.metrics[algo_name] = {}
                    self.metrics[algo_name]['alignment_fitness'] = float(alignment_fitness)
                    self.metrics[algo_name]['alignment_cost'] = float(avg_cost)
                    
                except Exception as align_e:
                    self.log(f"  ⚠ Alignments skipped: {str(align_e)[:50]}")
                
                # Store basic metrics
                if algo_name not in self.metrics:
                    self.metrics[algo_name] = {}
                
                self.metrics[algo_name]['token_replay_fitness'] = float(fitness)
                
            except Exception as e:
                self.log(f"  ✗ Conformance failed: {str(e)}")
                # Initialize empty metrics to prevent crashes later
                if algo_name not in self.metrics:
                    self.metrics[algo_name] = {}
                continue
        
        return True
    
    # ==================== STEP 4: EVALUATE QUALITY ====================
    def evaluate_quality(self):
        """Evaluate all quality metrics (fitness, precision, generalization, simplicity)"""
        self.log("\n" + "="*60)
        self.log("STEP 4: MODEL QUALITY EVALUATION")
        self.log("="*60)
        
        for algo_name, (net, im, fm, tree) in self.models.items():
            self.log(f"\n[{algo_name.upper()}] Quality Metrics:")
            
            try:
                # Fitness
                fitness_result = replay_fitness.apply(self.event_log, net, im, fm)
                if isinstance(fitness_result, dict):
                    fitness = fitness_result.get('average_trace_fitness',
                             fitness_result.get('log_fitness', 0))
                else:
                    fitness = fitness_result
                
                # Precision
                precision = precision_evaluator.apply(self.event_log, net, im, fm)
                
                # Generalization
                generalization = generalization_evaluator.apply(self.event_log, net, im, fm)
                
                # Simplicity
                simplicity = simplicity_evaluator.apply(net)
                
                # F1-Score (harmonic mean of fitness and precision)
                if (fitness + precision) > 0:
                    f1_score = 2 * (fitness * precision) / (fitness + precision)
                else:
                    f1_score = 0
                
                self.log(f"  Fitness:        {fitness:.4f}")
                self.log(f"  Precision:      {precision:.4f}")
                self.log(f"  Generalization: {generalization:.4f}")
                self.log(f"  Simplicity:     {simplicity:.4f}")
                self.log(f"  F1-Score:       {f1_score:.4f}")
                
                # Store metrics
                if algo_name not in self.metrics:
                    self.metrics[algo_name] = {}
                    
                self.metrics[algo_name].update({
                    'fitness': float(fitness),
                    'precision': float(precision),
                    'generalization': float(generalization),
                    'simplicity': float(simplicity),
                    'f1_score': float(f1_score)
                })
                
            except Exception as e:
                self.log(f"  ✗ Evaluation failed: {str(e)}")
                # Set default values to prevent crashes
                if algo_name not in self.metrics:
                    self.metrics[algo_name] = {}
                self.metrics[algo_name].update({
                    'fitness': 0.0,
                    'precision': 0.0,
                    'generalization': 0.0,
                    'simplicity': 0.0,
                    'f1_score': 0.0
                })
                continue
        
        return True
    
    # ==================== STEP 5: VISUALIZATIONS (FIXED) ====================
    def create_visualizations(self):
        """Create essential visualizations for paper"""
        self.log("\n" + "="*60)
        self.log("STEP 5: CREATING VISUALIZATIONS")
        self.log("="*60)
        
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        
        # 1. Petri Nets for each algorithm (FIXED PARAMETERS)
        for algo_name, (net, im, fm, tree) in self.models.items():
            try:
                self.log(f"\n[{algo_name.upper()}] Creating Petri Net...")
                
                # Method 1: Try with format parameter only
                try:
                    gviz = pn_visualizer.apply(
                        net, im, fm,
                        parameters={'format': 'png'},
                        variant=pn_visualizer.Variants.FREQUENCY,
                        log=self.event_log
                    )
                except:
                    # Method 2: Minimal parameters
                    gviz = pn_visualizer.apply(net, im, fm, log=self.event_log)
                
                output = os.path.join(viz_dir, f'petri_net_{algo_name}.png')
                pn_visualizer.save(gviz, output)
                self.log(f"  ✓ Saved: {output}")
                
            except Exception as e:
                self.log(f"  ✗ Failed: {str(e)[:100]}")
        
        # 2. DFG (FIXED SYNTAX)
        try:
            self.log(f"\n[DFG] Creating Directly-Follows Graph...")
            dfg, start_activities, end_activities = pm4py.discover_dfg(self.event_log)
            
            # FIXED: Use correct parameter structure
            gviz = dfg_visualizer.apply(
                dfg,
                log=self.event_log,
                parameters={
                    'format': 'png',
                    'start_activities': start_activities,
                    'end_activities': end_activities
                }
            )
            
            output = os.path.join(viz_dir, 'dfg_frequency.png')
            dfg_visualizer.save(gviz, output)
            self.log(f"  ✓ Saved: {output}")
            
        except Exception as e:
            self.log(f"  ✗ DFG Failed: {str(e)[:100]}")
            # Try alternative method
            try:
                self.log(f"  Trying alternative DFG method...")
                from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
                dfg = dfg_discovery.apply(self.event_log)
                sa = pm4py.get_start_activities(self.event_log)
                ea = pm4py.get_end_activities(self.event_log)
                
                gviz = dfg_visualizer.apply(dfg, parameters={'format': 'png'})
                output = os.path.join(viz_dir, 'dfg_frequency.png')
                dfg_visualizer.save(gviz, output)
                self.log(f"  ✓ Saved (alternative): {output}")
            except Exception as e2:
                self.log(f"  ✗ Alternative also failed: {str(e2)[:100]}")
        
        # 3. Performance DFG (FIXED)
        try:
            self.log(f"\n[DFG Performance] Creating Performance DFG...")
            from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
            
            dfg_perf = dfg_discovery.apply(self.event_log, 
                                          variant=dfg_discovery.Variants.PERFORMANCE)
            
            gviz = dfg_visualizer.apply(
                dfg_perf,
                variant=dfg_visualizer.Variants.PERFORMANCE,
                parameters={'format': 'png'}
            )
            
            output = os.path.join(viz_dir, 'dfg_performance.png')
            dfg_visualizer.save(gviz, output)
            self.log(f"  ✓ Saved: {output}")
            
        except Exception as e:
            self.log(f"  ✗ Performance DFG Failed: {str(e)[:100]}")
        
        self.log(f"\n✓ Visualizations completed")
        return True
    
    # ==================== STEP 6: EXPORT MODELS ====================
    def export_models(self):
        """Export models in standard formats"""
        self.log("\n" + "="*60)
        self.log("STEP 6: EXPORTING MODELS")
        self.log("="*60)
        
        models_dir = os.path.join(self.output_dir, 'models')
        
        for algo_name, (net, im, fm, tree) in self.models.items():
            try:
                # Export PNML (Petri Net)
                pnml_path = os.path.join(models_dir, f'{algo_name}_model.pnml')
                pm4py.write_pnml(net, im, fm, pnml_path)
                self.log(f"✓ {algo_name.upper()}: PNML exported to {pnml_path}")
                
                # Export BPMN (if possible)
                try:
                    bpmn_model = pm4py.convert_to_bpmn(net, im, fm)
                    bpmn_path = os.path.join(models_dir, f'{algo_name}_model.bpmn')
                    pm4py.write_bpmn(bpmn_model, bpmn_path)
                    self.log(f"✓ {algo_name.upper()}: BPMN exported to {bpmn_path}")
                except:
                    pass
                    
            except Exception as e:
                self.log(f"✗ {algo_name.upper()}: Export failed - {str(e)}")
        
        return True
    
    # ==================== STEP 7: COMPARISON REPORT (FIXED) ====================
    def generate_comparison_report(self):
        """Generate comparison report for paper"""
        self.log("\n" + "="*60)
        self.log("STEP 7: GENERATING COMPARISON REPORT")
        self.log("="*60)
        
        # Check if metrics exist
        if not self.metrics or all(not m for m in self.metrics.values()):
            self.log("✗ No metrics available - skipping report generation")
            return False
        
        # Export metrics to JSON
        json_path = os.path.join(self.output_dir, 'data', 'metrics_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.log(f"✓ Metrics JSON: {json_path}")
        
        # Create comparison table
        report_path = os.path.join(self.output_dir, 'reports', 'algorithm_comparison.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ALGORITHM COMPARISON REPORT FOR PAPER\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Table header
            f.write(f"{'Metric':<25} {'Inductive':<15} {'Heuristics':<15} {'Alpha':<15}\n")
            f.write("-"*70 + "\n")
            
            # Metrics comparison
            metrics_list = ['fitness', 'precision', 'generalization', 'simplicity', 'f1_score',
                          'token_replay_fitness', 'alignment_fitness', 'alignment_cost']
            
            for metric in metrics_list:
                values = []
                for algo in ['inductive', 'heuristics', 'alpha']:
                    val = self.metrics.get(algo, {}).get(metric, None)
                    if val is not None and isinstance(val, (int, float)):
                        values.append(f"{val:.4f}")
                    else:
                        values.append("N/A")
                
                f.write(f"{metric.capitalize():<25} {values[0]:<15} {values[1]:<15} {values[2]:<15}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION FOR PAPER:\n")
            f.write("-"*80 + "\n")
            
            # Find best model (with safety checks)
            try:
                valid_algos = [a for a in self.metrics if self.metrics[a].get('fitness', 0) > 0]
                
                if valid_algos:
                    best_fitness = max(valid_algos, key=lambda x: self.metrics[x].get('fitness', 0))
                    best_precision = max(valid_algos, key=lambda x: self.metrics[x].get('precision', 0))
                    best_f1 = max(valid_algos, key=lambda x: self.metrics[x].get('f1_score', 0))
                    
                    f.write(f"• Best Fitness: {best_fitness.upper()} "
                           f"({self.metrics[best_fitness]['fitness']:.4f})\n")
                    f.write(f"• Best Precision: {best_precision.upper()} "
                           f"({self.metrics[best_precision]['precision']:.4f})\n")
                    f.write(f"• Best F1-Score: {best_f1.upper()} "
                           f"({self.metrics[best_f1]['f1_score']:.4f})\n")
                else:
                    f.write("• No valid metrics computed\n")
                    
            except Exception as e:
                f.write(f"• Error computing best models: {str(e)}\n")
            
            f.write("\n" + "="*80 + "\n")
            
        self.log(f"✓ Comparison report: {report_path}")
        
        # Create comparison chart
        self.create_comparison_chart()
        
        return True
    
    def create_comparison_chart(self):
        """Create visual comparison chart for paper"""
        try:
            algorithms = [a for a in self.metrics if self.metrics[a].get('fitness', 0) > 0]
            
            if not algorithms:
                self.log("  ⚠ No valid metrics for chart")
                return
            
            metrics_to_plot = ['fitness', 'precision', 'generalization', 'simplicity']
            
            data = {metric: [self.metrics[algo].get(metric, 0) for algo in algorithms] 
                   for metric in metrics_to_plot}
            
            x = np.arange(len(algorithms))
            width = 0.2
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            for i, (metric, values) in enumerate(data.items()):
                ax.bar(x + i*width, values, width, label=metric.capitalize(), 
                      color=colors[i], edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Process Mining Algorithm Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels([a.capitalize() for a in algorithms])
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            chart_path = os.path.join(self.output_dir, 'visualizations', 'algorithm_comparison.png')
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log(f"✓ Comparison chart: {chart_path}")
        except Exception as e:
            self.log(f"✗ Chart creation failed: {str(e)}")
    
    # ==================== MAIN PIPELINE ====================
    def run(self):
        """Execute complete pipeline"""
        self.log("\n" + "="*80)
        self.log("PROCESS MINING PIPELINE FOR ACADEMIC PAPER")
        self.log("Analisis dan Pengoptimalan Jalur Proses Tiket Help-desk")
        self.log("="*80)
        
        start_time = datetime.now()
        
        # Execute all steps
        steps = [
            ("Load & Preprocess", self.load_and_preprocess),
            ("Discover Models", self.discover_all_models),
            ("Conformance Checking", self.conformance_checking),
            ("Quality Evaluation", self.evaluate_quality),
            ("Create Visualizations", self.create_visualizations),
            ("Export Models", self.export_models),
            ("Generate Report", self.generate_comparison_report)
        ]
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    self.log(f"\n⚠ Warning at: {step_name} (continuing...)")
            except Exception as e:
                self.log(f"\n✗ Error at {step_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        
        self.log("\n" + "="*80)
        self.log("PIPELINE COMPLETED!")
        self.log("="*80)
        self.log(f"⏱  Duration: {duration:.1f} seconds")
        self.log(f"\n📁 Output Structure:")
        self.log(f"  {self.output_dir}/")
        self.log(f"  ├── visualizations/")
        self.log(f"  │   ├── petri_net_inductive.png")
        self.log(f"  │   ├── petri_net_heuristics.png")
        self.log(f"  │   ├── petri_net_alpha.png")
        self.log(f"  │   ├── dfg_frequency.png")
        self.log(f"  │   ├── dfg_performance.png")
        self.log(f"  │   └── algorithm_comparison.png  ← FOR PAPER")
        self.log(f"  ├── models/")
        self.log(f"  │   ├── eventlog_for_disco.xes  ← IMPORT TO DISCO")
        self.log(f"  │   ├── inductive_model.pnml")
        self.log(f"  │   ├── heuristics_model.pnml")
        self.log(f"  │   └── alpha_model.pnml")
        self.log(f"  ├── reports/")
        self.log(f"  │   └── algorithm_comparison.txt  ← FOR PAPER TABLE")
        self.log(f"  └── data/")
        self.log(f"      └── metrics_comparison.json")
        
        self.log("\n📊 Quick Metrics:")
        for algo in self.metrics:
            m = self.metrics[algo]
            self.log(f"  [{algo.upper()}] F1={m.get('f1_score', 0):.3f} | "
                    f"Fitness={m.get('fitness', 0):.3f} | "
                    f"Precision={m.get('precision', 0):.3f}")
        
        self.log("\n📝 Next Steps for Paper:")
        self.log("  1. Review algorithm_comparison.txt for table")
        self.log("  2. Use algorithm_comparison.png as Figure")
        self.log("  3. Import eventlog_for_disco.xes to DISCO for comparison")
        self.log("  4. Analyze Petri Nets for process insights")
        self.log("  5. Use DFG performance for bottleneck discussion")
        
        # Save full report
        report_path = os.path.join(self.output_dir, 'reports', 'full_execution_log.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        return True


# ==================== EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SIMPLIFIED PROCESS MINING FOR ACADEMIC PAPER (FIXED VERSION)")
    print("="*80 + "\n")
    
    # Configuration
    CSV_PATH = 'D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/finale_preprocessed_renamed.csv'
    OUTPUT_DIR = 'D:/VSCODE/myenv2/share/Mining_Final_Project/output_paper'
    
    # Run pipeline
    pm = ProcessMiningPaper(CSV_PATH, OUTPUT_DIR)
    success = pm.run()
    
    if success:
        print("\n" + "="*80)
        print("✓ ALL DONE! Check output_paper/ folder")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠ PIPELINE COMPLETED WITH WARNINGS - Check output")
        print("="*80)
