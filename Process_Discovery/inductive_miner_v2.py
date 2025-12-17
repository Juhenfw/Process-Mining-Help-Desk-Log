"""
Process Discovery - Enhanced Visualization Version
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# PM4PY imports
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.statistics.variants.log import get as variants_get
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.filtering.dfg import dfg_filtering

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class ProcessDiscoveryEnhanced:
    """
    Enhanced Process Discovery dengan visualisasi yang lebih rapi
    """
    
    def __init__(self, input_file='finale_preprocessed.csv', 
                 output_dir='process_discovery_output',
                 case_id_col=None,
                 activity_col=None,
                 timestamp_col=None):
        """
        Inisialisasi Process Discovery
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.df = None
        self.event_log = None
        self.process_model = None
        self.initial_marking = None
        self.final_marking = None
        self.process_tree = None
        self.bpmn_model = None
        self.dfg = None
        self.metrics = {}
        self.report = []
        
        # Create organized output directories
        self.dir_visualizations = os.path.join(output_dir, 'visualizations')
        self.dir_models = os.path.join(output_dir, 'models')
        self.dir_reports = os.path.join(output_dir, 'reports')
        self.dir_data = os.path.join(output_dir, 'data')
        
        for directory in [self.output_dir, self.dir_visualizations, 
                         self.dir_models, self.dir_reports, self.dir_data]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Directory '{directory}' created")
    
    def log_report(self, message, print_msg=True):
        """Catat pesan ke report"""
        self.report.append(message)
        if print_msg:
            print(message)
    
    def add_separator(self):
        """Tambahkan separator"""
        separator = "=" * 80
        self.log_report(separator)
    
    # ========== LOAD & CONVERT (sama seperti sebelumnya) ==========
    def load_preprocessed_data(self):
        """Load data hasil preprocessing"""
        self.add_separator()
        self.log_report("\n📂 LOADING PREPROCESSED DATA\n")
        
        try:
            self.df = pd.read_csv(self.input_file)
            self.log_report(f"✅ Data loaded: '{self.input_file}'")
            self.log_report(f"   Rows: {len(self.df):,}")
            self.log_report(f"   Columns: {len(self.df.columns)}")
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def _detect_column(self, possible_names):
        """Deteksi nama kolom"""
        col_lower = {col.lower(): col for col in self.df.columns}
        for name in possible_names:
            if name.lower() in col_lower:
                return col_lower[name.lower()]
        
        # Partial match
        for col in self.df.columns:
            col_lower_str = col.lower()
            for name in possible_names:
                if name.lower() in col_lower_str:
                    return col
        return None
    
    def convert_to_event_log(self):
        """Convert pandas DataFrame ke PM4PY event log format"""
        self.add_separator()
        self.log_report("\n🔄 CONVERTING TO EVENT LOG FORMAT\n")
        
        try:
            # Detect or use specified columns
            case_col = self.case_id_col or self._detect_column([
                'case_id', 'caseid', 'case', 'case id'
            ])
            activity_col = self.activity_col or self._detect_column([
                'activity', 'activity_name', 'event', 'task'
            ])
            timestamp_col = self.timestamp_col or self._detect_column([
                'timestamp', 'time', 'datetime', 'date', 'complete timestamp'
            ])
            
            if not all([case_col, activity_col, timestamp_col]):
                self.log_report("❌ ERROR: Cannot detect required columns!")
                return False
            
            self.log_report(f"✅ Columns detected:")
            self.log_report(f"   Case ID: {case_col}")
            self.log_report(f"   Activity: {activity_col}")
            self.log_report(f"   Timestamp: {timestamp_col}")
            
            # Convert
            df_log = self.df.copy()
            df_log = df_log.rename(columns={
                case_col: 'case:concept:name',
                activity_col: 'concept:name',
                timestamp_col: 'time:timestamp'
            })
            df_log['time:timestamp'] = pd.to_datetime(df_log['time:timestamp'])
            
            self.event_log = log_converter.apply(
                df_log, 
                variant=log_converter.Variants.TO_EVENT_LOG
            )
            
            self.log_report(f"\n✅ Conversion successful!")
            self.log_report(f"   Total cases: {len(self.event_log):,}")
            self.log_report(f"   Total events: {sum([len(trace) for trace in self.event_log]):,}")
            
            self.metrics['total_cases'] = len(self.event_log)
            self.metrics['total_events'] = sum([len(trace) for trace in self.event_log])
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    # ========== DISCOVERY ==========
    def discover_process_model(self, algorithm='inductive'):
        """Discover process model"""
        self.add_separator()
        self.log_report(f"\n🔍 PROCESS DISCOVERY - {algorithm.upper()} MINER\n")
        
        try:
            if algorithm == 'inductive':
                self.process_tree = inductive_miner.apply(self.event_log)
                self.process_model, self.initial_marking, self.final_marking = \
                    pt_converter.apply(self.process_tree, variant=pt_converter.Variants.TO_PETRI_NET)
                
                self.log_report(f"✅ Inductive Miner successful!")
                self.log_report(f"   Places: {len(self.process_model.places)}")
                self.log_report(f"   Transitions: {len(self.process_model.transitions)}")
                self.log_report(f"   Arcs: {len(self.process_model.arcs)}")
            
            self.metrics['discovery_algorithm'] = algorithm
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def discover_dfg(self):
        """Discover DFG"""
        self.add_separator()
        self.log_report("\n🔗 DISCOVERING DIRECTLY-FOLLOWS GRAPH\n")
        
        try:
            self.dfg = dfg_discovery.apply(self.event_log, variant=dfg_discovery.Variants.FREQUENCY)
            start_activities = pm4py.get_start_activities(self.event_log)
            end_activities = pm4py.get_end_activities(self.event_log)
            
            self.log_report(f"✅ DFG discovered!")
            self.log_report(f"   Total edges: {len(self.dfg)}")
            self.log_report(f"   Start activities: {len(start_activities)}")
            self.log_report(f"   End activities: {len(end_activities)}")
            
            self.metrics['dfg_edges'] = len(self.dfg)
            self.metrics['start_activities'] = start_activities
            self.metrics['end_activities'] = end_activities
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    # ========== ENHANCED VISUALIZATIONS ==========
    
    def visualize_petri_net_enhanced(self):
        """Enhanced Petri Net visualization dengan multiple formats"""
        self.add_separator()
        self.log_report("\n🎨 VISUALIZING PETRI NET (ENHANCED)\n")
        
        try:
            # 1. High-quality PNG dengan frequency
            self.log_report("1. Creating high-quality Petri Net (PNG)...")
            output_png = os.path.join(self.dir_visualizations, 'petri_net_frequency.png')
            
            parameters = {
                pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png",
                "bgcolor": "white",
                "rankdir": "LR",  # Left to Right layout
            }
            
            gviz = pn_visualizer.apply(
                self.process_model, 
                self.initial_marking, 
                self.final_marking,
                parameters=parameters,
                variant=pn_visualizer.Variants.FREQUENCY,
                log=self.event_log
            )
            pn_visualizer.save(gviz, output_png)
            self.log_report(f"   ✅ PNG saved: '{output_png}'")
            
            # 2. SVG untuk quality maksimal
            self.log_report("\n2. Creating vector Petri Net (SVG)...")
            output_svg = os.path.join(self.dir_visualizations, 'petri_net_frequency.svg')
            
            parameters_svg = {
                pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "svg",
                "bgcolor": "white",
                "rankdir": "LR",
            }
            
            gviz_svg = pn_visualizer.apply(
                self.process_model, 
                self.initial_marking, 
                self.final_marking,
                parameters=parameters_svg,
                variant=pn_visualizer.Variants.FREQUENCY,
                log=self.event_log
            )
            pn_visualizer.save(gviz_svg, output_svg)
            self.log_report(f"   ✅ SVG saved: '{output_svg}' (zoomable, high quality)")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def visualize_dfg_enhanced(self):
        """Enhanced DFG dengan filtering dan multiple versions"""
        self.add_separator()
        self.log_report("\n🎨 VISUALIZING DFG (ENHANCED - MULTIPLE VERSIONS)\n")
        
        try:
            # Get performance DFG
            dfg_performance = dfg_discovery.apply(
                self.event_log, 
                variant=dfg_discovery.Variants.PERFORMANCE
            )
            
            # 1. Full DFG - Frequency
            self.log_report("1. Creating FULL DFG (Frequency)...")
            self._save_dfg_variant(
                self.dfg, 
                "dfg_full_frequency.png",
                variant=dfg_visualization.Variants.FREQUENCY,
                title="Full DFG - Frequency"
            )
            
            # 2. Filtered DFG - 80% paths (simplified)
            self.log_report("\n2. Creating SIMPLIFIED DFG (80% most frequent paths)...")
            dfg_filtered_80, sa_80, ea_80 = dfg_filtering.filter_dfg_on_paths_percentage(
                self.dfg,
                self.metrics['start_activities'],
                self.metrics['end_activities'],
                0.8  # Keep 80% most frequent paths
            )
            self._save_dfg_variant(
                dfg_filtered_80,
                "dfg_simplified_80pct.png",
                variant=dfg_visualization.Variants.FREQUENCY,
                title="Simplified DFG - 80% Paths",
                start_activities=sa_80,
                end_activities=ea_80
            )
            
            # 3. Filtered DFG - 95% paths
            self.log_report("\n3. Creating DETAILED DFG (95% paths)...")
            dfg_filtered_95, sa_95, ea_95 = dfg_filtering.filter_dfg_on_paths_percentage(
                self.dfg,
                self.metrics['start_activities'],
                self.metrics['end_activities'],
                0.95
            )
            self._save_dfg_variant(
                dfg_filtered_95,
                "dfg_detailed_95pct.png",
                variant=dfg_visualization.Variants.FREQUENCY,
                title="Detailed DFG - 95% Paths",
                start_activities=sa_95,
                end_activities=ea_95
            )
            
            # 4. Performance DFG (full)
            self.log_report("\n4. Creating PERFORMANCE DFG...")
            self._save_dfg_variant(
                dfg_performance,
                "dfg_performance.png",
                variant=dfg_visualization.Variants.PERFORMANCE,
                title="Performance DFG - Avg Duration"
            )
            
            # 5. Performance DFG (simplified)
            self.log_report("\n5. Creating SIMPLIFIED PERFORMANCE DFG...")
            dfg_perf_filtered, sa_pf, ea_pf = dfg_filtering.filter_dfg_on_paths_percentage(
                dfg_performance,
                self.metrics['start_activities'],
                self.metrics['end_activities'],
                0.8
            )
            self._save_dfg_variant(
                dfg_perf_filtered,
                "dfg_performance_simplified.png",
                variant=dfg_visualization.Variants.PERFORMANCE,
                title="Simplified Performance DFG",
                start_activities=sa_pf,
                end_activities=ea_pf
            )
            
            self.log_report("\n✅ All DFG visualizations created successfully!")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_dfg_variant(self, dfg, filename, variant, title, 
                          start_activities=None, end_activities=None):
        """Helper untuk save DFG variant"""
        output_path = os.path.join(self.dir_visualizations, filename)
        
        sa = start_activities if start_activities else self.metrics['start_activities']
        ea = end_activities if end_activities else self.metrics['end_activities']
        
        parameters = {
            "format": "png",
            "bgcolor": "white",
            "rankdir": "LR",
        }
        
        gviz = dfg_visualization.apply(
            dfg,
            log=self.event_log,
            variant=variant,
            parameters=parameters,
            activities_count=pm4py.get_event_attribute_values(self.event_log, "concept:name"),
            start_activities=sa,
            end_activities=ea
        )
        
        dfg_visualization.save(gviz, output_path)
        self.log_report(f"   ✅ {filename}")
    
    def visualize_bpmn_enhanced(self):
        """Enhanced BPMN visualization"""
        self.add_separator()
        self.log_report("\n🎨 VISUALIZING BPMN (ENHANCED)\n")
        
        try:
            self.bpmn_model = bpmn_converter.apply(
                self.process_model, 
                self.initial_marking, 
                self.final_marking
            )
            
            # PNG version
            output_png = os.path.join(self.dir_visualizations, 'bpmn_diagram.png')
            parameters_png = {
                "format": "png",
                "bgcolor": "white"
            }
            gviz_png = bpmn_visualizer.apply(self.bpmn_model, parameters=parameters_png)
            bpmn_visualizer.save(gviz_png, output_png)
            self.log_report(f"✅ BPMN PNG: '{output_png}'")
            
            # SVG version
            output_svg = os.path.join(self.dir_visualizations, 'bpmn_diagram.svg')
            parameters_svg = {
                "format": "svg",
                "bgcolor": "white"
            }
            gviz_svg = bpmn_visualizer.apply(self.bpmn_model, parameters=parameters_svg)
            bpmn_visualizer.save(gviz_svg, output_svg)
            self.log_report(f"✅ BPMN SVG: '{output_svg}'")
            
            return True
        except Exception as e:
            self.log_report(f"⚠️  BPMN error: {str(e)}")
            return False
    
    def visualize_process_tree_enhanced(self):
        """Enhanced Process Tree visualization"""
        self.add_separator()
        self.log_report("\n🌳 VISUALIZING PROCESS TREE (ENHANCED)\n")
        
        try:
            if self.process_tree is None:
                self.log_report("⚠️  Process tree not available")
                return False
            
            # PNG
            output_png = os.path.join(self.dir_visualizations, 'process_tree.png')
            parameters_png = {"format": "png", "bgcolor": "white"}
            gviz_png = pt_visualizer.apply(self.process_tree, parameters=parameters_png)
            pt_visualizer.save(gviz_png, output_png)
            self.log_report(f"✅ Process Tree PNG: '{output_png}'")
            
            # SVG
            output_svg = os.path.join(self.dir_visualizations, 'process_tree.svg')
            parameters_svg = {"format": "svg", "bgcolor": "white"}
            gviz_svg = pt_visualizer.apply(self.process_tree, parameters=parameters_svg)
            pt_visualizer.save(gviz_svg, output_svg)
            self.log_report(f"✅ Process Tree SVG: '{output_svg}'")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    # ========== ADDITIONAL VISUALIZATIONS ==========
    
    def create_variant_chart(self, top_n=20):
        """Create variant distribution chart"""
        self.add_separator()
        self.log_report(f"\n📊 CREATING VARIANT DISTRIBUTION CHART (Top {top_n})\n")
        
        try:
            variants = variants_get.get_variants(self.event_log)
            sorted_variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)
            
            # Prepare data
            variant_labels = []
            variant_counts = []
            
            for idx, (variant, traces) in enumerate(sorted_variants[:top_n], 1):
                variant_str = ' → '.join(list(variant)[:3])  # First 3 activities
                if len(variant) > 3:
                    variant_str += "..."
                variant_labels.append(f"V{idx}")
                variant_counts.append(len(traces))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            bars = ax.barh(variant_labels, variant_counts, color='steelblue', edgecolor='black')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', 
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Number of Cases', fontsize=12, fontweight='bold')
            ax.set_ylabel('Variant', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Process Variants Distribution', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'variant_distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"✅ Variant chart saved: '{output_path}'")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def create_case_duration_chart(self):
        """Create case duration distribution chart"""
        self.add_separator()
        self.log_report("\n📊 CREATING CASE DURATION DISTRIBUTION\n")
        
        try:
            # Calculate case durations
            case_durations = []
            for trace in self.event_log:
                if len(trace) > 0:
                    duration = (trace[-1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds() / 3600  # hours
                    case_durations.append(duration)
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Histogram
            ax1.hist(case_durations, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Case Duration (hours)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.set_title('Case Duration Distribution (Histogram)', fontsize=13, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            ax1.axvline(np.median(case_durations), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(case_durations):.1f}h')
            ax1.legend()
            
            # Boxplot
            ax2.boxplot(case_durations, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', edgecolor='black'),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))
            ax2.set_ylabel('Case Duration (hours)', fontsize=12, fontweight='bold')
            ax2.set_title('Case Duration Distribution (Boxplot)', fontsize=13, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'case_duration_distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"✅ Case duration chart saved: '{output_path}'")
            
            # Save statistics
            self.metrics['case_duration_mean'] = float(np.mean(case_durations))
            self.metrics['case_duration_median'] = float(np.median(case_durations))
            self.metrics['case_duration_std'] = float(np.std(case_durations))
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def create_activity_frequency_chart(self, top_n=15):
        """Create activity frequency chart"""
        self.add_separator()
        self.log_report(f"\n📊 CREATING ACTIVITY FREQUENCY CHART (Top {top_n})\n")
        
        try:
            activities = pm4py.get_event_attribute_values(self.event_log, "concept:name")
            sorted_activities = sorted(activities.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            activity_names = [act[0] for act in sorted_activities]
            activity_counts = [act[1] for act in sorted_activities]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            bars = ax.barh(activity_names, activity_counts, color='coral', edgecolor='black')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', 
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_ylabel('Activity', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Most Frequent Activities', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'activity_frequency.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"✅ Activity frequency chart saved: '{output_path}'")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    # ========== METRICS & ANALYSIS (copy from previous code) ==========
    
    def calculate_model_quality(self):
        """Calculate quality metrics"""
        self.add_separator()
        self.log_report("\n📏 CALCULATING MODEL QUALITY METRICS\n")
        
        try:
            # Fitness
            self.log_report("1. Calculating Fitness...")
            fitness = replay_fitness.apply(
                self.event_log, 
                self.process_model, 
                self.initial_marking, 
                self.final_marking,
                variant=replay_fitness.Variants.TOKEN_BASED
            )
            fitness_value = fitness['average_trace_fitness']
            self.log_report(f"   ✅ Fitness: {fitness_value:.4f}")
            
            # Precision
            self.log_report("\n2. Calculating Precision...")
            precision = precision_evaluator.apply(
                self.event_log, 
                self.process_model, 
                self.initial_marking, 
                self.final_marking,
                variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
            )
            self.log_report(f"   ✅ Precision: {precision:.4f}")
            
            # Generalization
            self.log_report("\n3. Calculating Generalization...")
            generalization = generalization_evaluator.apply(
                self.event_log, 
                self.process_model, 
                self.initial_marking, 
                self.final_marking
            )
            self.log_report(f"   ✅ Generalization: {generalization:.4f}")
            
            # Simplicity
            self.log_report("\n4. Calculating Simplicity...")
            num_places = len(self.process_model.places)
            num_transitions = len(self.process_model.transitions)
            num_arcs = len(self.process_model.arcs)
            complexity = num_places + num_transitions + num_arcs
            simplicity = 1 / (1 + np.log(complexity)) if complexity > 0 else 1
            
            self.log_report(f"   ✅ Simplicity: {simplicity:.4f}")
            
            # Save metrics
            self.metrics['fitness'] = float(fitness_value)
            self.metrics['precision'] = float(precision)
            self.metrics['generalization'] = float(generalization)
            self.metrics['simplicity'] = float(simplicity)
            
            # Overall quality
            overall_quality = (fitness_value * 0.4 + precision * 0.3 + 
                             generalization * 0.2 + simplicity * 0.1)
            self.metrics['overall_quality'] = float(overall_quality)
            
            self.log_report(f"\n📊 OVERALL QUALITY SCORE: {overall_quality:.4f}")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def discover_variants(self, top_n=10):
        """Analyze process variants"""
        self.add_separator()
        self.log_report(f"\n🔀 VARIANT ANALYSIS (Top {top_n})\n")
        
        try:
            variants = variants_get.get_variants(self.event_log)
            sorted_variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)
            
            total_variants = len(sorted_variants)
            total_cases = len(self.event_log)
            
            self.log_report(f"Total unique variants: {total_variants:,}")
            self.log_report(f"Total cases: {total_cases:,}")
            
            cumulative_coverage = 0
            variants_for_80_pct = 0
            
            self.log_report(f"\nTop {top_n} Most Frequent Variants:\n")
            
            for idx, (variant, traces) in enumerate(sorted_variants[:top_n], 1):
                freq = len(traces)
                pct = (freq / total_cases) * 100
                cumulative_coverage += pct
                
                if cumulative_coverage <= 80:
                    variants_for_80_pct = idx
                
                variant_str = ' → '.join(variant)
                if len(variant_str) > 100:
                    variant_str = variant_str[:100] + "..."
                
                self.log_report(f"{idx}. [{freq} cases, {pct:.2f}%, cumulative: {cumulative_coverage:.2f}%]")
                self.log_report(f"   {variant_str}\n")
            
            # Save metrics
            self.metrics['total_variants'] = total_variants
            self.metrics['variant_ratio'] = total_variants / total_cases
            self.metrics['variants_for_80pct'] = variants_for_80_pct
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def analyze_bottlenecks(self):
        """Identify bottlenecks"""
        self.add_separator()
        self.log_report("\n⏱️  BOTTLENECK ANALYSIS\n")
        
        try:
            activities = pm4py.get_event_attribute_values(self.event_log, "concept:name")
            activity_stats = []
            
            for activity in activities:
                filtered_log = attributes_filter.apply_events(
                    self.event_log,
                    [activity],
                    parameters={
                        attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name",
                        attributes_filter.Parameters.POSITIVE: True
                    }
                )
                
                if len(filtered_log) > 0:
                    durations = []
                    for trace in filtered_log:
                        if len(trace) >= 2:
                            duration = (trace[-1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds()
                            durations.append(duration)
                    
                    if durations:
                        activity_stats.append({
                            'activity': activity,
                            'frequency': len(filtered_log),
                            'mean_duration': np.mean(durations),
                            'median_duration': np.median(durations),
                            'max_duration': np.max(durations)
                        })
            
            activity_stats_sorted = sorted(activity_stats, key=lambda x: x['mean_duration'], reverse=True)
            
            self.log_report("✅ Top 10 Activities with longest duration:\n")
            
            for idx, stat in enumerate(activity_stats_sorted[:10], 1):
                self.log_report(f"{idx}. {stat['activity']}")
                self.log_report(f"   Frequency: {stat['frequency']}")
                self.log_report(f"   Mean Duration: {stat['mean_duration']/3600:.2f} hours")
                self.log_report(f"   Median Duration: {stat['median_duration']/3600:.2f} hours\n")
            
            # Detect bottlenecks
            all_durations = [s['mean_duration'] for s in activity_stats]
            overall_median = np.median(all_durations)
            bottlenecks = [s for s in activity_stats if s['mean_duration'] > 2 * overall_median]
            
            self.log_report(f"🚨 BOTTLENECKS DETECTED: {len(bottlenecks)} activities")
            for b in bottlenecks:
                self.log_report(f"   - {b['activity']}: {b['mean_duration']/3600:.2f} hours")
            
            self.metrics['bottlenecks'] = [b['activity'] for b in bottlenecks]
            self.metrics['bottleneck_count'] = len(bottlenecks)
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    # ========== EXPORT & REPORTING ==========
    
    def export_results(self):
        """Export all results"""
        self.add_separator()
        self.log_report("\n💾 EXPORTING RESULTS\n")
        
        try:
            # 1. Export Petri Net (PNML)
            self.log_report("1. Exporting Petri Net (PNML)...")
            pnml_path = os.path.join(self.dir_models, 'petri_net.pnml')
            pm4py.write_pnml(self.process_model, self.initial_marking, self.final_marking, pnml_path)
            self.log_report(f"   ✅ PNML: '{pnml_path}'")
            
            # 2. Export BPMN
            if self.bpmn_model:
                self.log_report("\n2. Exporting BPMN (XML)...")
                bpmn_path = os.path.join(self.dir_models, 'bpmn_model.bpmn')
                pm4py.write_bpmn(self.bpmn_model, bpmn_path)
                self.log_report(f"   ✅ BPMN: '{bpmn_path}'")
            
            # 3. Export Metrics (JSON)
            self.log_report("\n3. Exporting Metrics (JSON)...")
            metrics_path = os.path.join(self.dir_data, 'discovery_metrics.json')
            
            metrics_serializable = {}
            for key, value in self.metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_serializable[key] = float(value)
                elif isinstance(value, (list, dict)):
                    metrics_serializable[key] = value
                else:
                    metrics_serializable[key] = str(value)
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_serializable, f, indent=4, ensure_ascii=False)
            
            self.log_report(f"   ✅ Metrics: '{metrics_path}'")
            
            # 4. Export Event Log Statistics
            self.log_report("\n4. Exporting Event Log Statistics...")
            stats_path = os.path.join(self.dir_reports, 'event_log_statistics.txt')
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("EVENT LOG STATISTICS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total Cases: {self.metrics.get('total_cases', 0):,}\n")
                f.write(f"Total Events: {self.metrics.get('total_events', 0):,}\n")
                f.write(f"Total Variants: {self.metrics.get('total_variants', 0):,}\n")
                f.write(f"Discovery Algorithm: {self.metrics.get('discovery_algorithm', 'N/A')}\n")
                f.write(f"\nQUALITY METRICS:\n")
                f.write(f"  Fitness: {self.metrics.get('fitness', 0):.4f}\n")
                f.write(f"  Precision: {self.metrics.get('precision', 0):.4f}\n")
                f.write(f"  Generalization: {self.metrics.get('generalization', 0):.4f}\n")
                f.write(f"  Simplicity: {self.metrics.get('simplicity', 0):.4f}\n")
                f.write(f"  Overall Quality: {self.metrics.get('overall_quality', 0):.4f}\n")
            
            self.log_report(f"   ✅ Statistics: '{stats_path}'")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def generate_html_dashboard(self):
        """Generate interactive HTML dashboard"""
        self.add_separator()
        self.log_report("\n🌐 GENERATING INTERACTIVE HTML DASHBOARD\n")
        
        try:
            dashboard_path = os.path.join(self.output_dir, 'dashboard.html')
            
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Process Discovery Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .metrics-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        .metric-card h3 {{
            color: #667eea;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .section {{
            padding: 40px;
        }}
        .section h2 {{
            color: #333;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .visualization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}
        .viz-card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s;
        }}
        .viz-card:hover {{
            transform: scale(1.02);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        .viz-card h3 {{
            background: #667eea;
            color: white;
            padding: 15px;
            font-size: 1.2em;
        }}
        .viz-card img {{
            width: 100%;
            display: block;
        }}
        .viz-card p {{
            padding: 15px;
            color: #666;
        }}
        .quality-bars {{
            margin-top: 20px;
        }}
        .quality-item {{
            margin-bottom: 15px;
        }}
        .quality-item label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 1s ease;
        }}
        .nav-tabs {{
            display: flex;
            background: #f8f9fa;
            padding: 0 40px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .nav-tab {{
            padding: 15px 30px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1.1em;
            color: #666;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }}
        .nav-tab:hover {{
            color: #667eea;
        }}
        .nav-tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Process Discovery Dashboard</h1>
            <p>PM4PY Framework - Inductive Miner Analysis</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <!-- Metrics Overview -->
        <div class="metrics-container">
            <div class="metric-card">
                <h3>Total Cases</h3>
                <div class="metric-value">{self.metrics.get('total_cases', 0):,}</div>
                <div class="metric-label">Unique process instances</div>
            </div>
            <div class="metric-card">
                <h3>Total Events</h3>
                <div class="metric-value">{self.metrics.get('total_events', 0):,}</div>
                <div class="metric-label">Activity executions</div>
            </div>
            <div class="metric-card">
                <h3>Unique Variants</h3>
                <div class="metric-value">{self.metrics.get('total_variants', 0):,}</div>
                <div class="metric-label">Different process paths</div>
            </div>
            <div class="metric-card">
                <h3>Overall Quality</h3>
                <div class="metric-value">{self.metrics.get('overall_quality', 0):.3f}</div>
                <div class="metric-label">Model quality score</div>
            </div>
            <div class="metric-card">
                <h3>Bottlenecks</h3>
                <div class="metric-value">{self.metrics.get('bottleneck_count', 0)}</div>
                <div class="metric-label">Performance issues</div>
            </div>
            <div class="metric-card">
                <h3>DFG Edges</h3>
                <div class="metric-value">{self.metrics.get('dfg_edges', 0)}</div>
                <div class="metric-label">Activity transitions</div>
            </div>
        </div>
        
        <!-- Navigation Tabs -->
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('models')">Process Models</button>
            <button class="nav-tab" onclick="showTab('dfg')">DFG Analysis</button>
            <button class="nav-tab" onclick="showTab('statistics')">Statistics</button>
            <button class="nav-tab" onclick="showTab('quality')">Quality Metrics</button>
        </div>
        
        <!-- Tab Content: Process Models -->
        <div id="models" class="tab-content active">
            <div class="section">
                <h2>📊 Process Models</h2>
                <div class="visualization-grid">
                    <div class="viz-card">
                        <h3>Petri Net (with Frequency)</h3>
                        <img src="visualizations/petri_net_frequency.png" alt="Petri Net">
                        <p>Petri Net model showing activity frequencies and transitions</p>
                    </div>
                    <div class="viz-card">
                        <h3>BPMN Diagram</h3>
                        <img src="visualizations/bpmn_diagram.png" alt="BPMN">
                        <p>Business Process Model and Notation standard representation</p>
                    </div>
                    <div class="viz-card">
                        <h3>Process Tree</h3>
                        <img src="visualizations/process_tree.png" alt="Process Tree">
                        <p>Hierarchical tree structure of the discovered process</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Tab Content: DFG Analysis -->
        <div id="dfg" class="tab-content">
            <div class="section">
                <h2>🔗 Directly-Follows Graph Analysis</h2>
                <div class="visualization-grid">
                    <div class="viz-card">
                        <h3>Simplified DFG - 80% Paths</h3>
                        <img src="visualizations/dfg_simplified_80pct.png" alt="DFG 80%">
                        <p>Main process flow (80% most frequent paths)</p>
                    </div>
                    <div class="viz-card">
                        <h3>Full DFG - Frequency</h3>
                        <img src="visualizations/dfg_full_frequency.png" alt="DFG Full">
                        <p>Complete process flow with all paths and frequencies</p>
                    </div>
                    <div class="viz-card">
                        <h3>Performance DFG (Simplified)</h3>
                        <img src="visualizations/dfg_performance_simplified.png" alt="DFG Performance">
                        <p>Main paths with average duration times</p>
                    </div>
                    <div class="viz-card">
                        <h3>Performance DFG (Full)</h3>
                        <img src="visualizations/dfg_performance.png" alt="DFG Performance Full">
                        <p>Complete flow with performance metrics</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Tab Content: Statistics -->
        <div id="statistics" class="tab-content">
            <div class="section">
                <h2>📈 Statistical Analysis</h2>
                <div class="visualization-grid">
                    <div class="viz-card">
                        <h3>Variant Distribution</h3>
                        <img src="visualizations/variant_distribution.png" alt="Variants">
                        <p>Top 20 most frequent process variants</p>
                    </div>
                    <div class="viz-card">
                        <h3>Case Duration Distribution</h3>
                        <img src="visualizations/case_duration_distribution.png" alt="Duration">
                        <p>Distribution of case completion times</p>
                    </div>
                    <div class="viz-card">
                        <h3>Activity Frequency</h3>
                        <img src="visualizations/activity_frequency.png" alt="Activity Freq">
                        <p>Most frequently executed activities</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Tab Content: Quality Metrics -->
        <div id="quality" class="tab-content">
            <div class="section">
                <h2>⭐ Quality Metrics</h2>
                <div class="quality-bars">
                    <div class="quality-item">
                        <label>Fitness: {self.metrics.get('fitness', 0):.4f}</label>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {self.metrics.get('fitness', 0)*100}%">
                                {self.metrics.get('fitness', 0):.4f}
                            </div>
                        </div>
                        <p style="margin-top: 5px; color: #666;">How well the model reproduces the event log</p>
                    </div>
                    <div class="quality-item">
                        <label>Precision: {self.metrics.get('precision', 0):.4f}</label>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {self.metrics.get('precision', 0)*100}%">
                                {self.metrics.get('precision', 0):.4f}
                            </div>
                        </div>
                        <p style="margin-top: 5px; color: #666;">How precise the model is (not overgeneralized)</p>
                    </div>
                    <div class="quality-item">
                        <label>Generalization: {self.metrics.get('generalization', 0):.4f}</label>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {self.metrics.get('generalization', 0)*100}%">
                                {self.metrics.get('generalization', 0):.4f}
                            </div>
                        </div>
                        <p style="margin-top: 5px; color: #666;">Ability to handle unseen cases</p>
                    </div>
                    <div class="quality-item">
                        <label>Simplicity: {self.metrics.get('simplicity', 0):.4f}</label>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {self.metrics.get('simplicity', 0)*100}%">
                                {self.metrics.get('simplicity', 0):.4f}
                            </div>
                        </div>
                        <p style="margin-top: 5px; color: #666;">Structural simplicity of the model</p>
                    </div>
                    <div class="quality-item">
                        <label>Overall Quality: {self.metrics.get('overall_quality', 0):.4f}</label>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {self.metrics.get('overall_quality', 0)*100}%; background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);">
                                {self.metrics.get('overall_quality', 0):.4f}
                            </div>
                        </div>
                        <p style="margin-top: 5px; color: #666;">Weighted average of all metrics</p>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Process Discovery Dashboard | PM4PY Framework | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active from all tabs
            const tabs = document.querySelectorAll('.nav-tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Mark tab as active
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""
            
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.log_report(f"✅ HTML Dashboard created: '{dashboard_path}'")
            self.log_report(f"   Open this file in your browser for interactive view!")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def generate_discovery_report(self):
        """Generate comprehensive text report"""
        self.add_separator()
        self.log_report("\n📋 GENERATING DISCOVERY REPORT\n")
        
        try:
            report_path = os.path.join(self.dir_reports, 'discovery_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PROCESS DISCOVERY REPORT\n")
                f.write("PM4PY Framework - Inductive Miner\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input File: {self.input_file}\n\n")
                
                # Write all report lines
                for line in self.report:
                    f.write(line + "\n")
            
            self.log_report(f"✅ Discovery report saved: '{report_path}'")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    # ========== MAIN ORCHESTRATION ==========
    
    def run(self):
        """Main orchestration method"""
        print("\n" + "=" * 80)
        print("🚀 ENHANCED PROCESS DISCOVERY PIPELINE")
        print("=" * 80 + "\n")
        
        try:
            # Step 1: Load Data
            if not self.load_preprocessed_data():
                return False
            
            # Step 2: Convert to Event Log
            if not self.convert_to_event_log():
                return False
            
            # Step 3: Discover DFG
            if not self.discover_dfg():
                return False
            
            # Step 4: Discover Process Model
            if not self.discover_process_model(algorithm='inductive'):
                return False
            
            # Step 5: Calculate Model Quality
            if not self.calculate_model_quality():
                return False
            
            # Step 6: Analyze Variants
            if not self.discover_variants(top_n=10):
                return False
            
            # Step 7: Analyze Bottlenecks
            if not self.analyze_bottlenecks():
                return False
            
            # Step 8: Create All Visualizations
            self.log_report("\n" + "=" * 80)
            self.log_report("🎨 CREATING ENHANCED VISUALIZATIONS")
            self.log_report("=" * 80 + "\n")
            
            self.visualize_petri_net_enhanced()
            self.visualize_process_tree_enhanced()
            self.visualize_bpmn_enhanced()
            self.visualize_dfg_enhanced()
            
            # Step 9: Create Statistical Charts
            self.log_report("\n" + "=" * 80)
            self.log_report("📊 CREATING STATISTICAL CHARTS")
            self.log_report("=" * 80 + "\n")
            
            self.create_variant_chart(top_n=20)
            self.create_case_duration_chart()
            self.create_activity_frequency_chart(top_n=15)
            
            # Step 10: Export Results
            if not self.export_results():
                return False
            
            # Step 11: Generate Reports
            if not self.generate_discovery_report():
                return False
            
            # Step 12: Generate HTML Dashboard
            if not self.generate_html_dashboard():
                return False
            
            # Success Summary
            print("\n" + "=" * 80)
            print("✅ ENHANCED PROCESS DISCOVERY COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\n📂 All results saved in: '{self.output_dir}/'")
            print("\n📊 Key Metrics:")
            print(f"   • Total Cases: {self.metrics.get('total_cases', 0):,}")
            print(f"   • Total Events: {self.metrics.get('total_events', 0):,}")
            print(f"   • Total Variants: {self.metrics.get('total_variants', 0):,}")
            print(f"   • Fitness: {self.metrics.get('fitness', 0):.4f}")
            print(f"   • Precision: {self.metrics.get('precision', 0):.4f}")
            print(f"   • Overall Quality: {self.metrics.get('overall_quality', 0):.4f}")
            print(f"   • Bottlenecks Detected: {self.metrics.get('bottleneck_count', 0)}")
            
            print("\n🎯 Quick Access:")
            print(f"   • Open dashboard.html in browser for interactive view")
            print(f"   • Check visualizations/ folder for all process models")
            print(f"   • Read reports/discovery_report.txt for detailed analysis")
            
            return True
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR in discovery pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# ========== MAIN EXECUTION BLOCK ==========
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("   ╔═══════════════════════════════════════════════════════════╗")
    print("   ║     PROCESS MINING - ENHANCED PROCESS DISCOVERY          ║")
    print("   ║     PM4PY Framework with Inductive Miner                 ║")
    print("   ║     Enhanced Visualizations & Interactive Dashboard      ║")
    print("   ╚═══════════════════════════════════════════════════════════╝")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    try:
        # Create ProcessDiscoveryEnhanced instance
        print("🔧 Initializing Process Discovery...")
        discoverer = ProcessDiscoveryEnhanced(
            input_file='D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/finale_preprocessed.csv',
            output_dir='D:/VSCODE/myenv2/share/Mining_Final_Project/Process_Discovery/process_discovery_output',
            case_id_col='Case ID',
            activity_col='Activity',
            timestamp_col='Complete Timestamp'
        )
        
        print("✅ Initialization complete!\n")
        
        # Run enhanced discovery pipeline
        success = discoverer.run()
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 PROCESS DISCOVERY BERHASIL!")
            print("=" * 80)
            
            print(f"\n⏱️  Execution Time: {minutes} minutes {seconds} seconds")
            
            print("\n📁 Output Structure:")
            print("   process_discovery_output/")
            print("   ├── 📊 dashboard.html (⭐ START HERE!)")
            print("   ├── visualizations/")
            print("   │   ├── petri_net_frequency.png")
            print("   │   ├── petri_net_frequency.svg (high quality)")
            print("   │   ├── process_tree.png & .svg")
            print("   │   ├── bpmn_diagram.png & .svg")
            print("   │   ├── dfg_full_frequency.png")
            print("   │   ├── dfg_simplified_80pct.png (⭐ recommended)")
            print("   │   ├── dfg_detailed_95pct.png")
            print("   │   ├── dfg_performance.png")
            print("   │   ├── dfg_performance_simplified.png")
            print("   │   ├── variant_distribution.png")
            print("   │   ├── case_duration_distribution.png")
            print("   │   └── activity_frequency.png")
            print("   ├── models/")
            print("   │   ├── petri_net.pnml")
            print("   │   └── bpmn_model.bpmn")
            print("   ├── reports/")
            print("   │   ├── discovery_report.txt")
            print("   │   └── event_log_statistics.txt")
            print("   └── data/")
            print("       └── discovery_metrics.json")
            
            print("\n🚀 Next Steps:")
            print("   1. Open 'dashboard.html' in your browser for interactive exploration")
            print("   2. Review 'dfg_simplified_80pct.png' for main process flow")
            print("   3. Check 'reports/discovery_report.txt' for detailed analysis")
            print("   4. Examine bottlenecks in 'dfg_performance_simplified.png'")
            print("   5. Use SVG files for presentations (infinite zoom quality)")
            
            print("\n💡 Visualization Tips:")
            print("   • PNG files: Good for reports and documents")
            print("   • SVG files: Best for presentations (zoomable, vector graphics)")
            print("   • Simplified DFG (80%): Shows main process flow clearly")
            print("   • Full DFG: Shows all details but can be complex")
            print("   • Performance DFG: Highlights bottlenecks (red = slow)")
            
            print("\n📈 Quality Assessment:")
            fitness = discoverer.metrics.get('fitness', 0)
            precision = discoverer.metrics.get('precision', 0)
            overall = discoverer.metrics.get('overall_quality', 0)
            
            if overall >= 0.8:
                print("   ✅ Model Quality: EXCELLENT")
                print("      Your process model is highly reliable!")
            elif overall >= 0.7:
                print("   ✅ Model Quality: GOOD")
                print("      Your process model is reliable for analysis")
            elif overall >= 0.6:
                print("   ⚠️  Model Quality: FAIR")
                print("      Consider process standardization or noise filtering")
            else:
                print("   ⚠️  Model Quality: NEEDS IMPROVEMENT")
                print("      High process variability detected - review process design")
            
            if fitness < 0.7:
                print("   ⚠️  Low Fitness: Model doesn't capture all behaviors")
            if precision < 0.7:
                print("   ⚠️  Low Precision: Model allows too many extra behaviors")
            
            bottleneck_count = discoverer.metrics.get('bottleneck_count', 0)
            if bottleneck_count > 0:
                print(f"   🚨 {bottleneck_count} Bottleneck(s) Detected!")
                print("      Review performance DFG for optimization opportunities")
            
            variant_ratio = discoverer.metrics.get('variant_ratio', 0)
            if variant_ratio > 0.7:
                print(f"   📊 High Process Variability: {variant_ratio:.1%}")
                print("      Consider process standardization")
            
        else:
            print("\n" + "=" * 80)
            print("❌ PROCESS DISCOVERY FAILED")
            print("=" * 80)
            print("\n⚠️  Check error messages above for details")
            print("\n🔍 Common Issues:")
            print("   • File not found: Check input_file path")
            print("   • Column not detected: Verify column names in CSV")
            print("   • PM4PY error: Ensure pm4py is properly installed")
            print("   • Memory error: Dataset too large, consider filtering")
    
    except FileNotFoundError as e:
        print("\n" + "=" * 80)
        print("❌ FILE NOT FOUND ERROR")
        print("=" * 80)
        print(f"\n{str(e)}")
        print("\n💡 Solution:")
        print("   1. Check if 'finale_preprocessed.csv' exists")
        print("   2. Verify the file path is correct")
        print("   3. Run preprocessing script first if needed")
    
    except ImportError as e:
        print("\n" + "=" * 80)
        print("❌ IMPORT ERROR")
        print("=" * 80)
        print(f"\n{str(e)}")
        print("\n💡 Solution:")
        print("   Install required packages:")
        print("   pip install pm4py pandas numpy matplotlib seaborn")
    
    except MemoryError:
        print("\n" + "=" * 80)
        print("❌ MEMORY ERROR")
        print("=" * 80)
        print("\n⚠️  Dataset too large for available memory")
        print("\n💡 Solutions:")
        print("   1. Filter dataset to smaller time period")
        print("   2. Sample a subset of cases")
        print("   3. Increase system RAM")
        print("   4. Use cloud computing with more resources")
    
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  PROCESS INTERRUPTED BY USER")
        print("=" * 80)
        print("\n🛑 Process discovery stopped manually")
        print("   Partial results may be saved in output directory")
    
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ UNEXPECTED ERROR")
        print("=" * 80)
        print(f"\n{str(e)}")
        print("\n🔍 Full error trace:")
        import traceback
        traceback.print_exc()
        print("\n💡 If error persists:")
        print("   1. Check input data format")
        print("   2. Verify all dependencies are installed")
        print("   3. Try with a smaller dataset first")
        print("   4. Check PM4PY documentation for compatibility")
    
    finally:
        print("\n" + "=" * 80)
        print("Program execution completed.")
        print("=" * 80 + "\n")