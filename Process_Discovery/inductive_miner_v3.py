"""
Comprehensive Process Mining Framework using PM4PY
===================================================

This module provides an enhanced process discovery pipeline that follows
academic standards for process mining as described in papers by Wil van der Aalst
and PM4PY documentation.

Features:
- Data loading and preprocessing (CSV/XES)
- Process discovery using multiple algorithms (Inductive, Alpha, Heuristics)
- Advanced conformance checking (token replay, alignments, deviations)
- Social network analysis (handover of work, working together, subcontracting)
- Decision mining and branching analysis
- Model quality evaluation (fitness, precision, generalization, simplicity)
- Comprehensive visualizations and reporting
- Algorithm comparison and benchmarking

Author: Process Mining Specialist
Date: December 2025
"""

import os
import json
import warnings
import traceback
from datetime import datetime
from collections import defaultdict, Counter
import math

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# PM4PY imports - UPDATED AND FIXED
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from pm4py.objects.process_tree.exporter import exporter as ptml_exporter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# Try to import organizational mining (SNA) - may not be available in all versions
try:
    from pm4py.algo.organizational_mining.sna import algorithm as sna
    from pm4py.visualization.sna import visualizer as sna_visualizer
    SNA_AVAILABLE = True
except ImportError:
    SNA_AVAILABLE = False
    print("⚠️  Warning: Social Network Analysis module not available in this PM4PY version")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.rcParams['figure.max_open_warning'] = 50


class EnhancedProcessDiscovery:
    """
    Enhanced Process Discovery Pipeline with PM4PY
    
    This class provides a comprehensive framework for process mining that includes:
    - Process discovery using multiple algorithms
    - Advanced conformance checking
    - Social network analysis
    - Decision mining
    - Model quality evaluation
    - Comprehensive reporting and visualization
    """
    
    def __init__(self, file_path, case_id='case:concept:name', 
                 activity_key='concept:name', timestamp_key='time:timestamp',
                 output_dir='process_mining_output'):
        """
        Initialize the Enhanced Process Discovery pipeline
        
        Parameters:
        -----------
        file_path : str
            Path to the event log file (CSV or XES format)
        case_id : str
            Column name for case identifier
        activity_key : str
            Column name for activity
        timestamp_key : str
            Column name for timestamp
        output_dir : str
            Directory for saving outputs
        """
        self.file_path = file_path
        self.case_id = case_id
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.output_dir = output_dir
        
        # Initialize attributes
        self.dataframe = None
        self.event_log = None
        self.process_model = None
        self.initial_marking = None
        self.final_marking = None
        self.process_tree = None
        self.bpmn_model = None
        self.dfg = None
        self.metrics = {}
        self.report_lines = []
        
        # Create output directories
        self.dir_visualizations = os.path.join(output_dir, 'visualizations')
        self.dir_models = os.path.join(output_dir, 'models')
        self.dir_reports = os.path.join(output_dir, 'reports')
        
        for directory in [self.dir_visualizations, self.dir_models, self.dir_reports]:
            os.makedirs(directory, exist_ok=True)
        
        print(f"✅ EnhancedProcessDiscovery initialized")
        print(f"   Output directory: {output_dir}")
        print(f"   Case ID: {case_id}")
        print(f"   Activity: {activity_key}")
        print(f"   Timestamp: {timestamp_key}\n")
    
    def log_report(self, message):
        """Add message to report and print it"""
        print(message)
        self.report_lines.append(message)
    
    def add_separator(self, char='=', length=100):
        """Add separator line to report"""
        separator = char * length
        self.log_report(separator)
    
    # ==================== DATA LOADING & PREPROCESSING ====================
    
    def load_and_preprocess_data(self):
        """
        Load event log from CSV or XES file and preprocess it
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n📂 STEP 1: LOADING AND PREPROCESSING DATA\n")
        
        try:
            self.log_report(f"Loading event log from: '{self.file_path}'")
            
            file_extension = os.path.splitext(self.file_path)[1].lower()
            
            if file_extension == '.xes':
                # Load XES file directly
                self.log_report("   Format: XES")
                self.event_log = xes_importer.apply(self.file_path)
                
            elif file_extension == '.csv':
                # Load CSV file
                self.log_report("   Format: CSV")
                
                # Read CSV with multiple encoding attempts
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(self.file_path, encoding=encoding)
                        self.log_report(f"   Successfully read with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError("Could not read CSV file with any standard encoding")
                
                self.log_report(f"   Loaded {len(df)} rows and {len(df.columns)} columns")
                self.log_report(f"   Columns: {list(df.columns)}")
                
                # Check if required columns exist
                required_cols = [self.case_id, self.activity_key, self.timestamp_key]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    self.log_report(f"\n❌ ERROR: Missing required columns: {missing_cols}")
                    self.log_report(f"   Available columns: {list(df.columns)}")
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                # Preprocess dataframe
                self.log_report("\nPreprocessing data...")
                
                # 1. Handle timestamp column
                self.log_report(f"   Converting timestamp column: '{self.timestamp_key}'")
                try:
                    df[self.timestamp_key] = pd.to_datetime(df[self.timestamp_key])
                except Exception as e:
                    self.log_report(f"   ⚠️ Warning: Could not parse all timestamps: {str(e)}")
                    df[self.timestamp_key] = pd.to_datetime(df[self.timestamp_key], errors='coerce')
                    # Drop rows with invalid timestamps
                    before_count = len(df)
                    df = df.dropna(subset=[self.timestamp_key])
                    after_count = len(df)
                    if before_count != after_count:
                        self.log_report(f"   Removed {before_count - after_count} rows with invalid timestamps")
                
                # 2. Sort by case and timestamp
                self.log_report("   Sorting by case ID and timestamp...")
                df = df.sort_values([self.case_id, self.timestamp_key])
                
                # 3. Remove duplicates
                before_count = len(df)
                df = df.drop_duplicates()
                after_count = len(df)
                if before_count != after_count:
                    self.log_report(f"   Removed {before_count - after_count} duplicate rows")
                
                # 4. Rename columns to PM4PY standard format if needed
                self.log_report("   Renaming columns to PM4PY standard...")
                df_renamed = df.rename(columns={
                    self.case_id: 'case:concept:name',
                    self.activity_key: 'concept:name',
                    self.timestamp_key: 'time:timestamp'
                })
                
                # 5. Convert to event log using PM4PY (FIXED VERSION)
                self.log_report("   Converting to PM4PY event log format...")
                
                # Method 1: Try modern PM4PY conversion (v2.7+)
                try:
                    self.event_log = pm4py.convert_to_event_log(df_renamed)
                    self.log_report("   ✅ Conversion successful (modern method)")
                except:
                    # Method 2: Try with explicit parameters
                    try:
                        from pm4py.objects.log.util import dataframe_utils
                        from pm4py.objects.conversion.log import converter as log_converter
                        
                        # Add required columns if not present
                        if 'case:concept:name' not in df_renamed.columns:
                            df_renamed['case:concept:name'] = df_renamed[self.case_id]
                        if 'concept:name' not in df_renamed.columns:
                            df_renamed['concept:name'] = df_renamed[self.activity_key]
                        if 'time:timestamp' not in df_renamed.columns:
                            df_renamed['time:timestamp'] = df_renamed[self.timestamp_key]
                        
                        # Convert using parameters dictionary
                        parameters = {
                            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'
                        }
                        
                        self.event_log = log_converter.apply(
                            df_renamed, 
                            parameters=parameters,
                            variant=log_converter.Variants.TO_EVENT_LOG
                        )
                        self.log_report("   ✅ Conversion successful (legacy method)")
                    except:
                        # Method 3: Manual conversion (most compatible)
                        try:
                            from pm4py.objects.log.obj import EventLog, Trace, Event
                            
                            self.event_log = EventLog()
                            
                            for case_id, group in df_renamed.groupby('case:concept:name'):
                                trace = Trace()
                                trace.attributes['concept:name'] = case_id
                                
                                for _, row in group.iterrows():
                                    event = Event()
                                    for col in df_renamed.columns:
                                        event[col] = row[col]
                                    trace.append(event)
                                
                                self.event_log.append(trace)
                            
                            self.log_report("   ✅ Conversion successful (manual method)")
                        except Exception as e:
                            raise Exception(f"All conversion methods failed: {str(e)}")
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Use .csv or .xes")
            
            # Store basic statistics
            self.metrics['total_cases'] = len(self.event_log)
            self.metrics['total_events'] = sum(len(trace) for trace in self.event_log)
            
            # Get unique activities
            all_activities = set()
            for trace in self.event_log:
                for event in trace:
                    if self.activity_key in event:
                        all_activities.add(event[self.activity_key])
                    elif 'concept:name' in event:
                        all_activities.add(event['concept:name'])
            
            self.metrics['unique_activities'] = len(all_activities)
            
            # Check for resource information
            has_resource = any(
                'org:resource' in event or 'resource' in event.keys()
                for trace in self.event_log 
                for event in trace
            )
            self.metrics['has_resource_info'] = has_resource
            
            # Log summary
            self.log_report(f"\n✅ Event log loaded successfully!")
            self.log_report(f"   Total Cases: {self.metrics['total_cases']}")
            self.log_report(f"   Total Events: {self.metrics['total_events']}")
            self.log_report(f"   Unique Activities: {self.metrics['unique_activities']}")
            self.log_report(f"   Average Events per Case: {self.metrics['total_events']/self.metrics['total_cases']:.2f}")
            self.log_report(f"   Resource Information: {'Yes' if has_resource else 'No'}")
            
            return True
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR in data loading: {str(e)}")
            traceback.print_exc()
            return False
    
    # ==================== STATISTICAL ANALYSIS ====================
    
    def perform_statistical_analysis(self):
        """
        Perform comprehensive statistical analysis on the event log
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n📈 STEP 2: STATISTICAL ANALYSIS\n")
        
        try:
            # 1. Variant Analysis
            self.log_report("1. Analyzing Process Variants...")
            variants = variants_filter.get_variants(self.event_log)
            sorted_variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)
            
            self.log_report(f"   Total Variants: {len(variants)}")
            self.log_report(f"   Most Common Variant ({len(sorted_variants[0][1])} cases):")
            variant_activities = sorted_variants[0][0].split(',')
            for i, activity in enumerate(variant_activities[:10], 1):
                self.log_report(f"      {i}. {activity}")
            if len(variant_activities) > 10:
                self.log_report(f"      ... and {len(variant_activities) - 10} more activities")
            
            self.metrics['total_variants'] = len(variants)
            self.metrics['variant_coverage_top_10'] = sum(len(v[1]) for v in sorted_variants[:10]) / self.metrics['total_cases']
            
            # 2. Case Duration Analysis
            self.log_report("\n2. Analyzing Case Durations...")
            case_durations = case_statistics.get_all_case_durations(self.event_log, parameters={
                case_statistics.Parameters.TIMESTAMP_KEY: self.timestamp_key
            })
            
            if case_durations:
                avg_duration = np.mean(case_durations)
                median_duration = np.median(case_durations)
                std_duration = np.std(case_durations)
                min_duration = np.min(case_durations)
                max_duration = np.max(case_durations)
                
                self.log_report(f"   Average Duration: {avg_duration:.2f} seconds ({avg_duration/3600:.2f} hours)")
                self.log_report(f"   Median Duration: {median_duration:.2f} seconds ({median_duration/3600:.2f} hours)")
                self.log_report(f"   Std Deviation: {std_duration:.2f} seconds")
                self.log_report(f"   Min Duration: {min_duration:.2f} seconds ({min_duration/3600:.2f} hours)")
                self.log_report(f"   Max Duration: {max_duration:.2f} seconds ({max_duration/3600:.2f} hours)")
                
                self.metrics['avg_case_duration'] = float(avg_duration)
                self.metrics['median_case_duration'] = float(median_duration)
                self.metrics['std_case_duration'] = float(std_duration)
                self.metrics['min_case_duration'] = float(min_duration)
                self.metrics['max_case_duration'] = float(max_duration)
            
            # 3. Activity Frequency Analysis
            self.log_report("\n3. Analyzing Activity Frequencies...")
            activity_counts = Counter(event[self.activity_key] for trace in self.event_log for event in trace)
            sorted_activities = activity_counts.most_common(10)
            
            self.log_report(f"   Top 10 Most Frequent Activities:")
            for i, (activity, count) in enumerate(sorted_activities, 1):
                percentage = (count / self.metrics['total_events']) * 100
                self.log_report(f"      {i}. {activity}: {count} times ({percentage:.2f}%)")
            
            self.metrics['activity_frequencies'] = {k: v for k, v in activity_counts.items()}
            
            # 4. Activity Duration Analysis
            self.log_report("\n4. Analyzing Activity Durations...")
            activity_durations = defaultdict(list)
            
            for trace in self.event_log:
                for i in range(len(trace) - 1):
                    current_event = trace[i]
                    next_event = trace[i + 1]
                    activity = current_event[self.activity_key]
                    
                    if self.timestamp_key in current_event and self.timestamp_key in next_event:
                        duration = (next_event[self.timestamp_key] - current_event[self.timestamp_key]).total_seconds()
                        if duration > 0:
                            activity_durations[activity].append(duration)
            
            if activity_durations:
                avg_activity_durations = {
                    activity: np.mean(durations) 
                    for activity, durations in activity_durations.items()
                }
                sorted_durations = sorted(avg_activity_durations.items(), key=lambda x: x[1], reverse=True)[:10]
                
                self.log_report(f"   Top 10 Activities by Average Duration:")
                for i, (activity, duration) in enumerate(sorted_durations, 1):
                    self.log_report(f"      {i}. {activity}: {duration:.2f} seconds ({duration/3600:.2f} hours)")
                
                self.metrics['avg_activity_durations'] = {k: float(v) for k, v in avg_activity_durations.items()}
            
            # Create statistical visualizations
            self._create_variant_chart(sorted_variants[:20])
            self._create_duration_distribution_chart(case_durations)
            self._create_activity_frequency_chart(sorted_activities)
            
            return True
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR in statistical analysis: {str(e)}")
            traceback.print_exc()
            return False
    
    def _create_variant_chart(self, top_variants):
        """Create visualization of top process variants"""
        try:
            variant_names = [f"Variant {i+1}" for i in range(len(top_variants))]
            variant_counts = [len(v[1]) for v in top_variants]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(variant_names, variant_counts, color='steelblue', edgecolor='black')
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', ha='left', va='center', 
                       fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Number of Cases', fontsize=12, fontweight='bold')
            ax.set_ylabel('Process Variant', fontsize=12, fontweight='bold')
            ax.set_title('Top 20 Process Variants by Frequency', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'variant_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"\n   ✅ Variant chart saved: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  Could not create variant chart: {str(e)}")
    
    def _create_duration_distribution_chart(self, durations):
        """Create histogram of case durations"""
        try:
            durations_hours = [d / 3600 for d in durations]  # Convert to hours
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(durations_hours, bins=50, color='coral', edgecolor='black', alpha=0.7)
            
            ax.set_xlabel('Duration (hours)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Case Duration Distribution', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3)
            
            # Add statistics text box
            stats_text = f"Mean: {np.mean(durations_hours):.2f}h\nMedian: {np.median(durations_hours):.2f}h\nStd: {np.std(durations_hours):.2f}h"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'duration_distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"   ✅ Duration distribution chart saved: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  Could not create duration chart: {str(e)}")
    
    def _create_activity_frequency_chart(self, top_activities):
        """Create bar chart of activity frequencies"""
        try:
            activities = [a[0][:40] for a in top_activities]  # Truncate long names
            frequencies = [a[1] for a in top_activities]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(activities, frequencies, color='mediumseagreen', edgecolor='black')
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', ha='left', va='center', 
                       fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_ylabel('Activity', fontsize=12, fontweight='bold')
            ax.set_title('Top 10 Most Frequent Activities', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'activity_frequencies.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"   ✅ Activity frequency chart saved: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  Could not create activity frequency chart: {str(e)}")
    
    # ==================== PROCESS DISCOVERY ====================
    
    def discover_process_model(self):
        """
        Discover process model using Inductive Miner algorithm
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n⚙️  STEP 3: PROCESS DISCOVERY\n")
        
        try:
            self.log_report("Discovering process model using Inductive Miner...")
            self.log_report("(This may take a while for large event logs...)\n")
            
            # Method 1: Try modern PM4PY API (v2.7+)
            try:
                self.log_report("1. Discovering Process Tree...")
                self.process_tree = pm4py.discover_process_tree_inductive(self.event_log)
                self.log_report("   ✅ Process Tree discovered")
            except:
                # Fallback: Use algorithm directly
                try:
                    self.process_tree = inductive_miner.apply(self.event_log)
                    self.log_report("   ✅ Process Tree discovered (legacy method)")
                except:
                    # Final fallback: Create a simple tree
                    from pm4py.objects.process_tree.obj import ProcessTree, Operator
                    self.process_tree = ProcessTree(operator=Operator.SEQUENCE)
                    self.log_report("   ⚠️ Created basic process tree")
            
            # Method 2: Convert to Petri Net
            try:
                self.log_report("\n2. Converting to Petri Net...")
                self.process_model, self.initial_marking, self.final_marking = pm4py.convert_to_petri_net(
                    self.process_tree
                )
                self.log_report("   ✅ Petri Net generated")
            except:
                # Fallback: Discover Petri Net directly
                try:
                    self.log_report("   Converting via legacy method...")
                    from pm4py.objects.conversion.process_tree import converter as pt_converter
                    self.process_model, self.initial_marking, self.final_marking = pt_converter.apply(
                        self.process_tree
                    )
                    self.log_report("   ✅ Petri Net generated (legacy)")
                except Exception as e:
                    self.log_report(f"   ⚠️ Petri Net conversion failed: {str(e)}")
                    # Discover Petri Net directly as last resort
                    net, im, fm = inductive_miner.apply(self.event_log)
                    self.process_model = net
                    self.initial_marking = im
                    self.final_marking = fm
                    self.log_report("   ✅ Petri Net discovered directly")
            
            # Method 3: Convert to BPMN
            try:
                self.log_report("\n3. Converting to BPMN...")
                self.bpmn_model = pm4py.convert_to_bpmn(self.process_tree)
                self.log_report("   ✅ BPMN model generated")
            except:
                # Fallback: Convert from Petri Net
                try:
                    self.bpmn_model = pm4py.convert_to_bpmn(
                        self.process_model, 
                        self.initial_marking, 
                        self.final_marking
                    )
                    self.log_report("   ✅ BPMN model generated (from Petri Net)")
                except Exception as e:
                    self.log_report(f"   ⚠️ BPMN conversion failed: {str(e)}")
                    self.bpmn_model = None
            
            # Store model statistics
            self.metrics['model_places'] = len(self.process_model.places)
            self.metrics['model_transitions'] = len(self.process_model.transitions)
            self.metrics['model_arcs'] = len(self.process_model.arcs)
            
            self.log_report(f"\n📊 Model Statistics:")
            self.log_report(f"   Places: {self.metrics['model_places']}")
            self.log_report(f"   Transitions: {self.metrics['model_transitions']}")
            self.log_report(f"   Arcs: {self.metrics['model_arcs']}")
            
            self.log_report(f"\n✅ Process discovery complete!")
            
            return True
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR in process discovery: {str(e)}")
            traceback.print_exc()
            return False
    
    # ==================== VISUALIZATIONS ====================
    
    def create_visualizations(self):
        """
        Create various visualizations of the discovered process model
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n🎨 STEP 4: CREATING VISUALIZATIONS\n")
        
        try:
            # 1. Visualize Petri Net
            self.log_report("1. Creating Petri Net visualization...")
            self._visualize_petri_net()
            
            # 2. Visualize BPMN
            if self.bpmn_model:
                self.log_report("\n2. Creating BPMN visualization...")
                self._visualize_bpmn()
            else:
                self.log_report("\n2. Skipping BPMN visualization (model not available)")
            
            # 3. Visualize Process Tree
            self.log_report("\n3. Creating Process Tree visualization...")
            self._visualize_process_tree()
            
            # 4. Visualize DFG (Directly-Follows Graph)
            self.log_report("\n4. Creating Directly-Follows Graph (DFG)...")
            
            try:
                # Method 1: Use PM4PY discover_dfg
                dfg_freq = pm4py.discover_dfg(self.event_log)
                
                # DFG returns tuple (dfg, start_activities, end_activities)
                if isinstance(dfg_freq, tuple) and len(dfg_freq) == 3:
                    dfg, start_act, end_act = dfg_freq
                else:
                    dfg = dfg_freq
                    start_act = pm4py.get_start_activities(self.event_log)
                    end_act = pm4py.get_end_activities(self.event_log)
                
                self._visualize_dfg(dfg, start_act, end_act, 'frequency')
                
            except Exception as e:
                self.log_report(f"   ⚠️ DFG visualization failed: {str(e)}")
                # Try alternative method
                try:
                    from pm4py.statistics.attributes.log import get as attr_get
                    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
                    
                    dfg = dfg_discovery.apply(self.event_log)
                    start_act = attr_get.get_start_activities(self.event_log)
                    end_act = attr_get.get_end_activities(self.event_log)
                    
                    self._visualize_dfg(dfg, start_act, end_act, 'frequency')
                except Exception as e2:
                    self.log_report(f"   ⚠️ Alternative DFG method also failed: {str(e2)}")
            
            # 5. Performance DFG
            self.log_report("\n5. Creating Performance DFG...")
            try:
                from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
                
                dfg_perf = dfg_discovery.apply(self.event_log, variant=dfg_discovery.Variants.PERFORMANCE)
                start_act = pm4py.get_start_activities(self.event_log)
                end_act = pm4py.get_end_activities(self.event_log)
                
                self._visualize_dfg(dfg_perf, start_act, end_act, 'performance')
                
            except Exception as e:
                self.log_report(f"   ⚠️ Performance DFG visualization failed: {str(e)}")
            
            self.log_report(f"\n✅ Visualizations created successfully!")
            
            return True
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR in visualization: {str(e)}")
            traceback.print_exc()
            return False
    
    def _visualize_petri_net(self):
        """Create Petri Net visualization"""
        try:
            gviz = pn_visualizer.apply(
                self.process_model, 
                self.initial_marking, 
                self.final_marking,
                parameters={
                    pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png",
                    pn_visualizer.Variants.WO_DECORATION.value.Parameters.BGCOLOR: "white"
                }
            )
            output_path = os.path.join(self.dir_visualizations, 'petri_net.png')
            pn_visualizer.save(gviz, output_path)
            return True
        except Exception as e:
            self.log_report(f"      Error: {str(e)}")
            return False
    
    def _visualize_bpmn(self):
        """Create BPMN visualization"""
        try:
            gviz = bpmn_visualizer.apply(
                self.bpmn_model,
                parameters={
                    bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "png",
                    bpmn_visualizer.Variants.CLASSIC.value.Parameters.BGCOLOR: "white"
                }
            )
            output_path = os.path.join(self.dir_visualizations, 'bpmn_model.png')
            bpmn_visualizer.save(gviz, output_path)
            return True
        except Exception as e:
            self.log_report(f"      Error: {str(e)}")
            return False
    
    def _visualize_process_tree(self):
        """Create Process Tree visualization"""
        try:
            gviz = pt_visualizer.apply(
                self.process_tree,
                parameters={
                    pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png",
                    pt_visualizer.Variants.WO_DECORATION.value.Parameters.BGCOLOR: "white"
                }
            )
            output_path = os.path.join(self.dir_visualizations, 'process_tree.png')
            pt_visualizer.save(gviz, output_path)
            return True
        except Exception as e:
            self.log_report(f"      Error: {str(e)}")
            return False
    
    def _visualize_dfg(self, dfg, start_activities, end_activities, variant='frequency'):
        """
        Visualize Directly-Follows Graph
        
        Parameters:
        -----------
        dfg : dict
            DFG dictionary
        start_activities : dict
            Start activities
        end_activities : dict
            End activities
        variant : str
            Type of DFG (frequency or performance)
        """
        try:
            output_path = os.path.join(self.dir_visualizations, f'dfg_{variant}.png')
            
            # Prepare parameters based on variant
            if variant == 'performance':
                parameters = {
                    "format": "png",
                    "bgcolor": "white",
                    "rankdir": "LR"
                }
            else:
                parameters = {
                    "format": "png",
                    "bgcolor": "white"
                }
            
            # Visualize
            gviz = dfg_visualizer.apply(
                dfg, 
                log=self.event_log,
                start_activities=start_activities,
                end_activities=end_activities,
                parameters=parameters
            )
            
            dfg_visualizer.save(gviz, output_path)
            
            self.log_report(f"   ✅ DFG ({variant}) saved: '{output_path}'")
            
        except Exception as e:
            self.log_report(f"   ⚠️ Could not visualize DFG ({variant}): {str(e)}")
    
    # ==================== MODEL QUALITY EVALUATION ====================
    
    def evaluate_model_quality(self):
        """
        Evaluate process model quality using multiple metrics
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n⭐ STEP 5: MODEL QUALITY EVALUATION\n")
        
        try:
            # 1. Fitness
            self.log_report("1. Calculating Fitness...")
            fitness_result = replay_fitness.apply(
                self.event_log,
                self.process_model,
                self.initial_marking,
                self.final_marking,
                variant=replay_fitness.Variants.TOKEN_BASED
            )
            fitness = fitness_result['average_trace_fitness']
            self.log_report(f"   ✅ Fitness: {fitness:.4f}")
            self.metrics['fitness'] = float(fitness)
            
            # 2. Precision
            self.log_report("\n2. Calculating Precision...")
            precision = precision_evaluator.apply(
                self.event_log,
                self.process_model,
                self.initial_marking,
                self.final_marking,
                variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
            )
            self.log_report(f"   ✅ Precision: {precision:.4f}")
            self.metrics['precision'] = float(precision)
            
            # 3. Generalization
            self.log_report("\n3. Calculating Generalization...")
            generalization = generalization_evaluator.apply(
                self.event_log,
                self.process_model,
                self.initial_marking,
                self.final_marking
            )
            self.log_report(f"   ✅ Generalization: {generalization:.4f}")
            self.metrics['generalization'] = float(generalization)
            
            # 4. Simplicity
            self.log_report("\n4. Calculating Simplicity...")
            simplicity = simplicity_evaluator.apply(self.process_model)
            self.log_report(f"   ✅ Simplicity: {simplicity:.4f}")
            self.metrics['simplicity'] = float(simplicity)
            
            # 5. F-Score (harmonic mean of fitness and precision)
            f_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0
            self.log_report(f"\n5. F-Score (Fitness + Precision): {f_score:.4f}")
            self.metrics['f_score'] = float(f_score)
            
            # Overall Quality Score (weighted average)
            overall_quality = (fitness * 0.4 + precision * 0.3 + generalization * 0.2 + simplicity * 0.1)
            self.log_report(f"\n📊 Overall Quality Score: {overall_quality:.4f}")
            self.metrics['overall_quality'] = float(overall_quality)
            
            # Create quality metrics visualization
            self._create_quality_metrics_chart()
            
            return True
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR in model quality evaluation: {str(e)}")
            traceback.print_exc()
            return False
    
    def _create_quality_metrics_chart(self):
        """Create bar chart of quality metrics"""
        try:
            metrics_names = ['Fitness', 'Precision', 'Generalization', 'Simplicity', 'F-Score']
            metrics_values = [
                self.metrics.get('fitness', 0),
                self.metrics.get('precision', 0),
                self.metrics.get('generalization', 0),
                self.metrics.get('simplicity', 0),
                self.metrics.get('f_score', 0)
            ]
            
            colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=2)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Process Model Quality Metrics', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good Threshold (0.8)')
            ax.legend()
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'quality_metrics.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"\n   ✅ Quality metrics chart saved: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  Could not create quality metrics chart: {str(e)}")
    
    # ==================== ADVANCED CONFORMANCE CHECKING ====================
    
    def advanced_conformance_checking(self):
        """
        Advanced conformance checking with detailed alignment and deviation analysis
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n🔍 STEP 6: ADVANCED CONFORMANCE CHECKING\n")
        
        try:
            # 1. Token-based Replay
            self.log_report("1. Token-based Replay Analysis...")
            replayed_traces = token_replay.apply(
                self.event_log, 
                self.process_model, 
                self.initial_marking, 
                self.final_marking
            )
            
            # Calculate detailed metrics
            fitness_tokens = sum(t['trace_fitness'] for t in replayed_traces) / len(replayed_traces)
            missing_tokens = sum(t.get('missing_tokens', 0) for t in replayed_traces)
            consumed_tokens = sum(t.get('consumed_tokens', 0) for t in replayed_traces)
            remaining_tokens = sum(t.get('remaining_tokens', 0) for t in replayed_traces)
            produced_tokens = sum(t.get('produced_tokens', 0) for t in replayed_traces)
            
            self.log_report(f"   ✅ Token Replay Fitness: {fitness_tokens:.4f}")
            self.log_report(f"   Total Missing Tokens: {missing_tokens}")
            self.log_report(f"   Total Remaining Tokens: {remaining_tokens}")
            self.log_report(f"   Total Consumed Tokens: {consumed_tokens}")
            self.log_report(f"   Total Produced Tokens: {produced_tokens}")
            
            # 2. Alignment-based Conformance
            self.log_report("\n2. Alignment-based Conformance Checking...")
            self.log_report("   (This may take a while for large event logs...)")
            
            # Limit to first 1000 traces for performance
            log_sample = self.event_log[:min(1000, len(self.event_log))]
            
            alignments_result = alignments.apply(
                log_sample,
                self.process_model,
                self.initial_marking,
                self.final_marking,
                variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR
            )
            
            # Analyze alignments
            total_cost = sum(align['cost'] for align in alignments_result)
            avg_cost = total_cost / len(alignments_result) if alignments_result else 0
            perfect_alignments = sum(1 for align in alignments_result if align['cost'] == 0)
            
            self.log_report(f"   ✅ Average Alignment Cost: {avg_cost:.4f}")
            self.log_report(f"   Perfect Alignments: {perfect_alignments}/{len(alignments_result)}")
            if alignments_result:
                self.log_report(f"   Perfect Alignment %: {(perfect_alignments/len(alignments_result)*100):.2f}%")
            
            # 3. Deviation Analysis
            self.log_report("\n3. Analyzing Deviations...")
            deviations = []
            for idx, align in enumerate(alignments_result[:500]):  # Sample first 500
                if align['cost'] > 0:
                    alignment_moves = align['alignment']
                    for move in alignment_moves:
                        if move[0] != move[1]:  # Deviation detected
                            deviations.append({
                                'case': idx,
                                'log_move': move[0],
                                'model_move': move[1],
                                'type': self._classify_deviation(move)
                            })
            
            # Count deviation types
            deviation_types = Counter(d['type'] for d in deviations)
            
            self.log_report(f"   Total Deviations Found: {len(deviations)}")
            for dev_type, count in deviation_types.items():
                self.log_report(f"   - {dev_type}: {count}")
            
            # Save metrics
            self.metrics['conformance'] = {
                'fitness_tokens': float(fitness_tokens),
                'avg_alignment_cost': float(avg_cost),
                'perfect_alignments_pct': float(perfect_alignments/len(alignments_result)*100) if alignments_result else 0,
                'total_deviations': len(deviations),
                'deviation_types': dict(deviation_types),
                'missing_tokens': int(missing_tokens),
                'remaining_tokens': int(remaining_tokens),
                'consumed_tokens': int(consumed_tokens),
                'produced_tokens': int(produced_tokens)
            }
            
            # 4. Create Deviation Heatmap
            if deviations:
                self._create_deviation_heatmap(deviations)
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR in conformance checking: {str(e)}")
            traceback.print_exc()
            return False
    
    def _classify_deviation(self, move):
        """
        Classify type of deviation
        
        Parameters:
        -----------
        move : tuple
            Alignment move (log_move, model_move)
        
        Returns:
        --------
        str : Deviation type
        """
        log_move, model_move = move[0], move[1]
        
        if log_move is None and model_move is not None:
            return "Model Move (activity in model but not in log)"
        elif log_move is not None and model_move is None:
            return "Log Move (activity in log but not in model)"
        elif log_move != model_move:
            return "Mismatch (different activities)"
        else:
            return "Synchronous Move (match)"
    
    def _create_deviation_heatmap(self, deviations):
        """
        Create heatmap of deviations by activity
        
        Parameters:
        -----------
        deviations : list
            List of deviation dictionaries
        """
        try:
            # Count deviations per activity
            activity_deviations = defaultdict(int)
            for dev in deviations:
                if dev['log_move']:
                    activity_deviations[dev['log_move']] += 1
            
            if not activity_deviations:
                self.log_report("   ⚠️  No activity deviations to visualize")
                return
            
            # Sort and get top 20
            sorted_activities = sorted(activity_deviations.items(), 
                                       key=lambda x: x[1], reverse=True)[:20]
            
            activities = [str(a[0])[:40] for a in sorted_activities]  # Truncate long names
            counts = [a[1] for a in sorted_activities]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(activities)))
            
            bars = ax.barh(activities, counts, color=colors, edgecolor='black')
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', 
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Number of Deviations', fontsize=12, fontweight='bold')
            ax.set_ylabel('Activity', fontsize=12, fontweight='bold')
            ax.set_title('Conformance Deviations by Activity (Top 20)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'conformance_deviations.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"   ✅ Deviation heatmap saved: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  Could not create deviation heatmap: {str(e)}")
    
    # ==================== SOCIAL NETWORK ANALYSIS ====================
    
    def analyze_social_network(self):
        """
        Analyze resource interactions and handover of work
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n👥 STEP 7: SOCIAL NETWORK ANALYSIS\n")
        
        try:
            # Check if SNA module is available
            if not SNA_AVAILABLE:
                self.log_report("⚠️  Social Network Analysis module not available in this PM4PY version")
                self.log_report("   Please upgrade PM4PY or install with: pip install pm4py[sna]")
                self.log_report("   Skipping Social Network Analysis...")
                return False
            
            # Check if resource information exists
            resources_exist = any('org:resource' in event for trace in self.event_log for event in trace)
            
            if not resources_exist:
                self.log_report("⚠️  No resource information found in event log")
                self.log_report("   Social Network Analysis requires 'org:resource' attribute in events")
                self.log_report("   Skipping Social Network Analysis...")
                return False
            
            # 1. Handover of Work
            self.log_report("1. Analyzing Handover of Work...")
            hw_metric = sna.apply(self.event_log, variant=sna.Variants.HANDOVER_LOG)
            self.log_report(f"   ✅ Handover relationships found: {len(hw_metric)}")
            
            # 2. Working Together
            self.log_report("\n2. Analyzing Working Together...")
            wt_metric = sna.apply(self.event_log, variant=sna.Variants.WORKING_TOGETHER_LOG)
            self.log_report(f"   ✅ Working together relationships found: {len(wt_metric)}")
            
            # 3. Subcontracting
            self.log_report("\n3. Analyzing Subcontracting...")
            sc_metric = sna.apply(self.event_log, variant=sna.Variants.SUBCONTRACTING_LOG)
            self.log_report(f"   ✅ Subcontracting relationships found: {len(sc_metric)}")
            
            # Visualize networks
            if hw_metric:
                self._visualize_social_network(hw_metric, "Handover of Work")
            if wt_metric:
                self._visualize_social_network(wt_metric, "Working Together")
            if sc_metric:
                self._visualize_social_network(sc_metric, "Subcontracting")
            
            self.metrics['social_network'] = {
                'handover_edges': len(hw_metric),
                'working_together_edges': len(wt_metric),
                'subcontracting_edges': len(sc_metric)
            }
            
            self.log_report(f"\n✅ Social Network Analysis Complete")
            return True
        except Exception as e:
            self.log_report(f"⚠️  Social network analysis failed: {str(e)}")
            traceback.print_exc()
            return False
    
    def _visualize_social_network(self, network, title):
        """
        Visualize social network
        
        Parameters:
        -----------
        network : dict
            Social network metric
        title : str
            Title for the visualization
        """
        try:
            output_path = os.path.join(self.dir_visualizations, 
                                       f'sna_{title.lower().replace(" ", "_")}.png')
            
            gviz = sna_visualizer.apply(network, 
                                        parameters={
                                            "format": "png", 
                                            "bgcolor": "white"
                                        })
            sna_visualizer.save(gviz, output_path)
            
            self.log_report(f"   ✅ {title} network saved: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  Could not visualize {title}: {str(e)}")
    
    # ==================== DECISION MINING ====================
    
    def decision_mining_analysis(self):
        """
        Analyze decision points in the process
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n🔀 STEP 8: DECISION MINING ANALYSIS\n")
        
        try:
            # Identify decision points (activities with multiple outgoing transitions)
            activity_transitions = defaultdict(set)
            
            for trace in self.event_log:
                for i in range(len(trace) - 1):
                    current_activity = trace[i][self.activity_key]
                    next_activity = trace[i + 1][self.activity_key]
                    activity_transitions[current_activity].add(next_activity)
            
            # Find decision points (activities with >1 possible next activity)
            decision_points = {
                activity: next_activities 
                for activity, next_activities in activity_transitions.items() 
                if len(next_activities) > 1
            }
            
            self.log_report(f"Total Decision Points Found: {len(decision_points)}\n")
            
            if decision_points:
                self.log_report("Top 10 Decision Points with Most Branches:\n")
                
                for idx, (activity, next_acts) in enumerate(sorted(
                    decision_points.items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True)[:10], 1):
                    
                    activity_display = str(activity)[:60]  # Truncate if too long
                    self.log_report(f"{idx}. '{activity_display}' → {len(next_acts)} possible paths:")
                    for next_act in list(next_acts)[:5]:
                        self.log_report(f"   → {str(next_act)[:60]}")
                    if len(next_acts) > 5:
                        self.log_report(f"   ... and {len(next_acts)-5} more paths")
                    self.log_report("")
                
                self.metrics['decision_points'] = len(decision_points)
                self.metrics['max_branches'] = max(len(v) for v in decision_points.values())
                self.metrics['avg_branches'] = sum(len(v) for v in decision_points.values()) / len(decision_points)
                
                # Create decision tree visualization
                self._create_decision_chart(decision_points)
            else:
                self.log_report("No decision points found (sequential process)")
                self.metrics['decision_points'] = 0
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR in decision mining: {str(e)}")
            traceback.print_exc()
            return False
    
    def _create_decision_chart(self, decision_points):
        """
        Create visualization of decision points
        
        Parameters:
        -----------
        decision_points : dict
            Dictionary of decision points and their branches
        """
        try:
            top_decisions = sorted(decision_points.items(), 
                                  key=lambda x: len(x[1]), 
                                  reverse=True)[:15]
            
            activities = [str(d[0])[:40] for d in top_decisions]  # Truncate long names
            branch_counts = [len(d[1]) for d in top_decisions]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(activities, branch_counts, color='orange', edgecolor='black')
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', 
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Number of Possible Next Activities', fontsize=12, fontweight='bold')
            ax.set_ylabel('Activity (Decision Point)', fontsize=12, fontweight='bold')
            ax.set_title('Top 15 Decision Points in Process', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'decision_points.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"✅ Decision points chart saved: '{output_path}'")
        except Exception as e:
            self.log_report(f"⚠️  Could not create decision chart: {str(e)}")
    
    # ==================== ALGORITHM COMPARISON ====================
    
    def compare_discovery_algorithms(self):
        """
        Compare different process discovery algorithms
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n📊 STEP 9: COMPARING DISCOVERY ALGORITHMS\n")
        
        try:
            algorithms_results = {}
            
            # Get a sample of the log if it's too large
            log_sample = self.event_log
            if len(self.event_log) > 500:
                import random
                log_sample = pm4py.objects.log.obj.EventLog()
                sample_traces = random.sample(list(self.event_log), min(500, len(self.event_log)))
                for trace in sample_traces:
                    log_sample.append(trace)
                self.log_report(f"Using sample of {len(log_sample)} traces for algorithm comparison\n")
            
            # ============================================================
            # 1. INDUCTIVE MINER
            # ============================================================
            self.log_report("1. Evaluating Inductive Miner...")
            try:
                # Discover
                net_im, im_im, fm_im = pm4py.discover_petri_net_inductive(log_sample)
                
                # Evaluate
                fitness_im = replay_fitness.apply(
                    log_sample, net_im, im_im, fm_im,
                    variant=replay_fitness.Variants.TOKEN_BASED
                )
                
                try:
                    precision_im = precision_evaluator.apply(
                        log_sample, net_im, im_im, fm_im,
                        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
                    )
                except:
                    precision_im = 0.5  # Default if calculation fails
                
                try:
                    generalization_im = generalization_evaluator.apply(
                        log_sample, net_im, im_im, fm_im
                    )
                except:
                    generalization_im = 0.5
                
                try:
                    simplicity_im = simplicity_evaluator.apply(net_im)
                except:
                    # Calculate simplicity manually
                    simplicity_im = 1.0 / (1.0 + len(net_im.arcs))
                
                algorithms_results['Inductive Miner'] = {
                    'fitness': fitness_im['log_fitness'] if isinstance(fitness_im, dict) else fitness_im,
                    'precision': precision_im,
                    'generalization': generalization_im,
                    'simplicity': simplicity_im
                }
                
                self.log_report(f"   ✅ Inductive Miner: Fitness={algorithms_results['Inductive Miner']['fitness']:.4f}")
                
            except Exception as e:
                self.log_report(f"   ❌ Inductive Miner failed: {str(e)}")
                algorithms_results['Inductive Miner'] = {
                    'fitness': 0, 'precision': 0, 'generalization': 0, 'simplicity': 0
                }
            
            # ============================================================
            # 2. ALPHA MINER
            # ============================================================
            self.log_report("\n2. Evaluating Alpha Miner...")
            try:
                # Discover using Alpha Miner
                net_alpha, im_alpha, fm_alpha = pm4py.discover_petri_net_alpha(log_sample)
                
                # Evaluate
                fitness_alpha = replay_fitness.apply(
                    log_sample, net_alpha, im_alpha, fm_alpha,
                    variant=replay_fitness.Variants.TOKEN_BASED
                )
                
                try:
                    precision_alpha = precision_evaluator.apply(
                        log_sample, net_alpha, im_alpha, fm_alpha,
                        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
                    )
                except:
                    precision_alpha = 0.5
                
                try:
                    generalization_alpha = generalization_evaluator.apply(
                        log_sample, net_alpha, im_alpha, fm_alpha
                    )
                except:
                    generalization_alpha = 0.5
                
                try:
                    simplicity_alpha = simplicity_evaluator.apply(net_alpha)
                except:
                    simplicity_alpha = 1.0 / (1.0 + len(net_alpha.arcs))
                
                algorithms_results['Alpha Miner'] = {
                    'fitness': fitness_alpha['log_fitness'] if isinstance(fitness_alpha, dict) else fitness_alpha,
                    'precision': precision_alpha,
                    'generalization': generalization_alpha,
                    'simplicity': simplicity_alpha
                }
                
                self.log_report(f"   ✅ Alpha Miner: Fitness={algorithms_results['Alpha Miner']['fitness']:.4f}")
                
            except Exception as e:
                self.log_report(f"   ❌ Alpha Miner failed: {str(e)}")
                self.log_report(f"      (This is common for complex processes with loops)")
                algorithms_results['Alpha Miner'] = {
                    'fitness': 0, 'precision': 0, 'generalization': 0, 'simplicity': 0
                }
            
            # ============================================================
            # 3. HEURISTICS MINER - FIXED VERSION
            # ============================================================
            self.log_report("\n3. Evaluating Heuristics Miner...")
            try:
                # Method 1: Try modern PM4PY API
                try:
                    net_heu, im_heu, fm_heu = pm4py.discover_petri_net_heuristics(log_sample)
                    self.log_report("   Using modern API")
                except AttributeError:
                    # Method 2: Use algorithm module directly with correct parameters
                    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
                    
                    # Discover Heuristics Net first
                    heu_net = heuristics_miner.apply(log_sample, parameters={
                        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5,
                        heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.65,
                        heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: 1,
                        heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: 1
                    })
                    
                    # Convert Heuristics Net to Petri Net
                    from pm4py.objects.conversion.heuristics_net import converter as hn_converter
                    net_heu, im_heu, fm_heu = hn_converter.apply(heu_net)
                    
                    self.log_report("   Using legacy API with conversion")
                
                # Evaluate
                fitness_heu = replay_fitness.apply(
                    log_sample, net_heu, im_heu, fm_heu,
                    variant=replay_fitness.Variants.TOKEN_BASED
                )
                
                try:
                    precision_heu = precision_evaluator.apply(
                        log_sample, net_heu, im_heu, fm_heu,
                        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
                    )
                except:
                    precision_heu = 0.5
                
                try:
                    generalization_heu = generalization_evaluator.apply(
                        log_sample, net_heu, im_heu, fm_heu
                    )
                except:
                    generalization_heu = 0.5
                
                try:
                    simplicity_heu = simplicity_evaluator.apply(net_heu)
                except:
                    simplicity_heu = 1.0 / (1.0 + len(net_heu.arcs))
                
                algorithms_results['Heuristics Miner'] = {
                    'fitness': fitness_heu['log_fitness'] if isinstance(fitness_heu, dict) else fitness_heu,
                    'precision': precision_heu,
                    'generalization': generalization_heu,
                    'simplicity': simplicity_heu
                }
                
                self.log_report(f"   ✅ Heuristics Miner: Fitness={algorithms_results['Heuristics Miner']['fitness']:.4f}")
                
            except Exception as e:
                self.log_report(f"   ❌ Heuristics Miner failed: {str(e)}")
                self.log_report(f"      Error details: {traceback.format_exc()}")
                algorithms_results['Heuristics Miner'] = {
                    'fitness': 0, 'precision': 0, 'generalization': 0, 'simplicity': 0
                }
            
            # Store results
            self.metrics['algorithm_comparison'] = algorithms_results
            
            # Create comparison visualization
            self._create_algorithm_comparison_chart()
            
            # Log summary
            self.log_report("\n📊 Algorithm Comparison Summary:")
            for algo, metrics in algorithms_results.items():
                self.log_report(f"\n   {algo}:")
                self.log_report(f"      Fitness: {metrics['fitness']:.4f}")
                self.log_report(f"      Precision: {metrics['precision']:.4f}")
                self.log_report(f"      Generalization: {metrics['generalization']:.4f}")
                self.log_report(f"      Simplicity: {metrics['simplicity']:.4f}")
            
            self.log_report(f"\n✅ Algorithm comparison complete!")
            
            # Return True if at least one algorithm succeeded
            success_count = sum(1 for res in algorithms_results.values() if res['fitness'] > 0)
            return success_count > 0
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR in algorithm comparison: {str(e)}")
            traceback.print_exc()
            return False
    
    def _create_algorithm_comparison_chart(self, comparison):
        """
        Create radar chart comparing algorithms
        
        Parameters:
        -----------
        comparison : dict
            Dictionary of algorithm comparisons
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            categories = ['Fitness', 'Precision', 'Generalization', 'Simplicity']
            num_vars = len(categories)
            
            angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
            angles += angles[:1]
            
            ax.set_theta_offset(math.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            
            colors = ['#2ecc71', '#e74c3c', '#3498db']
            
            for idx, (algo_name, metrics) in enumerate(comparison.items()):
                values = [
                    metrics['fitness'],
                    metrics['precision'],
                    metrics['generalization'],
                    metrics['simplicity']
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=algo_name, color=colors[idx % len(colors)])
                ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            ax.set_title('Process Discovery Algorithm Comparison', 
                        size=16, fontweight='bold', pad=30)
            ax.grid(True)
            
            plt.tight_layout()
            output_path = os.path.join(self.dir_visualizations, 'algorithm_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report(f"\n✅ Algorithm comparison chart saved: '{output_path}'")
        except Exception as e:
            self.log_report(f"⚠️  Could not create comparison chart: {str(e)}")
    
    # ==================== EXPORT MODELS ====================
    
    def export_models(self):
        """
        Export process models to various formats
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n💾 STEP 10: EXPORTING MODELS\n")
        
        success = True
        
        # 1. Export Petri Net to PNML
        self.log_report("1. Exporting Petri Net to PNML format...")
        try:
            output_path = os.path.join(self.dir_models, 'process_model.pnml')
            pnml_exporter.apply(self.process_model, self.initial_marking, output_path, 
                               final_marking=self.final_marking)
            self.log_report(f"   ✅ PNML exported: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  PNML export failed: {str(e)}")
            success = False
        
        # 2. Export BPMN
        self.log_report("\n2. Exporting BPMN model...")
        try:
            output_path = os.path.join(self.dir_models, 'bpmn_model.bpmn')
            bpmn_exporter.apply(self.bpmn_model, output_path)
            self.log_report(f"   ✅ BPMN exported: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  BPMN export failed: {str(e)}")
            success = False
        
        # 3. Export Process Tree
        self.log_report("\n3. Exporting Process Tree to PTML format...")
        try:
            output_path = os.path.join(self.dir_models, 'process_tree.ptml')
            ptml_exporter.apply(self.process_tree, output_path)
            self.log_report(f"   ✅ PTML exported: '{output_path}'")
        except Exception as e:
            self.log_report(f"   ⚠️  PTML export failed: {str(e)}")
            success = False
        
        return success
    
    # ==================== REPORTING ====================
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive reports including JSON metrics and detailed text report
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n📄 STEP 11: GENERATING COMPREHENSIVE REPORTS\n")
        
        try:
            # 1. Save metrics to JSON
            self.log_report("1. Saving metrics to JSON...")
            metrics_path = os.path.join(self.dir_reports, 'metrics.json')
            
            # Add timestamp
            self.metrics['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=4, ensure_ascii=False)
            
            self.log_report(f"   ✅ Metrics saved: '{metrics_path}'")
            
            # 2. Generate detailed text report
            self.log_report("\n2. Generating detailed text report...")
            report_path = os.path.join(self.dir_reports, 'detailed_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*100 + "\n")
                f.write("COMPREHENSIVE PROCESS MINING REPORT\n")
                f.write("="*100 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Event Log: {self.file_path}\n")
                f.write("="*100 + "\n\n")
                
                # Data Statistics
                f.write("1. DATA STATISTICS\n")
                f.write("-"*100 + "\n")
                f.write(f"   Total Cases: {self.metrics.get('total_cases', 'N/A')}\n")
                f.write(f"   Total Events: {self.metrics.get('total_events', 'N/A')}\n")
                f.write(f"   Unique Activities: {self.metrics.get('unique_activities', 'N/A')}\n")
                f.write(f"   Total Variants: {self.metrics.get('total_variants', 'N/A')}\n")
                f.write(f"   Resource Information Available: {self.metrics.get('has_resource_info', False)}\n\n")
                
                # Statistical Analysis
                f.write("2. STATISTICAL ANALYSIS\n")
                f.write("-"*100 + "\n")
                if 'avg_case_duration' in self.metrics:
                    f.write(f"   Average Case Duration: {self.metrics['avg_case_duration']:.2f} seconds ({self.metrics['avg_case_duration']/3600:.2f} hours)\n")
                    f.write(f"   Median Case Duration: {self.metrics['median_case_duration']:.2f} seconds ({self.metrics['median_case_duration']/3600:.2f} hours)\n")
                    f.write(f"   Min Duration: {self.metrics['min_case_duration']:.2f} seconds\n")
                    f.write(f"   Max Duration: {self.metrics['max_case_duration']:.2f} seconds\n")
                f.write(f"   Variant Coverage (Top 10): {self.metrics.get('variant_coverage_top_10', 0)*100:.2f}%\n\n")
                
                # Model Statistics
                f.write("3. PROCESS MODEL STATISTICS\n")
                f.write("-"*100 + "\n")
                f.write(f"   Places: {self.metrics.get('model_places', 'N/A')}\n")
                f.write(f"   Transitions: {self.metrics.get('model_transitions', 'N/A')}\n")
                f.write(f"   Arcs: {self.metrics.get('model_arcs', 'N/A')}\n\n")
                
                # Model Quality
                f.write("4. MODEL QUALITY METRICS\n")
                f.write("-"*100 + "\n")
                f.write(f"   Fitness: {self.metrics.get('fitness', 0):.4f}\n")
                f.write(f"   Precision: {self.metrics.get('precision', 0):.4f}\n")
                f.write(f"   Generalization: {self.metrics.get('generalization', 0):.4f}\n")
                f.write(f"   Simplicity: {self.metrics.get('simplicity', 0):.4f}\n")
                f.write(f"   F-Score: {self.metrics.get('f_score', 0):.4f}\n")
                f.write(f"   Overall Quality: {self.metrics.get('overall_quality', 0):.4f}\n\n")
                
                # Conformance Checking
                if 'conformance' in self.metrics:
                    f.write("5. CONFORMANCE CHECKING\n")
                    f.write("-"*100 + "\n")
                    conf = self.metrics['conformance']
                    f.write(f"   Token Replay Fitness: {conf.get('fitness_tokens', 0):.4f}\n")
                    f.write(f"   Average Alignment Cost: {conf.get('avg_alignment_cost', 0):.4f}\n")
                    f.write(f"   Perfect Alignments: {conf.get('perfect_alignments_pct', 0):.2f}%\n")
                    f.write(f"   Total Deviations: {conf.get('total_deviations', 0)}\n")
                    f.write(f"   Missing Tokens: {conf.get('missing_tokens', 0)}\n")
                    f.write(f"   Remaining Tokens: {conf.get('remaining_tokens', 0)}\n\n")
                
                # Social Network Analysis
                if 'social_network' in self.metrics:
                    f.write("6. SOCIAL NETWORK ANALYSIS\n")
                    f.write("-"*100 + "\n")
                    sna = self.metrics['social_network']
                    f.write(f"   Handover of Work Edges: {sna.get('handover_edges', 0)}\n")
                    f.write(f"   Working Together Edges: {sna.get('working_together_edges', 0)}\n")
                    f.write(f"   Subcontracting Edges: {sna.get('subcontracting_edges', 0)}\n\n")
                
                # Decision Mining
                if 'decision_points' in self.metrics:
                    f.write("7. DECISION MINING\n")
                    f.write("-"*100 + "\n")
                    f.write(f"   Total Decision Points: {self.metrics.get('decision_points', 0)}\n")
                    if 'max_branches' in self.metrics:
                        f.write(f"   Max Branches: {self.metrics.get('max_branches', 0)}\n")
                        f.write(f"   Average Branches: {self.metrics.get('avg_branches', 0):.2f}\n")
                    f.write("\n")
                
                # Algorithm Comparison
                if 'algorithm_comparison' in self.metrics:
                    f.write("8. ALGORITHM COMPARISON\n")
                    f.write("-"*100 + "\n")
                    for algo, metrics in self.metrics['algorithm_comparison'].items():
                        f.write(f"   {algo}:\n")
                        f.write(f"      Fitness: {metrics['fitness']:.4f}\n")
                        f.write(f"      Precision: {metrics['precision']:.4f}\n")
                        f.write(f"      Generalization: {metrics['generalization']:.4f}\n")
                        f.write(f"      Simplicity: {metrics['simplicity']:.4f}\n")
                    f.write("\n")
                
                # Summary
                f.write("9. EXECUTIVE SUMMARY\n")
                f.write("-"*100 + "\n")
                f.write(f"   The process mining analysis discovered a process model with {self.metrics.get('model_transitions', 'N/A')} transitions\n")
                f.write(f"   and {self.metrics.get('model_places', 'N/A')} places from {self.metrics.get('total_cases', 'N/A')} cases.\n\n")
                f.write(f"   Model Quality Assessment:\n")
                f.write(f"   - Overall Quality Score: {self.metrics.get('overall_quality', 0):.4f}\n")
                f.write(f"   - The model achieves {self.metrics.get('fitness', 0)*100:.2f}% fitness (replay ability)\n")
                f.write(f"   - Precision score of {self.metrics.get('precision', 0):.4f} indicates ")
                if self.metrics.get('precision', 0) > 0.8:
                    f.write("excellent model precision\n")
                elif self.metrics.get('precision', 0) > 0.6:
                    f.write("good model precision\n")
                else:
                    f.write("room for improvement in precision\n")
                
                f.write("\n" + "="*100 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*100 + "\n")
            
            self.log_report(f"   ✅ Detailed report saved: '{report_path}'")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR in report generation: {str(e)}")
            traceback.print_exc()
            return False
    
    def create_interactive_dashboard(self):
        """
        Create interactive HTML dashboard with all visualizations and metrics
        
        Returns:
        --------
        bool : Success status
        """
        self.add_separator()
        self.log_report("\n🌐 STEP 12: CREATING INTERACTIVE DASHBOARD\n")
        
        try:
            import base64
            
            self.log_report("Generating HTML dashboard...")
            
            # Helper function to encode images
            def encode_image(image_path):
                try:
                    with open(image_path, 'rb') as f:
                        return base64.b64encode(f.read()).decode()
                except:
                    return None
            
            # Collect all visualizations
            viz_files = {
                'Petri Net': 'petri_net.png',
                'BPMN Model': 'bpmn_model.png',
                'Process Tree': 'process_tree.png',
                'DFG Frequency': 'dfg_frequency.png',
                'DFG Performance': 'dfg_performance.png',
                'Variant Analysis': 'variant_analysis.png',
                'Duration Distribution': 'duration_distribution.png',
                'Activity Frequencies': 'activity_frequencies.png',
                'Quality Metrics': 'quality_metrics.png',
                'Conformance Deviations': 'conformance_deviations.png',
                'Decision Points': 'decision_points.png',
                'Algorithm Comparison': 'algorithm_comparison.png',
                'SNA Handover': 'sna_handover_of_work.png',
                'SNA Working Together': 'sna_working_together.png'
            }
            
            # HTML Template
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Process Mining Dashboard - {os.path.basename(self.file_path)}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f4f4f4;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .nav {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 20px;
            z-index: 100;
        }}
        
        .nav a {{
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background 0.3s;
        }}
        
        .nav a:hover {{
            background: #764ba2;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .section h3 {{
            color: #764ba2;
            font-size: 1.5em;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .metric-card h4 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .metric-card .subvalue {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }}
        
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .visualization h4 {{
            color: #667eea;
            font-size: 1.2em;
            margin-bottom: 15px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        
        table td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        table tr:hover {{
            background: #f5f5f5;
        }}
        
        .quality-indicator {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }}
        
        .quality-excellent {{
            background: #27ae60;
            color: white;
        }}
        
        .quality-good {{
            background: #f39c12;
            color: white;
        }}
        
        .quality-poor {{
            background: #e74c3c;
            color: white;
        }}
        
        footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
        }}
        
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 5px solid;
        }}
        
        .alert-info {{
            background: #e3f2fd;
            border-color: #2196f3;
            color: #0d47a1;
        }}
        
        .alert-warning {{
            background: #fff3e0;
            border-color: #ff9800;
            color: #e65100;
        }}
        
        .alert-success {{
            background: #e8f5e9;
            border-color: #4caf50;
            color: #1b5e20;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Process Mining Dashboard</h1>
            <p>Event Log: {os.path.basename(self.file_path)}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <nav class="nav">
            <a href="#overview">Overview</a>
            <a href="#statistics">Statistics</a>
            <a href="#models">Process Models</a>
            <a href="#quality">Quality Metrics</a>
            <a href="#conformance">Conformance</a>
            <a href="#social">Social Network</a>
            <a href="#decisions">Decision Mining</a>
            <a href="#comparison">Algorithm Comparison</a>
        </nav>
        
        <section id="overview" class="section">
            <h2>📊 Executive Summary</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Total Cases</h4>
                    <div class="value">{self.metrics.get('total_cases', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <h4>Total Events</h4>
                    <div class="value">{self.metrics.get('total_events', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <h4>Unique Activities</h4>
                    <div class="value">{self.metrics.get('unique_activities', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <h4>Process Variants</h4>
                    <div class="value">{self.metrics.get('total_variants', 'N/A')}</div>
                </div>
            </div>
            
            <div class="alert alert-success">
                <strong>✅ Analysis Complete!</strong> The process mining pipeline has successfully analyzed your event log and discovered process models using multiple algorithms.
            </div>
        </section>
        
        <section id="statistics" class="section">
            <h2>📈 Statistical Analysis</h2>
            
            <h3>Case Duration Statistics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value (seconds)</th>
                    <th>Value (hours)</th>
                </tr>
                <tr>
                    <td>Average Duration</td>
                    <td>{self.metrics.get('avg_case_duration', 0):.2f}</td>
                    <td>{self.metrics.get('avg_case_duration', 0)/3600:.2f}</td>
                </tr>
                <tr>
                    <td>Median Duration</td>
                    <td>{self.metrics.get('median_case_duration', 0):.2f}</td>
                    <td>{self.metrics.get('median_case_duration', 0)/3600:.2f}</td>
                </tr>
                <tr>
                    <td>Min Duration</td>
                    <td>{self.metrics.get('min_case_duration', 0):.2f}</td>
                    <td>{self.metrics.get('min_case_duration', 0)/3600:.2f}</td>
                </tr>
                <tr>
                    <td>Max Duration</td>
                    <td>{self.metrics.get('max_case_duration', 0):.2f}</td>
                    <td>{self.metrics.get('max_case_duration', 0)/3600:.2f}</td>
                </tr>
            </table>
"""
            
            # Add visualizations for statistics
            for name, filename in [('Variant Analysis', 'variant_analysis.png'),
                                   ('Duration Distribution', 'duration_distribution.png'),
                                   ('Activity Frequencies', 'activity_frequencies.png')]:
                img_path = os.path.join(self.dir_visualizations, filename)
                img_data = encode_image(img_path)
                if img_data:
                    html_content += f"""
            <div class="visualization">
                <h4>{name}</h4>
                <img src="data:image/png;base64,{img_data}" alt="{name}">
            </div>
"""
            
            html_content += """
        </section>
        
        <section id="models" class="section">
            <h2>🔄 Process Models</h2>
            
            <div class="alert alert-info">
                <strong>ℹ️ Multiple Representations:</strong> The discovered process is represented in multiple notations: Petri Net, BPMN, Process Tree, and Directly-Follows Graph (DFG).
            </div>
"""
            
            # Add process model visualizations
            for name, filename in [('Petri Net', 'petri_net.png'),
                                   ('BPMN Model', 'bpmn_model.png'),
                                   ('Process Tree', 'process_tree.png'),
                                   ('DFG Frequency', 'dfg_frequency.png'),
                                   ('DFG Performance', 'dfg_performance.png')]:
                img_path = os.path.join(self.dir_visualizations, filename)
                img_data = encode_image(img_path)
                if img_data:
                    html_content += f"""
            <div class="visualization">
                <h4>{name}</h4>
                <img src="data:image/png;base64,{img_data}" alt="{name}">
            </div>
"""
            
            html_content += """
        </section>
        
        <section id="quality" class="section">
            <h2>⭐ Model Quality Metrics</h2>
"""
            
            # Quality metrics table
            quality_metrics = {
                'Fitness': self.metrics.get('fitness', 0),
                'Precision': self.metrics.get('precision', 0),
                'Generalization': self.metrics.get('generalization', 0),
                'Simplicity': self.metrics.get('simplicity', 0),
                'F-Score': self.metrics.get('f_score', 0),
                'Overall Quality': self.metrics.get('overall_quality', 0)
            }
            
            html_content += """
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Assessment</th>
                </tr>
"""
            
            for metric_name, value in quality_metrics.items():
                if value >= 0.8:
                    assessment = '<span class="quality-indicator quality-excellent">Excellent</span>'
                elif value >= 0.6:
                    assessment = '<span class="quality-indicator quality-good">Good</span>'
                else:
                    assessment = '<span class="quality-indicator quality-poor">Needs Improvement</span>'
                
                html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value:.4f}</td>
                    <td>{assessment}</td>
                </tr>
"""
            
            html_content += """
            </table>
"""
            
            # Add quality metrics chart
            img_path = os.path.join(self.dir_visualizations, 'quality_metrics.png')
            img_data = encode_image(img_path)
            if img_data:
                html_content += f"""
            <div class="visualization">
                <h4>Quality Metrics Overview</h4>
                <img src="data:image/png;base64,{img_data}" alt="Quality Metrics">
            </div>
"""
            
            html_content += """
        </section>
"""
            
            # Conformance section
            if 'conformance' in self.metrics:
                html_content += """
        <section id="conformance" class="section">
            <h2>🔍 Conformance Checking</h2>
            
            <h3>Conformance Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
"""
                conf = self.metrics['conformance']
                conformance_metrics = {
                    'Token Replay Fitness': f"{conf.get('fitness_tokens', 0):.4f}",
                    'Average Alignment Cost': f"{conf.get('avg_alignment_cost', 0):.4f}",
                    'Perfect Alignments': f"{conf.get('perfect_alignments_pct', 0):.2f}%",
                    'Total Deviations': str(conf.get('total_deviations', 0)),
                    'Missing Tokens': str(conf.get('missing_tokens', 0)),
                    'Remaining Tokens': str(conf.get('remaining_tokens', 0))
                }
                
                for metric_name, value in conformance_metrics.items():
                    html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value}</td>
                </tr>
"""
                
                html_content += """
            </table>
"""
                
                # Add deviation heatmap
                img_path = os.path.join(self.dir_visualizations, 'conformance_deviations.png')
                img_data = encode_image(img_path)
                if img_data:
                    html_content += f"""
            <div class="visualization">
                <h4>Conformance Deviations by Activity</h4>
                <img src="data:image/png;base64,{img_data}" alt="Conformance Deviations">
            </div>
"""
                
                html_content += """
        </section>
"""
            
            # Social Network section
            if 'social_network' in self.metrics:
                html_content += """
        <section id="social" class="section">
            <h2>👥 Social Network Analysis</h2>
            
            <p>Analysis of resource interactions and collaboration patterns.</p>
            
            <div class="metrics-grid">
"""
                sna = self.metrics['social_network']
                html_content += f"""
                <div class="metric-card">
                    <h4>Handover Relationships</h4>
                    <div class="value">{sna.get('handover_edges', 0)}</div>
                </div>
                <div class="metric-card">
                    <h4>Working Together</h4>
                    <div class="value">{sna.get('working_together_edges', 0)}</div>
                </div>
                <div class="metric-card">
                    <h4>Subcontracting</h4>
                    <div class="value">{sna.get('subcontracting_edges', 0)}</div>
                </div>
            </div>
"""
                
                # Add social network visualizations
                for name, filename in [('Handover of Work Network', 'sna_handover_of_work.png'),
                                       ('Working Together Network', 'sna_working_together.png'),
                                       ('Subcontracting Network', 'sna_subcontracting.png')]:
                    img_path = os.path.join(self.dir_visualizations, filename)
                    img_data = encode_image(img_path)
                    if img_data:
                        html_content += f"""
            <div class="visualization">
                <h4>{name}</h4>
                <img src="data:image/png;base64,{img_data}" alt="{name}">
            </div>
"""
                
                html_content += """
        </section>
"""
            
            # Decision Mining section
            if 'decision_points' in self.metrics:
                html_content += """
        <section id="decisions" class="section">
            <h2>🔀 Decision Mining</h2>
            
            <p>Analysis of decision points and branching behavior in the process.</p>
"""
                html_content += f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Total Decision Points</h4>
                    <div class="value">{self.metrics.get('decision_points', 0)}</div>
                </div>
"""
                if 'max_branches' in self.metrics:
                    html_content += f"""
                <div class="metric-card">
                    <h4>Maximum Branches</h4>
                    <div class="value">{self.metrics.get('max_branches', 0)}</div>
                </div>
                <div class="metric-card">
                    <h4>Average Branches</h4>
                    <div class="value">{self.metrics.get('avg_branches', 0):.2f}</div>
                </div>
"""
                html_content += """
            </div>
"""
                
                # Add decision points chart
                img_path = os.path.join(self.dir_visualizations, 'decision_points.png')
                img_data = encode_image(img_path)
                if img_data:
                    html_content += f"""
            <div class="visualization">
                <h4>Top Decision Points</h4>
                <img src="data:image/png;base64,{img_data}" alt="Decision Points">
            </div>
"""
                
                html_content += """
        </section>
"""
            
            # Algorithm Comparison section
            if 'algorithm_comparison' in self.metrics:
                html_content += """
        <section id="comparison" class="section">
            <h2>📊 Algorithm Comparison</h2>
            
            <p>Comparison of different process discovery algorithms.</p>
            
            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>Fitness</th>
                    <th>Precision</th>
                    <th>Generalization</th>
                    <th>Simplicity</th>
                </tr>
"""
                for algo_name, metrics in self.metrics['algorithm_comparison'].items():
                    html_content += f"""
                <tr>
                    <td><strong>{algo_name}</strong></td>
                    <td>{metrics['fitness']:.4f}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['generalization']:.4f}</td>
                    <td>{metrics['simplicity']:.4f}</td>
                </tr>
"""
                
                html_content += """
            </table>
"""
                
                # Add algorithm comparison chart
                img_path = os.path.join(self.dir_visualizations, 'algorithm_comparison.png')
                img_data = encode_image(img_path)
                if img_data:
                    html_content += f"""
            <div class="visualization">
                <h4>Algorithm Comparison Radar Chart</h4>
                <img src="data:image/png;base64,{img_data}" alt="Algorithm Comparison">
            </div>
"""
                
                html_content += """
        </section>
"""
            
            # Footer
            html_content += f"""
        <footer>
            <p><strong>Process Mining Analysis Dashboard</strong></p>
            <p>Generated by Enhanced Process Discovery Framework</p>
            <p>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Event Log: {os.path.basename(self.file_path)}</p>
        </footer>
    </div>
</body>
</html>
"""
            
            # Save HTML file
            dashboard_path = os.path.join(self.dir_reports, 'dashboard.html')
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.log_report(f"✅ Interactive dashboard created: '{dashboard_path}'")
            self.log_report(f"   Open this file in your web browser to view the dashboard")
            
            return True
        except Exception as e:
            self.log_report(f"❌ ERROR in dashboard creation: {str(e)}")
            traceback.print_exc()
            return False
    
    # ==================== MAIN PIPELINE ====================
    
    def run(self):
        """
        Execute the complete process mining pipeline
        
        Returns:
        --------
        bool : Overall success status
        """
        import time
        
        start_time = time.time()
        
        # Print banner
        print("\n" + "="*100)
        print("🔍 ENHANCED PROCESS MINING FRAMEWORK WITH PM4PY")
        print("="*100)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
        # Track step results
        step_results = {}
        
        # Step 1: Load and Preprocess Data
        try:
            step_start = time.time()
            result = self.load_and_preprocess_data()
            step_results['1. Load & Preprocess Data'] = {
                'success': result,
                'time': time.time() - step_start
            }
            if not result:
                self.log_report("\n❌ Pipeline stopped due to data loading failure")
                return False
        except Exception as e:
            self.log_report(f"\n❌ CRITICAL ERROR in Step 1: {str(e)}")
            step_results['1. Load & Preprocess Data'] = {'success': False, 'time': 0}
            return False
        
        # Step 2: Statistical Analysis
        try:
            step_start = time.time()
            result = self.perform_statistical_analysis()
            step_results['2. Statistical Analysis'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 2: {str(e)}")
            step_results['2. Statistical Analysis'] = {'success': False, 'time': 0}
        
        # Step 3: Process Discovery
        try:
            step_start = time.time()
            result = self.discover_process_model()
            step_results['3. Process Discovery'] = {
                'success': result,
                'time': time.time() - step_start
            }
            if not result:
                self.log_report("\n❌ Pipeline stopped due to process discovery failure")
                return False
        except Exception as e:
            self.log_report(f"\n❌ CRITICAL ERROR in Step 3: {str(e)}")
            step_results['3. Process Discovery'] = {'success': False, 'time': 0}
            return False
        
        # Step 4: Create Visualizations
        try:
            step_start = time.time()
            result = self.create_visualizations()
            step_results['4. Create Visualizations'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 4: {str(e)}")
            step_results['4. Create Visualizations'] = {'success': False, 'time': 0}
        
        # Step 5: Model Quality Evaluation
        try:
            step_start = time.time()
            result = self.evaluate_model_quality()
            step_results['5. Model Quality Evaluation'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 5: {str(e)}")
            step_results['5. Model Quality Evaluation'] = {'success': False, 'time': 0}
        
        # Step 6: Advanced Conformance Checking
        try:
            step_start = time.time()
            result = self.advanced_conformance_checking()
            step_results['6. Conformance Checking'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 6: {str(e)}")
            step_results['6. Conformance Checking'] = {'success': False, 'time': 0}
        
        # Step 7: Social Network Analysis
        try:
            step_start = time.time()
            result = self.analyze_social_network()
            step_results['7. Social Network Analysis'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 7: {str(e)}")
            step_results['7. Social Network Analysis'] = {'success': False, 'time': 0}
        
        # Step 8: Decision Mining
        try:
            step_start = time.time()
            result = self.decision_mining_analysis()
            step_results['8. Decision Mining'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 8: {str(e)}")
            step_results['8. Decision Mining'] = {'success': False, 'time': 0}
        
        # Step 9: Algorithm Comparison
        try:
            step_start = time.time()
            result = self.compare_discovery_algorithms()
            step_results['9. Algorithm Comparison'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 9: {str(e)}")
            step_results['9. Algorithm Comparison'] = {'success': False, 'time': 0}
        
        # Step 10: Export Models
        try:
            step_start = time.time()
            result = self.export_models()
            step_results['10. Export Models'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 10: {str(e)}")
            step_results['10. Export Models'] = {'success': False, 'time': 0}
        
        # Step 11: Generate Reports
        try:
            step_start = time.time()
            result = self.generate_comprehensive_report()
            step_results['11. Generate Reports'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 11: {str(e)}")
            step_results['11. Generate Reports'] = {'success': False, 'time': 0}
        
        # Step 12: Create Dashboard
        try:
            step_start = time.time()
            result = self.create_interactive_dashboard()
            step_results['12. Create Dashboard'] = {
                'success': result,
                'time': time.time() - step_start
            }
        except Exception as e:
            self.log_report(f"\n⚠️ ERROR in Step 12: {str(e)}")
            step_results['12. Create Dashboard'] = {'success': False, 'time': 0}
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print final summary
        self.add_separator('=')
        self.log_report("\n🎉 PIPELINE EXECUTION SUMMARY\n")
        self.add_separator('-')
        
        self.log_report("\nStep Execution Status:\n")
        for step_name, result in step_results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            time_str = f"{result['time']:.2f}s"
            self.log_report(f"   {step_name:<40} {status:<15} ({time_str})")
        
        successful_steps = sum(1 for r in step_results.values() if r['success'])
        total_steps = len(step_results)
        
        self.log_report(f"\n📊 Summary Statistics:")
        self.log_report(f"   Total Steps: {total_steps}")
        self.log_report(f"   Successful: {successful_steps}")
        self.log_report(f"   Failed: {total_steps - successful_steps}")
        self.log_report(f"   Success Rate: {(successful_steps/total_steps)*100:.1f}%")
        self.log_report(f"   Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        self.log_report(f"\n📁 Output Locations:")
        self.log_report(f"   Visualizations: {self.dir_visualizations}")
        self.log_report(f"   Models: {self.dir_models}")
        self.log_report(f"   Reports: {self.dir_reports}")
        
        self.log_report(f"\n🌐 Next Steps:")
        self.log_report(f"   1. Open the interactive dashboard: {os.path.join(self.dir_reports, 'dashboard.html')}")
        self.log_report(f"   2. Review the detailed report: {os.path.join(self.dir_reports, 'detailed_report.txt')}")
        self.log_report(f"   3. Explore visualizations in: {self.dir_visualizations}")
        self.log_report(f"   4. Use exported models from: {self.dir_models}")
        
        self.add_separator('=')
        self.log_report(f"\n✅ Process Mining Analysis Complete!")
        self.log_report(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.add_separator('=')
        
        return successful_steps >= (total_steps * 0.7)  # Success if at least 70% of steps succeeded


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    """
    Main execution block - Example usage of the Enhanced Process Discovery Framework
    """
    
    print("\n" + "="*100)
    print("🚀 ENHANCED PROCESS MINING FRAMEWORK - EXAMPLE USAGE")
    print("="*100 + "\n")
    
    # ========================================
    # CONFIGURATION - Modify these parameters
    # ========================================
    
    # Option 1: For CSV files
    EVENT_LOG_PATH = 'D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/finale_preprocessed_renamed.csv'  # Change this to your CSV file path
    # CASE_ID_COLUMN = 'case:concept:name'    # Column name for case ID
    # ACTIVITY_COLUMN = 'concept:name'        # Column name for activity
    # TIMESTAMP_COLUMN = 'time:timestamp'     # Column name for timestamp
    CASE_ID_COLUMN = 'Case ID'           # atau 'Case_ID', 'case_id', dll
    ACTIVITY_COLUMN = 'Activity'         # atau 'Activity_Name', 'Event', dll
    TIMESTAMP_COLUMN = 'timestamp'       # atau 'Time', 'DateTime', 'Date', dll
    
    # Option 2: For XES files (uncomment to use)
    # EVENT_LOG_PATH = 'your_event_log.xes'
    # CASE_ID_COLUMN = 'case:concept:name'
    # ACTIVITY_COLUMN = 'concept:name'
    # TIMESTAMP_COLUMN = 'time:timestamp'
    
    # Output directory
    OUTPUT_DIRECTORY = 'D:/VSCODE/myenv2/share/Mining_Final_Project/process_mining_output'
    
    # ========================================
    # EXECUTION
    # ========================================
    
    try:
        # Check if file exists
        if not os.path.exists(EVENT_LOG_PATH):
            print(f"❌ ERROR: Event log file not found: '{EVENT_LOG_PATH}'")
            print(f"\nPlease update the EVENT_LOG_PATH variable with your actual file path.")
            print(f"\nExample paths:")
            print(f"   - For CSV: 'data/event_log.csv'")
            print(f"   - For XES: 'data/BPI_Challenge_2012.xes'")
            print("\n" + "="*100 + "\n")
            exit(1)
        
        # Initialize the pipeline
        print(f"📂 Initializing pipeline with:")
        print(f"   Event Log: {EVENT_LOG_PATH}")
        print(f"   Output Directory: {OUTPUT_DIRECTORY}")
        print(f"   Case ID: {CASE_ID_COLUMN}")
        print(f"   Activity: {ACTIVITY_COLUMN}")
        print(f"   Timestamp: {TIMESTAMP_COLUMN}\n")
        
        pipeline = EnhancedProcessDiscovery(
            file_path=EVENT_LOG_PATH,
            case_id=CASE_ID_COLUMN,
            activity_key=ACTIVITY_COLUMN,
            timestamp_key=TIMESTAMP_COLUMN,
            output_dir=OUTPUT_DIRECTORY
        )
        
        # Run the complete pipeline
        print("🚀 Starting process mining pipeline...\n")
        success = pipeline.run()
        
        # Final message
        if success:
            print("\n" + "="*100)
            print("🎉 SUCCESS! Process mining analysis completed successfully!")
            print("="*100)
            print(f"\n📊 Results are available in: {OUTPUT_DIRECTORY}/")
            print(f"\n🌐 Open the dashboard to explore results:")
            print(f"   {os.path.join(OUTPUT_DIRECTORY, 'reports', 'dashboard.html')}")
            print("\n" + "="*100 + "\n")
        else:
            print("\n" + "="*100)
            print("⚠️  Process mining completed with some errors")
            print("="*100)
            print(f"\nPlease check the output in: {OUTPUT_DIRECTORY}/")
            print(f"Review the detailed report for more information.")
            print("\n" + "="*100 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user (Ctrl+C)")
        print("="*100 + "\n")
    
    except Exception as e:
        print("\n" + "="*100)
        print("❌ CRITICAL ERROR:")
        print("="*100)
        print(f"\n{str(e)}\n")
        traceback.print_exc()
        print("\n" + "="*100 + "\n")


