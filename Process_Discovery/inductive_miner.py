"""
Process Discovery using Inductive Miner - PM4PY Framework
Comprehensive script untuk menemukan process model dari event log
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
try:
    import pm4py
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
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
    from pm4py.statistics.traces.generic.log import case_statistics
    from pm4py.algo.filtering.log.variants import variants_filter
    from pm4py.algo.filtering.log.attributes import attributes_filter
    # from pm4py.statistics.sojourn_time.log import get as soj_time_get
    print("✅ PM4PY berhasil di-import")
except ImportError:
    print("❌ ERROR: PM4PY tidak terinstall!")
    print("Install dengan: pip install pm4py")
    exit(1)

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


class ProcessDiscovery:
    """
    Class untuk Process Discovery menggunakan PM4PY
    """
    
    def __init__(self, input_file='finale_preprocessed.csv', 
                 output_dir='process_discovery_output'):
        """
        Inisialisasi Process Discovery
        
        Args:
            input_file: File CSV hasil preprocessing
            output_dir: Directory untuk menyimpan output
        """
        self.input_file = input_file
        self.output_dir = output_dir
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
        
        # Buat output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Directory '{output_dir}' berhasil dibuat")
    
    def log_report(self, message, print_msg=True):
        """Catat pesan ke report"""
        self.report.append(message)
        if print_msg:
            print(message)
    
    def add_separator(self):
        """Tambahkan separator"""
        separator = "=" * 80
        self.log_report(separator)
    
    # ========== 1. LOAD DATA ==========
    def load_preprocessed_data(self):
        """
        Load data hasil preprocessing
        """
        self.add_separator()
        self.log_report("\n📂 LOADING PREPROCESSED DATA\n")
        
        try:
            self.df = pd.read_csv(self.input_file)
            self.log_report(f"✅ Data berhasil dimuat dari '{self.input_file}'")
            self.log_report(f"   Jumlah baris: {len(self.df):,}")
            self.log_report(f"   Jumlah kolom: {len(self.df.columns)}")
            self.log_report(f"   Kolom: {list(self.df.columns)}")
            
            return True
            
        except FileNotFoundError:
            self.log_report(f"❌ ERROR: File '{self.input_file}' tidak ditemukan!")
            return False
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    # ========== 2. CONVERT TO EVENT LOG ==========
    def convert_to_event_log(self):
        """
        Convert pandas DataFrame ke PM4PY event log format
        """
        self.add_separator()
        self.log_report("\n🔄 CONVERTING TO EVENT LOG FORMAT\n")
        
        try:
            # Deteksi kolom penting
            case_col = self._detect_column(['case_id', 'caseid', 'case', 'case id'])
            activity_col = self._detect_column(['activity', 'activity_name', 'event', 'task'])
            timestamp_col = self._detect_column(['timestamp', 'time', 'datetime', 'date'])
            
            if not all([case_col, activity_col, timestamp_col]):
                self.log_report("❌ ERROR: Tidak dapat mendeteksi kolom penting!")
                self.log_report(f"   Case ID: {case_col}")
                self.log_report(f"   Activity: {activity_col}")
                self.log_report(f"   Timestamp: {timestamp_col}")
                return False
            
            self.log_report("✅ Kolom terdeteksi:")
            self.log_report(f"   Case ID: {case_col}")
            self.log_report(f"   Activity: {activity_col}")
            self.log_report(f"   Timestamp: {timestamp_col}")
            
            # Rename kolom sesuai PM4PY standard
            df_log = self.df.copy()
            df_log = df_log.rename(columns={
                case_col: 'case:concept:name',
                activity_col: 'concept:name',
                timestamp_col: 'time:timestamp'
            })
            
            # Pastikan timestamp dalam format datetime
            df_log['time:timestamp'] = pd.to_datetime(df_log['time:timestamp'])
            
            # Convert ke event log PM4PY
            self.event_log = log_converter.apply(
                df_log, 
                variant=log_converter.Variants.TO_EVENT_LOG
            )
            
            self.log_report(f"\n✅ Conversion berhasil!")
            self.log_report(f"   Total cases: {len(self.event_log):,}")
            self.log_report(f"   Total events: {sum([len(trace) for trace in self.event_log]):,}")
            
            # Simpan info untuk metrics
            self.metrics['total_cases'] = len(self.event_log)
            self.metrics['total_events'] = sum([len(trace) for trace in self.event_log])
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR saat convert: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_column(self, possible_names):
        """
        Deteksi nama kolom (case-insensitive)
        """
        col_lower = {col.lower(): col for col in self.df.columns}
        for name in possible_names:
            if name.lower() in col_lower:
                return col_lower[name.lower()]
        return None
    
    # ========== 3. DISCOVER PROCESS MODEL ==========
    def discover_process_model(self, algorithm='inductive'):
        """
        Discover process model menggunakan berbagai algoritma
        
        Args:
            algorithm: 'inductive', 'heuristic', atau 'alpha'
        """
        self.add_separator()
        self.log_report(f"\n🔍 PROCESS DISCOVERY - {algorithm.upper()} MINER\n")
        
        try:
            if algorithm == 'inductive':
                self.log_report("Menjalankan Inductive Miner...")
                # Inductive Miner - paling robust dan handling noise
                self.process_tree = inductive_miner.apply(self.event_log)
                
                # Convert process tree ke Petri net
                self.process_model, self.initial_marking, self.final_marking = \
                    pt_converter.apply(self.process_tree, variant=pt_converter.Variants.TO_PETRI_NET)
                
                self.log_report("✅ Inductive Miner berhasil!")
                self.log_report(f"   Places: {len(self.process_model.places)}")
                self.log_report(f"   Transitions: {len(self.process_model.transitions)}")
                self.log_report(f"   Arcs: {len(self.process_model.arcs)}")
                
            elif algorithm == 'heuristic':
                self.log_report("Menjalankan Heuristic Miner...")
                # Heuristic Miner - good for discovering main process flow
                self.process_model, self.initial_marking, self.final_marking = \
                    heuristics_miner.apply_heu(self.event_log)
                
                self.log_report("✅ Heuristic Miner berhasil!")
                
            elif algorithm == 'alpha':
                self.log_report("Menjalankan Alpha Miner...")
                # Alpha Miner - classic algorithm
                self.process_model, self.initial_marking, self.final_marking = \
                    alpha_miner.apply(self.event_log)
                
                self.log_report("✅ Alpha Miner berhasil!")
            
            else:
                self.log_report(f"❌ ERROR: Algorithm '{algorithm}' tidak dikenali")
                return False
            
            self.metrics['discovery_algorithm'] = algorithm
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR saat discovery: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========== 4. VISUALIZATIONS ==========
    def visualize_petri_net(self, with_frequency=True):
        """
        Visualisasi Petri Net
        """
        self.add_separator()
        self.log_report("\n🎨 VISUALIZING PETRI NET\n")
        
        try:
            output_path = os.path.join(self.output_dir, 'petri_net.png')
            
            if with_frequency:
                # Visualisasi dengan frequency annotations
                self.log_report("Membuat Petri Net dengan frequency annotations...")
                parameters = {
                    pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"
                }
                gviz = pn_visualizer.apply(
                    self.process_model, 
                    self.initial_marking, 
                    self.final_marking,
                    parameters=parameters,
                    variant=pn_visualizer.Variants.FREQUENCY,
                    log=self.event_log
                )
            else:
                # Visualisasi standard
                self.log_report("Membuat Petri Net standard...")
                parameters = {
                    pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"
                }
                gviz = pn_visualizer.apply(
                    self.process_model, 
                    self.initial_marking, 
                    self.final_marking,
                    parameters=parameters
                )
            
            # Save visualization
            pn_visualizer.save(gviz, output_path)
            self.log_report(f"✅ Petri Net disimpan: '{output_path}'")
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def visualize_process_tree(self):
        """
        Visualisasi Process Tree
        """
        self.add_separator()
        self.log_report("\n🌳 VISUALIZING PROCESS TREE\n")
        
        try:
            if self.process_tree is None:
                self.log_report("⚠️  Process tree tidak tersedia (hanya untuk Inductive Miner)")
                return False
            
            output_path = os.path.join(self.output_dir, 'process_tree.png')
            
            self.log_report("Membuat Process Tree visualization...")
            parameters = {
                pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"
            }
            gviz = pt_visualizer.apply(self.process_tree, parameters=parameters)
            
            # Save
            pt_visualizer.save(gviz, output_path)
            self.log_report(f"✅ Process Tree disimpan: '{output_path}'")
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def visualize_bpmn(self):
        """
        Visualisasi BPMN diagram
        """
        self.add_separator()
        self.log_report("\n📊 VISUALIZING BPMN DIAGRAM\n")
        
        try:
            output_path = os.path.join(self.output_dir, 'bpmn_diagram.png')
            
            self.log_report("Converting Petri Net ke BPMN...")
            # Convert Petri net ke BPMN
            self.bpmn_model = bpmn_converter.apply(
                self.process_model, 
                self.initial_marking, 
                self.final_marking
            )
            
            self.log_report("Membuat BPMN visualization...")
            parameters = {
                bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "png"
            }
            gviz = bpmn_visualizer.apply(self.bpmn_model, parameters=parameters)
            
            # Save
            bpmn_visualizer.save(gviz, output_path)
            self.log_report(f"✅ BPMN Diagram disimpan: '{output_path}'")
            
            return True
            
        except Exception as e:
            self.log_report(f"⚠️  BPMN conversion error: {str(e)}")
            return False
    
    def discover_dfg(self):
        """
        Discover Directly-Follows Graph (DFG)
        """
        self.add_separator()
        self.log_report("\n🔗 DISCOVERING DIRECTLY-FOLLOWS GRAPH (DFG)\n")
        
        try:
            # Discover DFG dengan frequency
            self.log_report("Menghitung DFG dengan frequency...")
            self.dfg = dfg_discovery.apply(self.event_log, variant=dfg_discovery.Variants.FREQUENCY)
            
            # Get start dan end activities
            start_activities = pm4py.get_start_activities(self.event_log)
            end_activities = pm4py.get_end_activities(self.event_log)
            
            self.log_report(f"✅ DFG berhasil di-discover!")
            self.log_report(f"   Total edges: {len(self.dfg)}")
            self.log_report(f"   Start activities: {len(start_activities)}")
            self.log_report(f"   End activities: {len(end_activities)}")
            
            # Simpan info
            self.metrics['dfg_edges'] = len(self.dfg)
            self.metrics['start_activities'] = start_activities
            self.metrics['end_activities'] = end_activities
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    def visualize_dfg(self):
        """
        Visualisasi DFG dengan frequency dan performance
        """
        self.add_separator()
        self.log_report("\n🎨 VISUALIZING DFG\n")
        
        try:
            # DFG Frequency
            self.log_report("1. Membuat DFG Frequency visualization...")
            output_freq = os.path.join(self.output_dir, 'dfg_frequency.png')
            
            parameters_freq = {
                dfg_visualization.Variants.FREQUENCY.value.Parameters.FORMAT: "png"
            }
            
            gviz_freq = dfg_visualization.apply(
                self.dfg,
                log=self.event_log,
                variant=dfg_visualization.Variants.FREQUENCY,
                parameters=parameters_freq
            )
            dfg_visualization.save(gviz_freq, output_freq)
            self.log_report(f"   ✅ DFG Frequency: '{output_freq}'")
            
            # DFG Performance
            self.log_report("\n2. Membuat DFG Performance visualization...")
            output_perf = os.path.join(self.output_dir, 'dfg_performance.png')
            
            # Calculate performance DFG
            dfg_performance = dfg_discovery.apply(
                self.event_log, 
                variant=dfg_discovery.Variants.PERFORMANCE
            )
            
            parameters_perf = {
                dfg_visualization.Variants.PERFORMANCE.value.Parameters.FORMAT: "png"
            }
            
            gviz_perf = dfg_visualization.apply(
                dfg_performance,
                log=self.event_log,
                variant=dfg_visualization.Variants.PERFORMANCE,
                parameters=parameters_perf
            )
            dfg_visualization.save(gviz_perf, output_perf)
            self.log_report(f"   ✅ DFG Performance: '{output_perf}'")
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========== 5. MODEL QUALITY METRICS ==========
    def calculate_model_quality(self):
        """
        Hitung quality metrics: fitness, precision, generalization, simplicity
        """
        self.add_separator()
        self.log_report("\n📏 CALCULATING MODEL QUALITY METRICS\n")
        
        try:
            # 1. Fitness
            self.log_report("1. Menghitung Fitness...")
            fitness = replay_fitness.apply(
                self.event_log, 
                self.process_model, 
                self.initial_marking, 
                self.final_marking,
                variant=replay_fitness.Variants.TOKEN_BASED
            )
            fitness_value = fitness['average_trace_fitness']
            self.log_report(f"   ✅ Fitness: {fitness_value:.4f}")
            self.log_report(f"      (Seberapa baik model mereproduksi log)")
            
            # 2. Precision
            self.log_report("\n2. Menghitung Precision...")
            precision = precision_evaluator.apply(
                self.event_log, 
                self.process_model, 
                self.initial_marking, 
                self.final_marking,
                variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
            )
            self.log_report(f"   ✅ Precision: {precision:.4f}")
            self.log_report(f"      (Seberapa tepat model, tidak overgeneralize)")
            
            # 3. Generalization
            self.log_report("\n3. Menghitung Generalization...")
            generalization = generalization_evaluator.apply(
                self.event_log, 
                self.process_model, 
                self.initial_marking, 
                self.final_marking
            )
            self.log_report(f"   ✅ Generalization: {generalization:.4f}")
            self.log_report(f"      (Kemampuan model generalize ke unseen cases)")
            
            # 4. Simplicity
            self.log_report("\n4. Menghitung Simplicity...")
            # Simplicity = inverse complexity (jumlah nodes dan arcs)
            num_places = len(self.process_model.places)
            num_transitions = len(self.process_model.transitions)
            num_arcs = len(self.process_model.arcs)
            complexity = num_places + num_transitions + num_arcs
            simplicity = 1 / (1 + np.log(complexity)) if complexity > 0 else 1
            
            self.log_report(f"   ✅ Simplicity: {simplicity:.4f}")
            self.log_report(f"      (Kesederhanaan model: places={num_places}, transitions={num_transitions}, arcs={num_arcs})")
            
            # Simpan metrics
            self.metrics['fitness'] = float(fitness_value)
            self.metrics['precision'] = float(precision)
            self.metrics['generalization'] = float(generalization)
            self.metrics['simplicity'] = float(simplicity)
            
            # Overall quality score (weighted average)
            overall_quality = (fitness_value * 0.4 + precision * 0.3 + 
                             generalization * 0.2 + simplicity * 0.1)
            self.metrics['overall_quality'] = float(overall_quality)
            
            self.log_report(f"\n📊 OVERALL QUALITY SCORE: {overall_quality:.4f}")
            self.log_report(self._interpret_quality_score(overall_quality))
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR saat calculate metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _interpret_quality_score(self, score):
        """Interpretasi quality score"""
        if score >= 0.8:
            return "   🌟 Excellent! Model sangat berkualitas tinggi"
        elif score >= 0.7:
            return "   ✅ Good! Model berkualitas baik"
        elif score >= 0.6:
            return "   ⚠️  Fair. Model cukup, mungkin perlu tuning"
        else:
            return "   ❌ Poor. Model perlu improvement signifikan"
    
    # ========== 6. VARIANT ANALYSIS ==========
    def discover_variants(self, top_n=10):
        """
        Analisis process variants
        """
        self.add_separator()
        self.log_report(f"\n🔀 VARIANT ANALYSIS (Top {top_n})\n")
        
        try:
            # Get variants
            variants = variants_get.get_variants(self.event_log)
            
            # Sort by frequency
            sorted_variants = sorted(
                variants.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            
            total_variants = len(sorted_variants)
            total_cases = len(self.event_log)
            
            self.log_report(f"Total unique variants: {total_variants:,}")
            self.log_report(f"Total cases: {total_cases:,}")
            
            # Coverage analysis
            cumulative_coverage = 0
            variants_for_80_pct = 0
            
            self.log_report(f"\nTop {top_n} Most Frequent Variants:\n")
            
            for idx, (variant, traces) in enumerate(sorted_variants[:top_n], 1):
                freq = len(traces)
                pct = (freq / total_cases) * 100
                cumulative_coverage += pct
                
                # Hitung untuk 80% coverage
                if cumulative_coverage <= 80:
                    variants_for_80_pct = idx
                
                # Format variant
                variant_str = ' → '.join(variant)
                if len(variant_str) > 100:
                    variant_str = variant_str[:100] + "..."
                
                self.log_report(f"{idx}. [{freq} cases, {pct:.2f}%, cumulative: {cumulative_coverage:.2f}%]")
                self.log_report(f"   {variant_str}\n")
            
            self.log_report(f"📊 Insight:")
            self.log_report(f"   - {variants_for_80_pct} variants mencakup ~80% dari semua cases")
            self.log_report(f"   - Variant ratio: {total_variants/total_cases:.2%}")
            
            if total_variants / total_cases > 0.7:
                self.log_report(f"   ⚠️  Process sangat bervariasi (low standardization)")
            elif total_variants / total_cases < 0.3:
                self.log_report(f"   ✅ Process cukup terstandarisasi")
            
            # Simpan metrics
            self.metrics['total_variants'] = total_variants
            self.metrics['variant_ratio'] = total_variants / total_cases
            self.metrics['variants_for_80pct'] = variants_for_80_pct
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            return False
    
    # ========== 7. BOTTLENECK ANALYSIS ==========
    def analyze_bottlenecks(self):
        """
        Identifikasi bottlenecks dalam proses
        """
        self.add_separator()
        self.log_report("\n⏱️  BOTTLENECK ANALYSIS\n")
        
        try:
            # Calculate sojourn time per activity
            self.log_report("Menghitung sojourn time per activity...")
            
            # Get all activities
            activities = pm4py.get_event_attribute_values(self.event_log, "concept:name")
            
            # Calculate statistics per activity
            activity_stats = []
            
            for activity in activities:
                # Filter log untuk activity ini
                filtered_log = attributes_filter.apply_events(
                    self.event_log,
                    [activity],
                    parameters={
                        attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name",
                        attributes_filter.Parameters.POSITIVE: True
                    }
                )
                
                if len(filtered_log) > 0:
                    # Hitung case durations untuk cases dengan activity ini
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
            
            # Sort by mean duration
            activity_stats_sorted = sorted(
                activity_stats, 
                key=lambda x: x['mean_duration'], 
                reverse=True
            )
            
            self.log_report("✅ Top 10 Activities dengan durasi terlama:\n")
            
            for idx, stat in enumerate(activity_stats_sorted[:10], 1):
                self.log_report(f"{idx}. {stat['activity']}")
                self.log_report(f"   Frequency: {stat['frequency']}")
                self.log_report(f"   Mean Duration: {stat['mean_duration']/3600:.2f} hours")
                self.log_report(f"   Median Duration: {stat['median_duration']/3600:.2f} hours")
                self.log_report(f"   Max Duration: {stat['max_duration']/3600:.2f} hours\n")
            
            # Identifikasi bottlenecks (activities dengan durasi > 2 * median)
            all_durations = [s['mean_duration'] for s in activity_stats]
            overall_median = np.median(all_durations)
            bottlenecks = [s for s in activity_stats if s['mean_duration'] > 2 * overall_median]
            
            self.log_report(f"\n🚨 BOTTLENECKS DETECTED: {len(bottlenecks)} activities")
            self.log_report(f"   (Activities dengan durasi > 2x median)")
            
            for b in bottlenecks:
                self.log_report(f"   - {b['activity']}: {b['mean_duration']/3600:.2f} hours")
            
            self.metrics['bottlenecks'] = [b['activity'] for b in bottlenecks]
            self.metrics['bottleneck_count'] = len(bottlenecks)
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========== 8. EXPORT RESULTS ==========
    def export_results(self):
        """
        Export semua hasil
        """
        self.add_separator()
        self.log_report("\n💾 EXPORTING RESULTS\n")
        
        try:
            # 1. Export Petri Net ke PNML
            self.log_report("1. Exporting Petri Net (PNML)...")
            pnml_path = os.path.join(self.output_dir, 'petri_net.pnml')
            pm4py.write_pnml(
                self.process_model, 
                self.initial_marking, 
                self.final_marking, 
                pnml_path
            )
            self.log_report(f"   ✅ PNML saved: '{pnml_path}'")
            
            # 2. Export BPMN (jika ada)
            if self.bpmn_model:
                self.log_report("\n2. Exporting BPMN (XML)...")
                bpmn_path = os.path.join(self.output_dir, 'bpmn_model.bpmn')
                pm4py.write_bpmn(self.bpmn_model, bpmn_path)
                self.log_report(f"   ✅ BPMN saved: '{bpmn_path}'")
            
            # 3. Export Metrics ke JSON
            self.log_report("\n3. Exporting Metrics (JSON)...")
            metrics_path = os.path.join(self.output_dir, 'discovery_metrics.json')
            
            # Convert metrics untuk JSON serialization
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
            
            self.log_report(f"   ✅ Metrics saved: '{metrics_path}'")
            
            # 4. Export Event Log Statistics
            self.log_report("\n4. Exporting Event Log Statistics...")
            stats_path = os.path.join(self.output_dir, 'event_log_statistics.txt')
            
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
            
            self.log_report(f"   ✅ Statistics saved: '{stats_path}'")
            
            # 5. List semua output files
            self.log_report("\n📂 OUTPUT FILES:")
            output_files = [
                'petri_net.png',
                'petri_net.pnml',
                'process_tree.png',
                'bpmn_diagram.png',
                'bpmn_model.bpmn',
                'dfg_frequency.png',
                'dfg_performance.png',
                'discovery_metrics.json',
                'event_log_statistics.txt',
                'discovery_report.txt'
            ]
            
            for file in output_files:
                file_path = os.path.join(self.output_dir, file)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path) / 1024  # KB
                    self.log_report(f"   ✅ {file} ({size:.2f} KB)")
                else:
                    self.log_report(f"   ⚠️  {file} (not created)")
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR saat export: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========== 9. GENERATE DISCOVERY REPORT ==========
    def generate_discovery_report(self):
        """
        Generate comprehensive discovery report
        """
        self.add_separator()
        self.log_report("\n📋 GENERATING DISCOVERY REPORT\n")
        
        try:
            report_path = os.path.join(self.output_dir, 'discovery_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write("PROCESS DISCOVERY REPORT\n")
                f.write("PM4PY Framework - Inductive Miner\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input File: {self.input_file}\n")
                f.write(f"Output Directory: {self.output_dir}\n\n")
                
                # Summary
                f.write("=" * 80 + "\n")
                f.write("SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total Cases: {self.metrics.get('total_cases', 0):,}\n")
                f.write(f"Total Events: {self.metrics.get('total_events', 0):,}\n")
                f.write(f"Total Variants: {self.metrics.get('total_variants', 0):,}\n")
                f.write(f"Variant Ratio: {self.metrics.get('variant_ratio', 0):.2%}\n")
                f.write(f"Discovery Algorithm: {self.metrics.get('discovery_algorithm', 'N/A').upper()}\n\n")
                
                # Model Information
                f.write("=" * 80 + "\n")
                f.write("MODEL INFORMATION\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"DFG Edges: {self.metrics.get('dfg_edges', 0)}\n")
                f.write(f"Start Activities: {len(self.metrics.get('start_activities', {}))}\n")
                f.write(f"End Activities: {len(self.metrics.get('end_activities', {}))}\n\n")
                
                # Quality Metrics
                f.write("=" * 80 + "\n")
                f.write("QUALITY METRICS\n")
                f.write("=" * 80 + "\n\n")
                fitness = self.metrics.get('fitness', 0)
                precision = self.metrics.get('precision', 0)
                generalization = self.metrics.get('generalization', 0)
                simplicity = self.metrics.get('simplicity', 0)
                overall = self.metrics.get('overall_quality', 0)
                
                f.write(f"Fitness:         {fitness:.4f}  {self._get_metric_bar(fitness)}\n")
                f.write(f"                 Seberapa baik model mereproduksi event log\n\n")
                f.write(f"Precision:       {precision:.4f}  {self._get_metric_bar(precision)}\n")
                f.write(f"                 Seberapa tepat model (tidak overgeneralize)\n\n")
                f.write(f"Generalization:  {generalization:.4f}  {self._get_metric_bar(generalization)}\n")
                f.write(f"                 Kemampuan model handle unseen cases\n\n")
                f.write(f"Simplicity:      {simplicity:.4f}  {self._get_metric_bar(simplicity)}\n")
                f.write(f"                 Kesederhanaan struktur model\n\n")
                f.write(f"Overall Quality: {overall:.4f}  {self._get_metric_bar(overall)}\n")
                f.write(f"                 {self._interpret_quality_score(overall)}\n\n")
                
                # Variant Analysis
                f.write("=" * 80 + "\n")
                f.write("VARIANT ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total Unique Variants: {self.metrics.get('total_variants', 0):,}\n")
                f.write(f"Variants for 80% Coverage: {self.metrics.get('variants_for_80pct', 0)}\n")
                f.write(f"Variant Ratio: {self.metrics.get('variant_ratio', 0):.2%}\n\n")
                
                variant_ratio = self.metrics.get('variant_ratio', 0)
                if variant_ratio > 0.7:
                    f.write("⚠️  INSIGHT: Process sangat bervariasi (low standardization)\n")
                    f.write("   Rekomendasi: Pertimbangkan untuk standardisasi proses\n\n")
                elif variant_ratio < 0.3:
                    f.write("✅ INSIGHT: Process cukup terstandarisasi\n\n")
                else:
                    f.write("✅ INSIGHT: Process memiliki variasi yang wajar\n\n")
                
                # Bottleneck Analysis
                f.write("=" * 80 + "\n")
                f.write("BOTTLENECK ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                bottlenecks = self.metrics.get('bottlenecks', [])
                f.write(f"Bottlenecks Detected: {len(bottlenecks)}\n\n")
                
                if bottlenecks:
                    f.write("Activities dengan durasi > 2x median:\n")
                    for idx, activity in enumerate(bottlenecks, 1):
                        f.write(f"  {idx}. {activity}\n")
                    f.write("\n⚠️  Rekomendasi: Investigate bottleneck activities untuk optimization\n\n")
                else:
                    f.write("✅ Tidak ada bottleneck signifikan terdeteksi\n\n")
                
                # Recommendations
                f.write("=" * 80 + "\n")
                f.write("RECOMMENDATIONS\n")
                f.write("=" * 80 + "\n\n")
                
                recommendations = []
                
                # Quality-based recommendations
                if fitness < 0.7:
                    recommendations.append("⚠️  Fitness rendah: Model tidak mereproduksi log dengan baik. "
                                         "Pertimbangkan algoritma discovery lain atau parameter tuning.")
                
                if precision < 0.7:
                    recommendations.append("⚠️  Precision rendah: Model terlalu general. "
                                         "Pertimbangkan filtering noise atau menggunakan Heuristic Miner.")
                
                if len(bottlenecks) > 0:
                    recommendations.append(f"🚨 {len(bottlenecks)} bottleneck terdeteksi. "
                                         "Prioritaskan optimisasi pada activities tersebut.")
                
                if variant_ratio > 0.7:
                    recommendations.append("📊 Variant ratio tinggi menunjukkan process kurang terstandar. "
                                         "Pertimbangkan process redesign atau standardisasi.")
                
                if overall < 0.7:
                    recommendations.append("⚠️  Overall quality di bawah threshold. "
                                         "Review model dan pertimbangkan improvement.")
                
                if not recommendations:
                    recommendations.append("✅ Model dalam kondisi baik. Tidak ada rekomendasi khusus.")
                
                for idx, rec in enumerate(recommendations, 1):
                    f.write(f"{idx}. {rec}\n\n")
                
                # Next Steps
                f.write("=" * 80 + "\n")
                f.write("NEXT STEPS\n")
                f.write("=" * 80 + "\n\n")
                f.write("1. Review visualisasi (Petri Net, BPMN, DFG, Process Tree)\n")
                f.write("2. Analyze variant analysis untuk understand process behavior\n")
                f.write("3. Investigate bottleneck activities untuk optimization opportunities\n")
                f.write("4. Perform conformance checking dengan actual process execution\n")
                f.write("5. Consider process simulation untuk what-if analysis\n")
                f.write("6. Implement process improvements berdasarkan findings\n\n")
                
                # Footer
                f.write("=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            self.log_report(f"✅ Discovery report saved: '{report_path}'")
            
            return True
            
        except Exception as e:
            self.log_report(f"❌ ERROR generating report: {str(e)}")
            return False
    
    def _get_metric_bar(self, value):
        """Create visual bar untuk metric value"""
        filled = int(value * 20)
        empty = 20 - filled
        return "[" + "█" * filled + "░" * empty + "]"
    
    # ========== 10. MAIN ORCHESTRATION ==========
    def run(self):
        """
        Main orchestration method untuk menjalankan process discovery pipeline
        """
        print("\n" + "=" * 80)
        print("🚀 PROCESS DISCOVERY PIPELINE")
        print("=" * 80 + "\n")
        
        try:
            # Step 1: Load Data
            self.log_report("STEP 1: Loading preprocessed data...")
            if not self.load_preprocessed_data():
                return False
            
            # Step 2: Convert to Event Log
            self.log_report("\nSTEP 2: Converting to event log format...")
            if not self.convert_to_event_log():
                return False
            
            # Step 3: Discover DFG
            self.log_report("\nSTEP 3: Discovering Directly-Follows Graph...")
            if not self.discover_dfg():
                return False
            
            # Step 4: Visualize DFG
            self.log_report("\nSTEP 4: Visualizing DFG...")
            self.visualize_dfg()
            
            # Step 5: Discover Process Model
            self.log_report("\nSTEP 5: Discovering process model (Inductive Miner)...")
            if not self.discover_process_model(algorithm='inductive'):
                return False
            
            # Step 6: Visualize Petri Net
            self.log_report("\nSTEP 6: Visualizing Petri Net...")
            self.visualize_petri_net(with_frequency=True)
            
            # Step 7: Visualize Process Tree
            self.log_report("\nSTEP 7: Visualizing Process Tree...")
            self.visualize_process_tree()
            
            # Step 8: Visualize BPMN
            self.log_report("\nSTEP 8: Visualizing BPMN...")
            self.visualize_bpmn()
            
            # Step 9: Calculate Model Quality
            self.log_report("\nSTEP 9: Calculating model quality metrics...")
            self.calculate_model_quality()
            
            # Step 10: Discover Variants
            self.log_report("\nSTEP 10: Analyzing process variants...")
            self.discover_variants(top_n=10)
            
            # Step 11: Analyze Bottlenecks
            self.log_report("\nSTEP 11: Analyzing bottlenecks...")
            self.analyze_bottlenecks()
            
            # Step 12: Export Results
            self.log_report("\nSTEP 12: Exporting results...")
            self.export_results()
            
            # Step 13: Generate Report
            self.log_report("\nSTEP 13: Generating discovery report...")
            self.generate_discovery_report()
            
            # Success
            print("\n" + "=" * 80)
            print("✅ PROCESS DISCOVERY COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\nAll results saved in: '{self.output_dir}/'")
            print("\nKey Metrics:")
            print(f"  • Total Cases: {self.metrics.get('total_cases', 0):,}")
            print(f"  • Total Variants: {self.metrics.get('total_variants', 0):,}")
            print(f"  • Overall Quality: {self.metrics.get('overall_quality', 0):.4f}")
            print(f"  • Bottlenecks Detected: {self.metrics.get('bottleneck_count', 0)}")
            
            return True
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR in discovery pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# ========== MAIN EXECUTION BLOCK ==========
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("   PROCESS MINING - PROCESS DISCOVERY")
    print("   Using PM4PY Framework with Inductive Miner")
    print("=" * 80 + "\n")
    
    start_time = time.time()

    # Load dan rename
    df = pd.read_csv('D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/finale_preprocessed.csv')
    df = df.rename(columns={'Complete Timestamp': 'timestamp'})
    df.to_csv('D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/finale_preprocessed_renamed.csv', index=False)
    
    try:
        # Create ProcessDiscovery instance
        discoverer = ProcessDiscovery(
            input_file='D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/finale_preprocessed_renamed.csv',
            output_dir='D:/VSCODE/myenv2/share/Mining_Final_Project/Process_Discovery/process_discovery_output'
        )
        
        # Run discovery pipeline
        success = discoverer.run()
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        if success:
            print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
            print("\n🎉 Process Discovery berhasil!")
            print("\nOutput Files:")
            print("  1. petri_net.png - Petri Net visualization")
            print("  2. petri_net.pnml - Petri Net model (PNML format)")
            print("  3. process_tree.png - Process Tree visualization")
            print("  4. bpmn_diagram.png - BPMN visualization")
            print("  5. dfg_frequency.png - DFG with frequency")
            print("  6. dfg_performance.png - DFG with performance")
            print("  7. discovery_metrics.json - All metrics")
            print("  8. discovery_report.txt - Comprehensive report")
            print("\nNext Steps:")
            print("  1. Review 'discovery_report.txt' untuk analysis lengkap")
            print("  2. Examine visualisasi untuk understand process flow")
            print("  3. Investigate bottlenecks untuk process improvement")
            print("  4. Perform conformance checking jika diperlukan")
        else:
            print("\n❌ Process Discovery gagal. Check error messages di atas.")
    
    except FileNotFoundError:
        print("\n❌ ERROR: File 'finale_preprocessed.csv' tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan preprocessing terlebih dahulu.")
    
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {str(e)}")
        print("Pastikan PM4PY sudah terinstall: pip install pm4py")
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "=" * 80)
        print("Program selesai.")
        print("=" * 80 + "\n")