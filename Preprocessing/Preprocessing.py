"""
Script Preprocessing Data Process Mining - Comprehensive Version
Mencakup: EDA, Data Quality Checks, Timestamp Processing, Feature Engineering,
Outlier Detection, dan Validation untuk Process Mining
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import os

warnings.filterwarnings('ignore')

# Konfigurasi visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class ProcessMiningPreprocessor:
    """
    Class untuk preprocessing data process mining
    """
    
    def __init__(self, input_file='finale.csv', output_file='finale_preprocessed.csv', 
                 report_file='preprocessing_report.txt'):
        """
        Inisialisasi preprocessor
        
        Args:
            input_file: Nama file input CSV
            output_file: Nama file output CSV
            report_file: Nama file laporan preprocessing
        """
        self.input_file = input_file
        self.output_file = output_file
        self.report_file = report_file
        self.df = None
        self.df_original = None
        self.report = []
        self.stats = {}
        
    def log_report(self, message, print_msg=True):
        """
        Catat pesan ke report dan print ke console
        
        Args:
            message: Pesan yang akan dicatat
            print_msg: Boolean untuk print ke console
        """
        self.report.append(message)
        if print_msg:
            print(message)
    
    def add_separator(self):
        """Tambahkan separator untuk report"""
        separator = "=" * 80
        self.log_report(separator)
    
    # ========== 1. LOADING DATA ==========
    def load_data(self):
        """
        Load data dari CSV dengan error handling
        """
        try:
            self.log_report("\n🔄 MEMUAT DATA...\n")
            self.df = pd.read_csv(self.input_file)
            self.df_original = self.df.copy()
            self.log_report(f"✅ Data berhasil dimuat dari '{self.input_file}'")
            self.log_report(f"   Jumlah baris: {len(self.df):,}")
            self.log_report(f"   Jumlah kolom: {len(self.df.columns)}")
            return True
        except FileNotFoundError:
            self.log_report(f"❌ ERROR: File '{self.input_file}' tidak ditemukan!")
            return False
        except Exception as e:
            self.log_report(f"❌ ERROR saat memuat data: {str(e)}")
            return False
    
    # ========== 2. EXPLORATORY DATA ANALYSIS (EDA) ==========
    def perform_eda(self):
        """
        Lakukan Exploratory Data Analysis lengkap
        """
        self.add_separator()
        self.log_report("\n📊 EXPLORATORY DATA ANALYSIS (EDA)\n")
        
        # Info Dataset
        self.log_report("1. INFORMASI DATASET:")
        self.log_report(f"   Shape: {self.df.shape}")
        self.log_report(f"   Kolom: {list(self.df.columns)}")
        self.log_report("\n   Tipe Data:")
        for col, dtype in self.df.dtypes.items():
            self.log_report(f"   - {col}: {dtype}")
        
        # Statistik Deskriptif
        self.log_report("\n2. STATISTIK DESKRIPTIF:")
        self.log_report(str(self.df.describe(include='all')))
        
        # Simpan info untuk report
        self.stats['original_shape'] = self.df.shape
        self.stats['original_columns'] = list(self.df.columns)
        
        # Cek Missing Values
        self._check_missing_values()
        
        # Cek Duplikasi
        self._check_duplicates()
        
        # Tampilkan sample data
        self.log_report("\n3. SAMPLE DATA (5 baris pertama):")
        self.log_report(str(self.df.head()))
    
    def _check_missing_values(self):
        """
        Cek dan visualisasi missing values
        """
        self.log_report("\n4. CEK MISSING VALUES:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            self.log_report("   ⚠️  Ditemukan missing values:")
            self.log_report(str(missing_df))
            self.stats['missing_values'] = missing_df.to_dict()
            
            # Visualisasi Missing Values
            self._visualize_missing_values(missing_df)
        else:
            self.log_report("   ✅ Tidak ada missing values")
            self.stats['missing_values'] = None
    
    def _visualize_missing_values(self, missing_df):
        """
        Visualisasi missing values
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar chart
            missing_df['Missing Count'].plot(kind='bar', ax=ax1, color='coral')
            ax1.set_title('Missing Values Count per Column', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Columns')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Heatmap
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, ax=ax2, cmap='viridis')
            ax2.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('missing_values_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            self.log_report("   📊 Visualisasi missing values disimpan: 'missing_values_visualization.png'")
        except Exception as e:
            self.log_report(f"   ⚠️  Tidak dapat membuat visualisasi: {str(e)}")
    
    def _check_duplicates(self):
        """
        Cek duplikasi data
        """
        self.log_report("\n5. CEK DUPLIKASI DATA:")
        duplicates = self.df.duplicated().sum()
        duplicate_pct = (duplicates / len(self.df)) * 100
        
        self.log_report(f"   Total baris duplikat: {duplicates:,} ({duplicate_pct:.2f}%)")
        self.stats['duplicates_count'] = duplicates
        
        if duplicates > 0:
            self.log_report("   ⚠️  Ditemukan data duplikat yang perlu dibersihkan")
        else:
            self.log_report("   ✅ Tidak ada data duplikat")
    
    # ========== 3. DATA QUALITY CHECKS & CLEANING ==========
    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values dengan berbagai strategi
        
        Args:
            strategy: 'drop', 'ffill', 'bfill', atau 'mean'
        """
        self.add_separator()
        self.log_report("\n🧹 HANDLING MISSING VALUES\n")
        
        initial_rows = len(self.df)
        
        if self.df.isnull().sum().sum() == 0:
            self.log_report("✅ Tidak ada missing values yang perlu di-handle")
            return
        
        try:
            if strategy == 'drop':
                self.df = self.df.dropna()
                self.log_report(f"   Strategi: Drop rows dengan missing values")
            elif strategy == 'ffill':
                self.df = self.df.fillna(method='ffill')
                self.log_report(f"   Strategi: Forward fill")
            elif strategy == 'bfill':
                self.df = self.df.fillna(method='bfill')
                self.log_report(f"   Strategi: Backward fill")
            elif strategy == 'mean':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
                self.log_report(f"   Strategi: Fill dengan mean (kolom numerik)")
            
            rows_removed = initial_rows - len(self.df)
            self.log_report(f"   Baris yang dihapus: {rows_removed:,}")
            self.log_report(f"   Baris tersisa: {len(self.df):,}")
            self.stats['missing_handled'] = rows_removed
            
        except Exception as e:
            self.log_report(f"❌ ERROR saat handling missing values: {str(e)}")
    
    def remove_duplicates(self):
        """
        Hapus data duplikat
        """
        self.add_separator()
        self.log_report("\n🧹 MENGHAPUS DUPLIKASI DATA\n")
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        rows_removed = initial_rows - len(self.df)
        
        self.log_report(f"   Baris duplikat dihapus: {rows_removed:,}")
        self.log_report(f"   Baris tersisa: {len(self.df):,}")
        self.stats['duplicates_removed'] = rows_removed
    
    def validate_data_format(self):
        """
        Validasi format data untuk process mining
        """
        self.add_separator()
        self.log_report("\n✅ VALIDASI FORMAT DATA\n")
        
        # Identifikasi kolom untuk Case ID, Activity, Timestamp
        possible_case_cols = ['case_id', 'caseid', 'case', 'id', 'case id']
        possible_activity_cols = ['activity', 'activity_name', 'event', 'task']
        possible_timestamp_cols = ['timestamp', 'time', 'datetime', 'date', 'start_time', 'end_time']
        
        # Deteksi kolom (case-insensitive)
        col_lower = {col.lower(): col for col in self.df.columns}
        
        case_col = None
        activity_col = None
        timestamp_col = None
        
        for pc in possible_case_cols:
            if pc in col_lower:
                case_col = col_lower[pc]
                break
        
        for ac in possible_activity_cols:
            if ac in col_lower:
                activity_col = col_lower[ac]
                break
        
        for tc in possible_timestamp_cols:
            if tc in col_lower:
                timestamp_col = col_lower[tc]
                break
        
        self.log_report("   Kolom teridentifikasi:")
        self.log_report(f"   - Case ID: {case_col if case_col else '❌ Tidak ditemukan'}")
        self.log_report(f"   - Activity: {activity_col if activity_col else '❌ Tidak ditemukan'}")
        self.log_report(f"   - Timestamp: {timestamp_col if timestamp_col else '❌ Tidak ditemukan'}")
        
        # Simpan nama kolom untuk digunakan nanti
        self.stats['case_col'] = case_col
        self.stats['activity_col'] = activity_col
        self.stats['timestamp_col'] = timestamp_col
        
        # Validasi keberadaan kolom penting
        if not all([case_col, activity_col, timestamp_col]):
            self.log_report("\n   ⚠️  PERINGATAN: Beberapa kolom penting tidak ditemukan!")
            self.log_report("   Pastikan dataset memiliki kolom: Case ID, Activity, Timestamp")
        else:
            self.log_report("\n   ✅ Semua kolom penting teridentifikasi")
    
    # ========== 4. TIMESTAMP PREPROCESSING ==========
    def process_timestamps(self):
        """
        Proses timestamp: parse, sort, validate
        """
        self.add_separator()
        self.log_report("\n⏰ PREPROCESSING TIMESTAMP\n")
        
        timestamp_col = self.stats.get('timestamp_col')
        case_col = self.stats.get('case_col')
        
        if not timestamp_col:
            self.log_report("   ⚠️  Kolom timestamp tidak ditemukan, skip timestamp processing")
            return
        
        try:
            # Parse timestamp
            self.log_report("   1. Parsing timestamp ke datetime format...")
            original_type = self.df[timestamp_col].dtype
            self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col], errors='coerce')
            
            # Check invalid timestamps
            invalid_timestamps = self.df[timestamp_col].isnull().sum()
            if invalid_timestamps > 0:
                self.log_report(f"      ⚠️  {invalid_timestamps} timestamp tidak valid ditemukan dan diset ke NaT")
                self.df = self.df.dropna(subset=[timestamp_col])
            
            self.log_report(f"      ✅ Timestamp berhasil diparse dari {original_type} ke datetime64")
            
            # Sort berdasarkan case ID dan timestamp
            if case_col:
                self.log_report("\n   2. Sorting data berdasarkan Case ID dan Timestamp...")
                self.df = self.df.sort_values([case_col, timestamp_col]).reset_index(drop=True)
                self.log_report("      ✅ Data berhasil disortir")
            
            # Validasi timestamp consistency
            self._validate_timestamp_consistency()
            
            # Calculate durations
            self._calculate_durations()
            
        except Exception as e:
            self.log_report(f"   ❌ ERROR saat processing timestamp: {str(e)}")
    
    def _validate_timestamp_consistency(self):
        """
        Validasi konsistensi timestamp
        """
        self.log_report("\n   3. Validasi konsistensi timestamp...")
        
        timestamp_col = self.stats.get('timestamp_col')
        case_col = self.stats.get('case_col')
        
        if not all([timestamp_col, case_col]):
            return
        
        try:
            # Check untuk timestamp yang tidak berurutan dalam satu case
            inconsistent_cases = []
            for case_id in self.df[case_col].unique():
                case_data = self.df[self.df[case_col] == case_id]
                timestamps = case_data[timestamp_col].values
                if len(timestamps) > 1 and not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                    inconsistent_cases.append(case_id)
            
            if inconsistent_cases:
                self.log_report(f"      ⚠️  {len(inconsistent_cases)} case memiliki timestamp tidak konsisten")
                self.stats['inconsistent_timestamps'] = len(inconsistent_cases)
            else:
                self.log_report("      ✅ Semua timestamp konsisten dan berurutan")
                self.stats['inconsistent_timestamps'] = 0
                
        except Exception as e:
            self.log_report(f"      ⚠️  Error validasi: {str(e)}")
    
    def _calculate_durations(self):
        """
        Hitung case duration dan activity duration
        """
        self.log_report("\n   4. Menghitung durasi...")
        
        timestamp_col = self.stats.get('timestamp_col')
        case_col = self.stats.get('case_col')
        
        if not all([timestamp_col, case_col]):
            return
        
        try:
            # Case Duration (durasi dari start hingga end per case)
            case_durations = self.df.groupby(case_col)[timestamp_col].agg(['min', 'max'])
            case_durations['case_duration_seconds'] = (case_durations['max'] - case_durations['min']).dt.total_seconds()
            case_durations['case_duration_hours'] = case_durations['case_duration_seconds'] / 3600
            
            # Merge ke dataframe
            self.df = self.df.merge(case_durations[['case_duration_seconds', 'case_duration_hours']], 
                                    left_on=case_col, right_index=True, how='left')
            
            # Time between events (activity duration)
            self.df['time_since_last_event'] = self.df.groupby(case_col)[timestamp_col].diff().dt.total_seconds()
            
            self.log_report("      ✅ Case duration dan time between events berhasil dihitung")
            self.log_report(f"      Rata-rata case duration: {case_durations['case_duration_hours'].mean():.2f} jam")
            self.log_report(f"      Median case duration: {case_durations['case_duration_hours'].median():.2f} jam")
            
            self.stats['avg_case_duration_hours'] = case_durations['case_duration_hours'].mean()
            self.stats['median_case_duration_hours'] = case_durations['case_duration_hours'].median()
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR menghitung durasi: {str(e)}")
    
    # ========== 5. CASE ID & ACTIVITY PROCESSING ==========
    def process_case_and_activity(self):
        """
        Proses Case ID dan Activity: validasi, standardisasi, analisis
        """
        self.add_separator()
        self.log_report("\n🔍 PROCESSING CASE ID & ACTIVITY\n")
        
        case_col = self.stats.get('case_col')
        activity_col = self.stats.get('activity_col')
        
        if not all([case_col, activity_col]):
            self.log_report("   ⚠️  Kolom Case ID atau Activity tidak ditemukan")
            return
        
        # Validasi Case IDs
        self._validate_case_ids()
        
        # Standardisasi Activity Names
        self._standardize_activity_names()
        
        # Hitung frekuensi activities
        self._calculate_activity_frequencies()
        
        # Identifikasi case variants
        self._identify_case_variants()
    
    def _validate_case_ids(self):
        """
        Validasi Case IDs
        """
        self.log_report("   1. Validasi Case IDs...")
        
        case_col = self.stats.get('case_col')
        
        total_cases = self.df[case_col].nunique()
        total_events = len(self.df)
        avg_events_per_case = total_events / total_cases
        
        self.log_report(f"      Total unique cases: {total_cases:,}")
        self.log_report(f"      Total events: {total_events:,}")
        self.log_report(f"      Rata-rata events per case: {avg_events_per_case:.2f}")
        
        self.stats['total_cases'] = total_cases
        self.stats['total_events'] = total_events
        self.stats['avg_events_per_case'] = avg_events_per_case
        
        # Check untuk case IDs yang null
        null_cases = self.df[case_col].isnull().sum()
        if null_cases > 0:
            self.log_report(f"      ⚠️  {null_cases} events dengan Case ID null ditemukan")
            self.df = self.df.dropna(subset=[case_col])
            self.log_report(f"      ✅ Events dengan Case ID null telah dihapus")
    
    def _standardize_activity_names(self):
        """
        Standardisasi nama activities
        """
        self.log_report("\n   2. Standardisasi Activity Names...")
        
        activity_col = self.stats.get('activity_col')
        
        try:
            # Simpan activities original untuk comparison
            original_activities = self.df[activity_col].unique()
            
            # Lowercase dan trim whitespace
            self.df[activity_col] = self.df[activity_col].astype(str).str.strip().str.lower()
            
            # Standardized activities
            standardized_activities = self.df[activity_col].unique()
            
            self.log_report(f"      Original unique activities: {len(original_activities)}")
            self.log_report(f"      Standardized unique activities: {len(standardized_activities)}")
            
            if len(original_activities) > len(standardized_activities):
                diff = len(original_activities) - len(standardized_activities)
                self.log_report(f"      ✅ {diff} duplicate activities ditemukan dan distandarisasi")
            else:
                self.log_report("      ✅ Activities berhasil distandarisasi")
            
            self.stats['unique_activities'] = len(standardized_activities)
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR standardisasi: {str(e)}")
    
    def _calculate_activity_frequencies(self):
        """
        Hitung frekuensi activities
        """
        self.log_report("\n   3. Menghitung frekuensi activities...")
        
        activity_col = self.stats.get('activity_col')
        
        try:
            activity_freq = self.df[activity_col].value_counts()
            
            self.log_report(f"      Top 10 Activities:")
            for idx, (activity, count) in enumerate(activity_freq.head(10).items(), 1):
                pct = (count / len(self.df)) * 100
                self.log_report(f"      {idx}. {activity}: {count:,} ({pct:.2f}%)")
            
            self.stats['activity_frequencies'] = activity_freq.to_dict()
            
            # Visualisasi
            self._visualize_activity_frequencies(activity_freq)
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR: {str(e)}")
    
    def _visualize_activity_frequencies(self, activity_freq):
        """
        Visualisasi frekuensi activities
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            activity_freq.head(15).plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title('Top 15 Activity Frequencies', fontsize=14, fontweight='bold')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Activity')
            plt.tight_layout()
            plt.savefig('activity_frequencies.png', dpi=300, bbox_inches='tight')
            plt.close()
            self.log_report("      📊 Visualisasi disimpan: 'activity_frequencies.png'")
        except Exception as e:
            self.log_report(f"      ⚠️  Tidak dapat membuat visualisasi: {str(e)}")
    
    def _identify_case_variants(self):
        """
        Identifikasi case variants (urutan activities yang unik)
        """
        self.log_report("\n   4. Mengidentifikasi Case Variants...")
        
        case_col = self.stats.get('case_col')
        activity_col = self.stats.get('activity_col')
        
        try:
            # Group by case dan buat sequence dari activities
            variants = self.df.groupby(case_col)[activity_col].apply(lambda x: ' -> '.join(x)).reset_index()
            variants.columns = [case_col, 'variant']
            
            # Hitung frekuensi variants
            variant_freq = variants['variant'].value_counts()
            
            self.log_report(f"      Total unique variants: {len(variant_freq):,}")
            self.log_report(f"\n      Top 5 Most Common Variants:")
            for idx, (variant, count) in enumerate(variant_freq.head(5).items(), 1):
                pct = (count / len(variants)) * 100
                self.log_report(f"      {idx}. [{count} cases, {pct:.2f}%] {variant[:100]}...")
            
            # Merge variant info ke dataframe
            self.df = self.df.merge(variants, on=case_col, how='left')
            
            self.stats['total_variants'] = len(variant_freq)
            self.stats['top_variant_frequency'] = variant_freq.iloc[0] if len(variant_freq) > 0 else 0
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR: {str(e)}")
    
    # ========== 6. FEATURE ENGINEERING ==========
    def engineer_features(self):
        """
        Create additional features untuk analisis process mining
        """
        self.add_separator()
        self.log_report("\n⚙️  FEATURE ENGINEERING\n")
        
        case_col = self.stats.get('case_col')
        activity_col = self.stats.get('activity_col')
        timestamp_col = self.stats.get('timestamp_col')
        
        if not all([case_col, activity_col, timestamp_col]):
            self.log_report("   ⚠️  Kolom penting tidak lengkap, skip feature engineering")
            return
        
        try:
            # 1. Activity Sequence Number
            self.log_report("   1. Membuat Activity Sequence Number...")
            self.df['activity_sequence_num'] = self.df.groupby(case_col).cumcount() + 1
            self.log_report("      ✅ Activity sequence number ditambahkan")
            
            # 2. Extract Temporal Features
            self.log_report("\n   2. Extract temporal features...")
            self.df['hour'] = self.df[timestamp_col].dt.hour
            self.df['day'] = self.df[timestamp_col].dt.day
            self.df['month'] = self.df[timestamp_col].dt.month
            self.df['year'] = self.df[timestamp_col].dt.year
            self.df['weekday'] = self.df[timestamp_col].dt.dayofweek  # 0=Monday, 6=Sunday
            self.df['weekday_name'] = self.df[timestamp_col].dt.day_name()
            self.df['is_weekend'] = self.df['weekday'].isin([5, 6]).astype(int)
            self.log_report("      ✅ Temporal features ditambahkan (hour, day, month, weekday, is_weekend)")
            
            # 3. Activity Position (first, middle, last)
            self.log_report("\n   3. Menentukan Activity Position...")
            case_sizes = self.df.groupby(case_col).size()
            self.df['case_size'] = self.df[case_col].map(case_sizes)
            
            def get_position(row):
                if row['activity_sequence_num'] == 1:
                    return 'first'
                elif row['activity_sequence_num'] == row['case_size']:
                    return 'last'
                else:
                    return 'middle'
            
            self.df['activity_position'] = self.df.apply(get_position, axis=1)
            self.log_report("      ✅ Activity position ditambahkan (first, middle, last)")
            
            # 4. Time between events (jika belum ada)
            if 'time_since_last_event' not in self.df.columns:
                self.log_report("\n   4. Menghitung time between events...")
                self.df['time_since_last_event'] = self.df.groupby(case_col)[timestamp_col].diff().dt.total_seconds()
                self.log_report("      ✅ Time between events ditambahkan")
            
            # 5. Cumulative time in case
            self.log_report("\n   5. Menghitung cumulative time in case...")
            first_timestamp = self.df.groupby(case_col)[timestamp_col].transform('first')
            self.df['cumulative_time_seconds'] = (self.df[timestamp_col] - first_timestamp).dt.total_seconds()
            self.df['cumulative_time_hours'] = self.df['cumulative_time_seconds'] / 3600
            self.log_report("      ✅ Cumulative time ditambahkan")
            
            self.log_report("\n   ✅ Feature engineering selesai!")
            self.log_report(f"   Total features sekarang: {len(self.df.columns)}")
            
        except Exception as e:
            self.log_report(f"   ❌ ERROR dalam feature engineering: {str(e)}")
    
    # ========== 7. OUTLIER DETECTIO ==========
    def detect_outliers(self):
        """
        Deteksi outliers dalam case duration dan activity count
        """
        self.add_separator()
        self.log_report("\n🔎 OUTLIER DETECTION\n")
        
        case_col = self.stats.get('case_col')
        
        if not case_col:
            self.log_report("   ⚠️  Case column tidak ditemukan, skip outlier detection")
            return
        
        # 1. Outliers berdasarkan Case Duration
        self._detect_duration_outliers()
        
        # 2. Outliers berdasarkan Activity Count
        self._detect_activity_count_outliers()
        
        # 3. Visualisasi Outliers
        self._visualize_outliers()
    
    def _detect_duration_outliers(self):
        """
        Deteksi outliers berdasarkan case duration menggunakan IQR method
        """
        self.log_report("   1. Deteksi Duration Outliers (IQR Method)...")
        
        case_col = self.stats.get('case_col')
        
        if 'case_duration_hours' not in self.df.columns:
            self.log_report("      ⚠️  Case duration belum dihitung, skip")
            return
        
        try:
            # Ambil unique case durations
            case_durations = self.df.groupby(case_col)['case_duration_hours'].first()
            
            # Hitung IQR
            Q1 = case_durations.quantile(0.25)
            Q3 = case_durations.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identifikasi outliers
            outliers = case_durations[(case_durations < lower_bound) | (case_durations > upper_bound)]
            outlier_cases = outliers.index.tolist()
            
            # Tandai outliers di dataframe
            self.df['is_duration_outlier'] = self.df[case_col].isin(outlier_cases).astype(int)
            
            self.log_report(f"      Q1 (25th percentile): {Q1:.2f} hours")
            self.log_report(f"      Q3 (75th percentile): {Q3:.2f} hours")
            self.log_report(f"      IQR: {IQR:.2f} hours")
            self.log_report(f"      Lower Bound: {lower_bound:.2f} hours")
            self.log_report(f"      Upper Bound: {upper_bound:.2f} hours")
            self.log_report(f"      ✅ {len(outliers)} cases terdeteksi sebagai duration outliers ({len(outliers)/len(case_durations)*100:.2f}%)")
            
            if len(outliers) > 0:
                self.log_report(f"\n      Top 5 Longest Duration Cases:")
                for idx, (case_id, duration) in enumerate(outliers.nlargest(5).items(), 1):
                    self.log_report(f"      {idx}. Case {case_id}: {duration:.2f} hours")
            
            self.stats['duration_outliers'] = len(outliers)
            self.stats['duration_outlier_percentage'] = len(outliers)/len(case_durations)*100
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR: {str(e)}")
    
    def _detect_activity_count_outliers(self):
        """
        Deteksi outliers berdasarkan jumlah activity per case
        """
        self.log_report("\n   2. Deteksi Activity Count Outliers (IQR Method)...")
        
        case_col = self.stats.get('case_col')
        
        try:
            # Hitung jumlah activities per case
            activity_counts = self.df.groupby(case_col).size()
            
            # Hitung IQR
            Q1 = activity_counts.quantile(0.25)
            Q3 = activity_counts.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identifikasi outliers
            outliers = activity_counts[(activity_counts < lower_bound) | (activity_counts > upper_bound)]
            outlier_cases = outliers.index.tolist()
            
            # Tandai outliers di dataframe
            self.df['is_activity_count_outlier'] = self.df[case_col].isin(outlier_cases).astype(int)
            
            self.log_report(f"      Q1 (25th percentile): {Q1:.0f} activities")
            self.log_report(f"      Q3 (75th percentile): {Q3:.0f} activities")
            self.log_report(f"      IQR: {IQR:.0f}")
            self.log_report(f"      Lower Bound: {max(0, lower_bound):.0f} activities")
            self.log_report(f"      Upper Bound: {upper_bound:.0f} activities")
            self.log_report(f"      ✅ {len(outliers)} cases terdeteksi sebagai activity count outliers ({len(outliers)/len(activity_counts)*100:.2f}%)")
            
            if len(outliers) > 0:
                self.log_report(f"\n      Top 5 Cases dengan Activity Terbanyak:")
                for idx, (case_id, count) in enumerate(outliers.nlargest(5).items(), 1):
                    self.log_report(f"      {idx}. Case {case_id}: {count} activities")
            
            self.stats['activity_count_outliers'] = len(outliers)
            self.stats['activity_count_outlier_percentage'] = len(outliers)/len(activity_counts)*100
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR: {str(e)}")
    
    def _visualize_outliers(self):
        """
        Visualisasi outliers dengan boxplot dan histogram
        """
        self.log_report("\n   3. Membuat visualisasi outliers...")
        
        case_col = self.stats.get('case_col')
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Case Duration Boxplot
            if 'case_duration_hours' in self.df.columns:
                case_durations = self.df.groupby(case_col)['case_duration_hours'].first()
                axes[0, 0].boxplot(case_durations.values, vert=True)
                axes[0, 0].set_title('Case Duration Distribution (Boxplot)', fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel('Duration (hours)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Case Duration Histogram
            if 'case_duration_hours' in self.df.columns:
                axes[0, 1].hist(case_durations.values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                axes[0, 1].set_title('Case Duration Distribution (Histogram)', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Duration (hours)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Activity Count Boxplot
            activity_counts = self.df.groupby(case_col).size()
            axes[1, 0].boxplot(activity_counts.values, vert=True)
            axes[1, 0].set_title('Activity Count per Case (Boxplot)', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Number of Activities')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Activity Count Histogram
            axes[1, 1].hist(activity_counts.values, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Activity Count per Case (Histogram)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Number of Activities')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('outliers_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_report("      📊 Visualisasi outliers disimpan: 'outliers_visualization.png'")
            
        except Exception as e:
            self.log_report(f"      ⚠️  Tidak dapat membuat visualisasi: {str(e)}")
    
    # ========== 8. VALIDATION FOR PROCESS MINING ==========
    def validate_for_process_mining(self):
        """
        Validasi final untuk memastikan data siap untuk process mining
        """
        self.add_separator()
        self.log_report("\n✅ VALIDASI DATA UNTUK PROCESS MINING\n")
        
        case_col = self.stats.get('case_col')
        activity_col = self.stats.get('activity_col')
        timestamp_col = self.stats.get('timestamp_col')
        
        validation_passed = True
        
        # 1. Check kolom penting
        self.log_report("   1. Validasi Kolom Penting...")
        if not all([case_col, activity_col, timestamp_col]):
            self.log_report("      ❌ GAGAL: Kolom penting (Case ID, Activity, Timestamp) tidak lengkap")
            validation_passed = False
        else:
            self.log_report("      ✅ Semua kolom penting tersedia")
        
        # 2. Check incomplete cases
        self._check_incomplete_cases()
        
        # 3. Validate activity sequences
        self._validate_activity_sequences()
        
        # 4. Check for circular dependencies
        self._check_circular_dependencies()
        
        # 5. Validate minimum requirements
        self._validate_minimum_requirements()
        
        # Final validation result
        if validation_passed:
            self.log_report("\n   🎉 VALIDASI BERHASIL! Data siap untuk Process Mining")
            self.stats['validation_status'] = 'PASSED'
        else:
            self.log_report("\n   ⚠️  PERHATIAN: Beberapa validasi tidak lolos, review diperlukan")
            self.stats['validation_status'] = 'WARNING'
    
    def _check_incomplete_cases(self):
        """
        Check untuk cases yang tidak lengkap atau suspicious
        """
        self.log_report("\n   2. Check Incomplete Cases...")
        
        case_col = self.stats.get('case_col')
        activity_col = self.stats.get('activity_col')
        
        try:
            # Cases dengan hanya 1 activity (mungkin incomplete)
            single_activity_cases = self.df.groupby(case_col).size()
            single_activity_cases = single_activity_cases[single_activity_cases == 1]
            
            if len(single_activity_cases) > 0:
                pct = len(single_activity_cases) / self.df[case_col].nunique() * 100
                self.log_report(f"      ⚠️  {len(single_activity_cases)} cases dengan hanya 1 activity ({pct:.2f}%)")
                self.stats['single_activity_cases'] = len(single_activity_cases)
            else:
                self.log_report("      ✅ Tidak ada cases dengan single activity")
                self.stats['single_activity_cases'] = 0
            
            # Cases tanpa start atau end activity yang jelas
            # Asumsi: activity pertama adalah start, terakhir adalah end
            case_starts = self.df.groupby(case_col)[activity_col].first()
            case_ends = self.df.groupby(case_col)[activity_col].last()
            
            unique_starts = case_starts.nunique()
            unique_ends = case_ends.nunique()
            
            self.log_report(f"\n      Info Start/End Activities:")
            self.log_report(f"      - Unique start activities: {unique_starts}")
            self.log_report(f"      - Unique end activities: {unique_ends}")
            
            if unique_starts > 5 or unique_ends > 5:
                self.log_report(f"      ⚠️  Banyak variasi start/end activities, mungkin tidak ada standard process flow")
            else:
                self.log_report(f"      ✅ Variasi start/end activities dalam batas wajar")
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR: {str(e)}")
    
    def _validate_activity_sequences(self):
        """
        Validasi activity sequences untuk mendeteksi pola aneh
        """
        self.log_report("\n   3. Validasi Activity Sequences...")
        
        case_col = self.stats.get('case_col')
        activity_col = self.stats.get('activity_col')
        
        try:
            # Detect duplicate consecutive activities (activity yang berulang langsung)
            self.df['prev_activity'] = self.df.groupby(case_col)[activity_col].shift(1)
            consecutive_duplicates = self.df[self.df[activity_col] == self.df['prev_activity']]
            
            if len(consecutive_duplicates) > 0:
                pct = len(consecutive_duplicates) / len(self.df) * 100
                self.log_report(f"      ⚠️  {len(consecutive_duplicates)} events dengan consecutive duplicate activities ({pct:.2f}%)")
                self.log_report(f"      Contoh: {consecutive_duplicates[[case_col, activity_col]].head(3).to_dict('records')}")
                self.stats['consecutive_duplicates'] = len(consecutive_duplicates)
            else:
                self.log_report("      ✅ Tidak ada consecutive duplicate activities")
                self.stats['consecutive_duplicates'] = 0
            
            # Drop temporary column
            self.df.drop('prev_activity', axis=1, inplace=True)
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR: {str(e)}")
    
    def _check_circular_dependencies(self):
        """
        Check untuk circular dependencies atau loop dalam process
        """
        self.log_report("\n   4. Check Circular Dependencies...")
        
        case_col = self.stats.get('case_col')
        activity_col = self.stats.get('activity_col')
        
        try:
            # Detect cases dengan activity yang muncul lebih dari sekali
            activity_repeats = self.df.groupby([case_col, activity_col]).size().reset_index(name='count')
            repeated_activities = activity_repeats[activity_repeats['count'] > 1]
            
            cases_with_loops = repeated_activities[case_col].nunique()
            
            if cases_with_loops > 0:
                pct = cases_with_loops / self.df[case_col].nunique() * 100
                self.log_report(f"      ⚠️  {cases_with_loops} cases dengan repeated activities/potential loops ({pct:.2f}%)")
                
                # Show most repeated activity
                most_repeated = repeated_activities.nlargest(5, 'count')
                self.log_report(f"      Top repeated activities:")
                for _, row in most_repeated.iterrows():
                    self.log_report(f"      - Case {row[case_col]}, Activity '{row[activity_col]}': {row['count']} occurrences")
                
                self.stats['cases_with_loops'] = cases_with_loops
            else:
                self.log_report("      ✅ Tidak ada repeated activities (no loops detected)")
                self.stats['cases_with_loops'] = 0
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR: {str(e)}")
    
    def _validate_minimum_requirements(self):
        """
        Validasi minimum requirements untuk process mining
        """
        self.log_report("\n   5. Validasi Minimum Requirements...")
        
        case_col = self.stats.get('case_col')
        
        try:
            min_cases_required = 10
            min_events_required = 50
            min_activities_required = 3
            
            total_cases = self.df[case_col].nunique()
            total_events = len(self.df)
            total_activities = self.stats.get('unique_activities', 0)
            
            checks = [
                (total_cases >= min_cases_required, f"Minimum {min_cases_required} cases", total_cases),
                (total_events >= min_events_required, f"Minimum {min_events_required} events", total_events),
                (total_activities >= min_activities_required, f"Minimum {min_activities_required} unique activities", total_activities)
            ]
            
            all_passed = True
            for passed, requirement, actual in checks:
                status = "✅" if passed else "❌"
                self.log_report(f"      {status} {requirement}: {actual}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                self.log_report("\n      ✅ Semua minimum requirements terpenuhi")
            else:
                self.log_report("\n      ❌ Beberapa minimum requirements tidak terpenuhi")
            
            self.stats['minimum_requirements_passed'] = all_passed
            
        except Exception as e:
            self.log_report(f"      ❌ ERROR: {str(e)}")
    
    # ========== 9. EXPORT DATA ==========
    def export_data(self):
        """
        Export hasil preprocessing
        """
        self.add_separator()
        self.log_report("\n💾 EXPORT DATA\n")
        
        try:
            # 1. Save cleaned data
            self.log_report("   1. Menyimpan cleaned data...")
            self.df.to_csv(self.output_file, index=False)
            file_size = os.path.getsize(self.output_file) / (1024 * 1024)  # MB
            self.log_report(f"      ✅ Cleaned data disimpan: '{self.output_file}' ({file_size:.2f} MB)")
            
            # 2. Save statistics to JSON
            self.log_report("\n   2. Menyimpan statistics...")
            import json
            stats_file = 'preprocessing_statistics.json'
            
            # Convert numpy types to native Python types for JSON serialization
            stats_serializable = {}
            for key, value in self.stats.items():
                if isinstance(value, (np.integer, np.floating)):
                    stats_serializable[key] = float(value)
                elif isinstance(value, (np.ndarray, pd.Series)):
                    stats_serializable[key] = value.tolist()
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    stats_serializable[key] = {str(k): (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                                              for k, v in value.items()}
                else:
                    stats_serializable[key] = value
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_serializable, f, indent=4, ensure_ascii=False)
            
            self.log_report(f"      ✅ Statistics disimpan: '{stats_file}'")
            
            # 3. Create backup of original data
            self.log_report("\n   3. Membuat backup original data...")
            backup_file = 'finale_original_backup.csv'
            self.df_original.to_csv(backup_file, index=False)
            backup_size = os.path.getsize(backup_file) / (1024 * 1024)  # MB
            self.log_report(f"      ✅ Backup original data: '{backup_file}' ({backup_size:.2f} MB)")
            
            self.log_report("\n   🎉 Semua data berhasil di-export!")
            
        except Exception as e:
            self.log_report(f"   ❌ ERROR saat export: {str(e)}")
    
    # ========== 10. GENERATE SUMMARY REPORT ==========
    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        self.add_separator()
        self.log_report("\n📋 SUMMARY REPORT\n")
        
        # Before/After Comparison
        self.log_report("=" * 80)
        self.log_report("BEFORE vs AFTER PREPROCESSING")
        self.log_report("=" * 80)
        
        original_shape = self.stats.get('original_shape', (0, 0))
        current_shape = self.df.shape
        
        self.log_report(f"\nJumlah Baris:")
        self.log_report(f"  Before: {original_shape[0]:,}")
        self.log_report(f"  After:  {current_shape[0]:,}")
        rows_change = current_shape[0] - original_shape[0]
        change_pct = (rows_change / original_shape[0] * 100) if original_shape[0] > 0 else 0
        self.log_report(f"  Change: {rows_change:,} ({change_pct:+.2f}%)")
        
        self.log_report(f"\nJumlah Kolom:")
        self.log_report(f"  Before: {original_shape[1]}")
        self.log_report(f"  After:  {current_shape[1]}")
        self.log_report(f"  Added:  {current_shape[1] - original_shape[1]} new features")
        
        # Data Quality Metrics
        self.add_separator()
        self.log_report("\nDATA QUALITY METRICS")
        self.log_report("=" * 80)
        
        self.log_report(f"\n✅ Data Cleaning:")
        self.log_report(f"  - Missing values handled: {self.stats.get('missing_handled', 0):,}")
        self.log_report(f"  - Duplicates removed: {self.stats.get('duplicates_removed', 0):,}")
        self.log_report(f"  - Duration outliers detected: {self.stats.get('duration_outliers', 0):,}")
        self.log_report(f"  - Activity count outliers: {self.stats.get('activity_count_outliers', 0):,}")
        
        self.log_report(f"\n📊 Process Mining Metrics:")
        self.log_report(f"  - Total unique cases: {self.stats.get('total_cases', 0):,}")
        self.log_report(f"  - Total events: {self.stats.get('total_events', 0):,}")
        self.log_report(f"  - Avg events per case: {self.stats.get('avg_events_per_case', 0):.2f}")
        self.log_report(f"  - Unique activities: {self.stats.get('unique_activities', 0)}")
        self.log_report(f"  - Total variants: {self.stats.get('total_variants', 0):,}")
        self.log_report(f"  - Avg case duration: {self.stats.get('avg_case_duration_hours', 0):.2f} hours")
        self.log_report(f"  - Median case duration: {self.stats.get('median_case_duration_hours', 0):.2f} hours")
        
        self.log_report(f"\n⚠️  Potential Issues:")
        self.log_report(f"  - Single activity cases: {self.stats.get('single_activity_cases', 0)}")
        self.log_report(f"  - Cases with loops: {self.stats.get('cases_with_loops', 0)}")
        self.log_report(f"  - Consecutive duplicates: {self.stats.get('consecutive_duplicates', 0)}")
        
        # Validation Status
        self.add_separator()
        self.log_report("\nVALIDATION STATUS")
        self.log_report("=" * 80)
        validation_status = self.stats.get('validation_status', 'UNKNOWN')
        status_icon = "✅" if validation_status == "PASSED" else "⚠️"
        self.log_report(f"\n{status_icon} Overall Status: {validation_status}")
        
        # Recommendations
        self.add_separator()
        self.log_report("\nRECOMMENDATIONS")
        self.log_report("=" * 80)
        
        recommendations = []
        
        # Check outliers
        duration_outliers_pct = self.stats.get('duration_outlier_percentage', 0)
        if duration_outliers_pct > 5:
            recommendations.append(f"⚠️  {duration_outliers_pct:.1f}% cases memiliki duration outliers. "
                                 "Pertimbangkan untuk investigasi atau filter cases tersebut.")
        
        # Check loops
        cases_with_loops = self.stats.get('cases_with_loops', 0)
        if cases_with_loops > 0:
            recommendations.append(f"⚠️  {cases_with_loops} cases memiliki repeated activities (loops). "
                                 "Pastikan ini adalah behavior yang expected dalam proses bisnis.")
        
        # Check variants
        total_variants = self.stats.get('total_variants', 0)
        total_cases = self.stats.get('total_cases', 1)
        variant_ratio = total_variants / total_cases if total_cases > 0 else 0
        if variant_ratio > 0.8:
            recommendations.append(f"⚠️  Variant ratio tinggi ({variant_ratio:.2%}). "
                                 "Process mungkin terlalu flexible atau tidak terstruktur.")
        
        # Check activity count
        single_activity_cases = self.stats.get('single_activity_cases', 0)
        if single_activity_cases > 0:
            recommendations.append(f"⚠️  {single_activity_cases} cases hanya memiliki 1 activity. "
                                 "Review apakah cases ini complete atau perlu difilter.")
        
        if not recommendations:
            self.log_report("\n✅ Tidak ada recommendations khusus. Data dalam kondisi baik!")
        else:
            for i, rec in enumerate(recommendations, 1):
                self.log_report(f"\n{i}. {rec}")
        
        # Next Steps
        self.add_separator()
        self.log_report("\nNEXT STEPS")
        self.log_report("=" * 80)
        self.log_report("\n1. Review preprocessing report dan visualisasi yang dihasilkan")
        self.log_report("2. Analisis activity frequencies dan case variants")
        self.log_report("3. Pertimbangkan untuk filter outliers jika diperlukan")
        self.log_report("4. Lanjutkan ke process discovery menggunakan tools seperti:")
        self.log_report("   - PM4Py untuk process mining")
        self.log_report("   - ProM untuk advanced analysis")
        self.log_report("   - Disco untuk visualisasi interaktif")
        
        # Save report to file
        self._save_report_to_file()
    
    def _save_report_to_file(self):
        """
        Save report ke text file
        """
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.report))
            
            file_size = os.path.getsize(self.report_file) / 1024  # KB
            self.log_report(f"\n📄 Report disimpan: '{self.report_file}' ({file_size:.2f} KB)")
            
        except Exception as e:
            self.log_report(f"\n❌ ERROR menyimpan report: {str(e)}")
    
    # ========== 11. MAIN ORCHESTRATION METHOD ==========
    def run(self):
        """
        Main orchestration method untuk menjalankan semua preprocessing steps
        """
        print("\n" + "=" * 80)
        print("🚀 PROCESS MINING PREPROCESSING PIPELINE")
        print("=" * 80 + "\n")

        try:
            # Step 1: Load Data
            if not self.load_data():
                print("\n❌ Preprocessing gagal: Tidak dapat memuat data")
                return False

            # Step 2: EDA
            self.perform_eda()

            # Step 3: Data Cleaning
            self.handle_missing_values(strategy='drop')
            self.remove_duplicates()

            # Step 4: Data Validation
            self.validate_data_format()

            # Step 5: Timestamp Processing
            self.process_timestamps()

            # Step 6: Case & Activity Processing
            self.process_case_and_activity()

            # Step 7: Feature Engineering
            self.engineer_features()

            # Step 8: Outlier Detection
            self.detect_outliers()

            # Step 9: Final Validation
            self.validate_for_process_mining()

            # Step 10: Export Data
            self.export_data()

            # Step 11: Generate Summary Report
            self.generate_summary_report()

            print("\n" + "=" * 80)
            print("✅ PREPROCESSING SELESAI!")
            print("=" * 80)
            print(f"\nOutput Files:")
            print(f"  1. {self.output_file} - Cleaned data")
            print(f"  2. {self.report_file} - Detailed report")
            print(f"  3. preprocessing_statistics.json - Statistics")
            print(f"  4. Various PNG visualizations")

            return True

        except Exception as e:
            print(f"\n❌ ERROR dalam preprocessing pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# ========== MAIN EXECUTION BLOCK ==========
if __name__ == "__main__":
    import time

    print("\n" + "=" * 80)
    print("   PROCESS MINING DATA PREPROCESSING")
    print("   Comprehensive Event Log Cleaning & Feature Engineering")
    print("=" * 80 + "\n")

    start_time = time.time()

    try:
        # Instantiate preprocessor
        preprocessor = ProcessMiningPreprocessor(
            input_file='D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/finale.csv',
            output_file='D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/finale_preprocessed.csv',
            report_file='D:/VSCODE/myenv2/share/Mining_Final_Project/Preprocessing/preprocessing_report.txt'
        )

        # Run preprocessing pipeline
        success = preprocessor.run()

        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time

        if success:
            print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
            print("\n🎉 Preprocessing berhasil! Silakan review output files.")
            print("\nNext steps:")
            print("  1. Review 'preprocessing_report.txt' untuk detail lengkap")
            print("  2. Check visualisasi PNG untuk insights")
            print("  3. Load 'finale_preprocessed.csv' untuk process mining analysis")
        else:
            print("\n❌ Preprocessing gagal. Check error messages di atas.")

    except FileNotFoundError:
        print("\n❌ ERROR: File 'finale.csv' tidak ditemukan!")
        print("Pastikan file berada di directory yang sama dengan script ini.")

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "=" * 80)
        print("Program selesai.")
        print("=" * 80 + "\n")