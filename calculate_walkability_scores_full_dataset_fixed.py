import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WalkabilityScorerFullDataset:
    """
    Calculate walkability scores for the ENTIRE MassDOT Roads dataset.
    Optimized for handling large datasets with chunked processing and robust type handling.
    """
    
    def __init__(self, geojson_path, dtype_json_path='massdot_roads_dtypes.json', chunk_size=10000):
        """Initialize the walkability scorer."""
        self.geojson_path = geojson_path
        self.dtype_json_path = dtype_json_path
        self.chunk_size = chunk_size
        self.data = None
        self.dtypes = None
        self.total_features = 0
        self.processed_features = 0
        self.features_with_complete_data = 0
        
        # Define required attributes with actual column names
        self.required_attributes = [
            'RT_SIDEWLK',      # Right sidewalk
            'LT_SIDEWLK',      # Left sidewalk
            'AADT',            # Traffic volume
            'SPEED_LIM',       # Speed limit
            'CLASS',           # Road classification
            'ADMIN_TYPE',      # Administrative type
            'TERRAIN',         # Terrain type
            'LENGTH_MI',       # Segment length
            'SHLDR_RT_W',      # Right shoulder width
            'SHLDR_LT_W',      # Left shoulder width
            'CURB',            # Curb presence
            'STREET_NAME',     # Street name
            'CITY'             # City
        ]
        
        # Define user class weights
        self.user_weights = {
            'seniors': {
                'sidewalk_presence': 0.35,
                'sidewalk_width': 0.20,
                'traffic_safety': 0.20,
                'segment_length': 0.15,
                'terrain': 0.10
            },
            'children': {
                'sidewalk_presence': 0.40,
                'traffic_safety': 0.30,
                'road_classification': 0.20,
                'sidewalk_width': 0.10
            },
            'mobility_impaired': {
                'sidewalk_presence': 0.40,
                'sidewalk_width': 0.25,
                'terrain': 0.15,
                'road_classification': 0.10,
                'curb_presence': 0.10
            },
            'athletes': {
                'segment_length': 0.30,
                'traffic_safety': 0.25,
                'terrain': 0.20,
                'shoulder_width': 0.15,
                'sidewalk_presence': 0.10
            },
            'standard': {
                'sidewalk_presence': 0.35,
                'traffic_safety': 0.25,
                'sidewalk_width': 0.20,
                'road_classification': 0.20
            }
        }
    
    def safe_float(self, value, default=0):
        """Safely convert value to float."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def safe_int(self, value, default=0):
        """Safely convert value to int."""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def safe_str(self, value, default=''):
        """Safely convert value to string."""
        try:
            return str(value) if value is not None else default
        except:
            return default
    
    def load_data_info(self):
        """Load basic information about the dataset without loading all data."""
        print(f"Loading dataset information from: {self.geojson_path}")
        
        with open(self.geojson_path, 'r') as f:
            self.data = json.load(f)
        
        self.total_features = len(self.data.get('features', []))
        print(f"Total features in dataset: {self.total_features:,}")
        
        # Load data types
        with open(self.dtype_json_path, 'r') as f:
            self.dtypes = json.load(f)
    
    def check_complete_data(self, props):
        """Check if a feature has complete data for all required attributes."""
        for attr in self.required_attributes:
            if attr not in props or props[attr] is None:
                return False
            
            # Additional checks for numeric fields
            if attr in ['RT_SIDEWLK', 'LT_SIDEWLK', 'AADT', 'SPEED_LIM', 
                       'LENGTH_MI', 'SHLDR_RT_W', 'SHLDR_LT_W']:
                val = self.safe_float(props[attr], None)
                if val is None or val < 0:
                    return False
            
            # Check for empty strings
            if isinstance(props[attr], str) and props[attr].strip() == '':
                return False
        
        return True
    
    def calculate_sidewalk_presence_score(self, props, user_class):
        """Calculate sidewalk presence score."""
        right_sidewalk = self.safe_float(props.get('RT_SIDEWLK', 0))
        left_sidewalk = self.safe_float(props.get('LT_SIDEWLK', 0))
        
        has_right = right_sidewalk > 0
        has_left = left_sidewalk > 0
        has_any = has_right or has_left
        has_both = has_right and has_left
        
        if has_both:
            score = 100
        elif has_any:
            score = 60
        else:
            score = 0
        
        if user_class == 'athletes':
            if not has_any:
                right_shoulder = self.safe_float(props.get('SHLDR_RT_W', 0))
                left_shoulder = self.safe_float(props.get('SHLDR_LT_W', 0))
                if right_shoulder > 4 or left_shoulder > 4:
                    score = 40
        
        elif user_class in ['children', 'mobility_impaired']:
            if not has_both:
                score = score * 0.5
        
        return score
    
    def calculate_sidewalk_width_score(self, props, user_class):
        """Calculate sidewalk width adequacy score."""
        right_width = self.safe_float(props.get('RT_SIDEWLK', 0))
        left_width = self.safe_float(props.get('LT_SIDEWLK', 0))
        
        if right_width == 0 and left_width == 0:
            return 0
        
        max_width = max(right_width, left_width)
        
        if max_width >= 6:
            score = 100
        elif max_width >= 5:
            score = 80
        elif max_width >= 4:
            score = 60
        elif max_width >= 3:
            score = 40
        else:
            score = 20
        
        if user_class == 'mobility_impaired':
            if max_width < 5:
                score = score * 0.5
        elif user_class == 'children':
            if max_width < 4:
                score = score * 0.7
        
        return score
    
    def calculate_traffic_safety_score(self, props):
        """Calculate traffic safety score."""
        score = 50
        
        aadt = self.safe_float(props.get('AADT', 0))
        if aadt > 0:
            if aadt < 1000:
                score += 30
            elif aadt < 5000:
                score += 20
            elif aadt < 15000:
                score += 10
            elif aadt < 30000:
                score -= 10
            else:
                score -= 20
        
        speed = self.safe_float(props.get('SPEED_LIM', 0))
        if speed > 0:
            if speed <= 25:
                score += 20
            elif speed <= 35:
                score += 10
            elif speed <= 45:
                score -= 10
            else:
                score -= 20
        
        return max(0, min(100, score))
    
    def calculate_road_classification_score(self, props, user_class):
        """Calculate score based on road classification."""
        road_class = self.safe_int(props.get('CLASS', 0))
        admin_type = self.safe_int(props.get('ADMIN_TYPE', 0))
        
        class_scores = {
            1: 10,
            2: 20,
            3: 40,
            4: 60,
            5: 80,
            6: 30,
            7: 50
        }
        
        score = class_scores.get(road_class, 50)
        
        if user_class == 'athletes':
            if road_class in [3, 4]:
                score += 10
        elif user_class in ['children', 'seniors']:
            if road_class == 5:
                score += 20
            elif road_class in [1, 2]:
                score -= 20
        
        if admin_type in [1, 2]:
            score -= 30
        
        return max(0, min(100, score))
    
    def calculate_terrain_score(self, props, user_class):
        """Calculate terrain suitability score."""
        terrain = self.safe_str(props.get('TERRAIN', '')).lower()
        
        terrain_scores = {
            'flat': 100,
            'rolling': 70,
            'hilly': 40,
            'mountainous': 20
        }
        
        score = 70
        
        for terrain_type, base_score in terrain_scores.items():
            if terrain_type in terrain:
                score = base_score
                break
        
        if user_class == 'athletes':
            if 'rolling' in terrain or 'hilly' in terrain:
                score += 20
        elif user_class in ['seniors', 'mobility_impaired']:
            if 'hilly' in terrain or 'mountainous' in terrain:
                score -= 30
        
        return max(0, min(100, score))
    
    def calculate_segment_length_score(self, props, user_class):
        """Calculate score based on segment length preferences."""
        length_mi = self.safe_float(props.get('LENGTH_MI', 0))
        
        if length_mi <= 0:
            return 50
        
        length_ft = length_mi * 5280
        
        if user_class == 'seniors':
            if length_ft < 500:
                return 100
            elif length_ft < 1000:
                return 80
            elif length_ft < 2000:
                return 60
            else:
                return 40
        
        elif user_class == 'athletes':
            if length_ft > 5000:
                return 100
            elif length_ft > 2500:
                return 80
            elif length_ft > 1000:
                return 60
            else:
                return 40
        
        else:
            if 500 <= length_ft <= 2000:
                return 100
            elif 200 <= length_ft <= 5000:
                return 80
            else:
                return 60
    
    def calculate_shoulder_score(self, props):
        """Calculate shoulder availability score."""
        right_shoulder = self.safe_float(props.get('SHLDR_RT_W', 0))
        left_shoulder = self.safe_float(props.get('SHLDR_LT_W', 0))
        
        max_shoulder = max(right_shoulder, left_shoulder)
        
        if max_shoulder >= 6:
            return 100
        elif max_shoulder >= 4:
            return 70
        elif max_shoulder >= 2:
            return 40
        else:
            return 0
    
    def calculate_curb_score(self, props):
        """Calculate curb presence score."""
        curbs = self.safe_str(props.get('CURB', '')).lower()
        
        if 'yes' in curbs or 'both' in curbs:
            return 100
        elif 'one' in curbs or 'right' in curbs or 'left' in curbs:
            return 70
        elif 'no' in curbs or 'none' in curbs:
            return 30
        else:
            return 50
    
    def calculate_segment_score(self, props, user_class):
        """Calculate walkability score for a single road segment."""
        scores = {}
        
        scores['sidewalk_presence'] = self.calculate_sidewalk_presence_score(props, user_class)
        scores['sidewalk_width'] = self.calculate_sidewalk_width_score(props, user_class)
        scores['traffic_safety'] = self.calculate_traffic_safety_score(props)
        scores['road_classification'] = self.calculate_road_classification_score(props, user_class)
        scores['terrain'] = self.calculate_terrain_score(props, user_class)
        scores['segment_length'] = self.calculate_segment_length_score(props, user_class)
        scores['shoulder_width'] = self.calculate_shoulder_score(props)
        scores['curb_presence'] = self.calculate_curb_score(props)
        
        weights = self.user_weights[user_class]
        
        total_score = 0
        total_weight = 0
        
        for factor, weight in weights.items():
            if factor in scores:
                total_score += scores[factor] * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 50
        
        return round(final_score, 2)
    
    def process_dataset(self, output_dir='walkability_results_full'):
        """Process the entire dataset in chunks."""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Initialize results files
        results_files = {}
        for user_class in self.user_weights.keys():
            filename = f'{output_dir}/scores_{user_class}.csv'
            results_files[user_class] = open(filename, 'w')
            results_files[user_class].write('feature_index,street_name,city,score\n')
        
        # Track errors
        errors_logged = 0
        error_file = open(f'{output_dir}/processing_errors.log', 'w')
        
        # Process features in chunks
        features = self.data.get('features', [])
        num_chunks = (self.total_features + self.chunk_size - 1) // self.chunk_size
        
        print(f"\nProcessing {self.total_features:,} features in {num_chunks} chunks...")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.total_features)
            chunk_features = features[start_idx:end_idx]
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (features {start_idx:,} to {end_idx:,})")
            
            # Process each feature in the chunk
            for i, feature in enumerate(chunk_features):
                feature_idx = start_idx + i
                
                try:
                    props = feature.get('properties', {})
                    
                    # Check if has complete data
                    if self.check_complete_data(props):
                        self.features_with_complete_data += 1
                        
                        # Calculate scores for each user class
                        for user_class in self.user_weights.keys():
                            score = self.calculate_segment_score(props, user_class)
                            
                            # Write to file with safe string conversion
                            street_name = self.safe_str(props.get('STREET_NAME', '')).replace(',', ';')
                            city = self.safe_str(props.get('CITY', '')).replace(',', ';')
                            results_files[user_class].write(f'{feature_idx},{street_name},{city},{score}\n')
                    
                    self.processed_features += 1
                    
                except Exception as e:
                    errors_logged += 1
                    error_file.write(f"Error at feature {feature_idx}: {str(e)}\n")
                    self.processed_features += 1
                
                # Progress update
                if (self.processed_features % 5000) == 0:
                    progress = (self.processed_features / self.total_features) * 100
                    print(f"  Progress: {self.processed_features:,}/{self.total_features:,} ({progress:.1f}%)")
                    print(f"  Features with complete data: {self.features_with_complete_data:,}")
                    if errors_logged > 0:
                        print(f"  Errors encountered: {errors_logged}")
        
        # Close all files
        for f in results_files.values():
            f.close()
        error_file.close()
        
        print(f"\nProcessing complete!")
        print(f"Total features processed: {self.processed_features:,}")
        print(f"Features with complete data: {self.features_with_complete_data:,}")
        print(f"Percentage with complete data: {(self.features_with_complete_data/self.processed_features)*100:.1f}%")
        if errors_logged > 0:
            print(f"Total errors logged: {errors_logged}")
        
        # Generate summary statistics
        self.generate_summary(output_dir)
    
    def generate_summary(self, output_dir):
        """Generate summary statistics from the results."""
        print("\nGenerating summary statistics...")
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_features': self.total_features,
            'processed_features': self.processed_features,
            'features_with_complete_data': self.features_with_complete_data,
            'percentage_complete': (self.features_with_complete_data/self.processed_features)*100 if self.processed_features > 0 else 0,
            'user_classes': list(self.user_weights.keys()),
            'score_statistics': {}
        }
        
        # Calculate statistics for each user class
        for user_class in self.user_weights.keys():
            filename = f'{output_dir}/scores_{user_class}.csv'
            
            # Read scores
            scores = []
            with open(filename, 'r') as f:
                f.readline()  # Skip header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        try:
                            score = float(parts[3])
                            scores.append(score)
                        except:
                            pass
            
            if scores:
                summary['score_statistics'][user_class] = {
                    'count': len(scores),
                    'mean': round(np.mean(scores), 2),
                    'median': round(np.median(scores), 2),
                    'std': round(np.std(scores), 2),
                    'min': round(np.min(scores), 2),
                    'max': round(np.max(scores), 2),
                    'percentiles': {
                        '10th': round(np.percentile(scores, 10), 2),
                        '25th': round(np.percentile(scores, 25), 2),
                        '50th': round(np.percentile(scores, 50), 2),
                        '75th': round(np.percentile(scores, 75), 2),
                        '90th': round(np.percentile(scores, 90), 2)
                    },
                    'score_distribution': {
                        'Excellent (80-100)': sum(1 for s in scores if s >= 80),
                        'Good (60-80)': sum(1 for s in scores if 60 <= s < 80),
                        'Fair (40-60)': sum(1 for s in scores if 40 <= s < 60),
                        'Poor (20-40)': sum(1 for s in scores if 20 <= s < 40),
                        'Very Poor (0-20)': sum(1 for s in scores if s < 20)
                    }
                }
        
        # Save summary
        summary_path = f'{output_dir}/analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("WALKABILITY ANALYSIS SUMMARY - FULL DATASET")
        print("="*80)
        print(f"Total features in dataset: {summary['total_features']:,}")
        print(f"Features with complete data: {summary['features_with_complete_data']:,}")
        print(f"Percentage with complete data: {summary['percentage_complete']:.1f}%")
        
        print("\nScore Statistics by User Class:")
        print("-"*50)
        
        for user_class, stats in summary['score_statistics'].items():
            print(f"\n{user_class.upper().replace('_', ' ')}:")
            print(f"  Mean: {stats['mean']:.1f}")
            print(f"  Median: {stats['median']:.1f}")
            print(f"  Std Dev: {stats['std']:.1f}")
            print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
            
            # Score distribution
            dist = stats['score_distribution']
            excellent_good = dist['Excellent (80-100)'] + dist['Good (60-80)']
            poor_very_poor = dist['Poor (20-40)'] + dist['Very Poor (0-20)']
            print(f"  Excellent/Good: {excellent_good:,} ({excellent_good/stats['count']*100:.1f}%)")
            print(f"  Poor/Very Poor: {poor_very_poor:,} ({poor_very_poor/stats['count']*100:.1f}%)")


def main():
    """Main function to run walkability analysis for the full dataset."""
    # Set paths
    geojson_path = "/Users/tommaso/prototypescorings/MassDOTRoads_gdb_1226590767708312459.geojson"
    dtype_json_path = "/Users/tommaso/prototypescorings/massdot_roads_dtypes.json"
    
    # Create scorer
    print("="*80)
    print("WALKABILITY ANALYSIS - FULL MASSDOT DATASET")
    print("="*80)
    
    scorer = WalkabilityScorerFullDataset(geojson_path, dtype_json_path, chunk_size=10000)
    
    # Load dataset info
    scorer.load_data_info()
    
    # Process the entire dataset
    scorer.process_dataset()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("Results saved to 'walkability_results_full' directory")
    print("="*80)


if __name__ == "__main__":
    main()
