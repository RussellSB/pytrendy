import pandas as pd
import numpy as np
from collections import Counter

class PyTrendyResults: 
    """
    Wrapper class for accessing and analyzing detected trend segments.

    This class provides utilities for summarizing, filtering, and exporting trend segments
    detected by the `detect_trends` pipeline. It encapsulates both raw segment data and
    enhanced metrics such as rankings and signal-to-noise ratios.
    """

    def __init__(self, segments):
        """
        Initializes the results object with a list of segments.

        Args:
            segments (list):
                List of dictionaries representing individual trend segments.
        """
        self.segments = segments
        self.set_best()
        self.set_summary()
        self.set_segments_df()

    def set_best(self):
        """
        This prioritises both longest segment length (days) and steepness of trend.
        
        results.best returns best based on total_change (cumulative sum of differences). 

        Identifies the best trend segment based on steepness and duration.

        The segment with the lowest `change_rank` is selected as the best.
        """
        if not any('change_rank' in segment for segment in self.segments):
            self.best = None
            return
        self.best = min(self.segments, key=lambda x: x.get('change_rank'))

    def set_summary(self):
        """
        Computes summary statistics and stores a compact DataFrame of segments.

        Includes counts by direction and trend class, highest total change, and a tabular view.
        """
        summary = {}

        direction_counts = Counter(seg["direction"] for seg in self.segments)
        summary["direction_counts"] = dict(direction_counts)
        
        trend_class_counts = Counter(seg["trend_class"] for seg in self.segments if "trend_class" in seg)
        summary["trend_class_counts"] = dict(trend_class_counts)

        changes = [seg.get("total_change", 0) for seg in self.segments if "total_change" in seg]
        summary['highest_total_change'] = np.max(changes) if len(changes) > 0 else None

        # Set summary df (without extra details)
        df = pd.DataFrame(self.segments)
        cols = ['time_index', 'direction', 'start', 'end', 'days']
        if len(changes) > 1: cols += ['total_change', 'change_rank']
        df = df[cols]

        df = df.set_index('time_index')
        summary['df'] = df

        # Set summary
        self.summary = summary

    def print_summary(self):
        """
        Prints a human-readable summary of detected trends.

        Includes counts, best segment info, and a full tabular display.
        """

        uptrends = self.summary['direction_counts']['Up'] if 'Up' in self.summary['direction_counts'] else 0
        downtrends = self.summary['direction_counts']['Down'] if 'Down' in self.summary['direction_counts'] else 0
        flats = self.summary['direction_counts']['Flat'] if 'Flat' in self.summary['direction_counts'] else 0 
        noise = self.summary['direction_counts']['Noise'] if 'Noise' in self.summary['direction_counts'] else 0 
        print(f'Detected: \n- {uptrends} Uptrends. \n- {downtrends} Downtrends.\n- {flats} Flats.\n- {noise} Noise.\n')

        if len(self.filter_segments(direction='Up/Down')) == 0:
            print('Detected no trends...')
            return
        else:
            print(f'The best detected trend is {self.best["direction"]} between dates {self.best["start"]} - {self.best["end"]}\n')

        print('Full Results:')
        print('-------------------------------------------------------------------------------\n', 
              self.summary['df'],
            '\n-------------------------------------------------------------------------------')

    def set_segments_df(self):
        """
        Converts the segment list into a pandas DataFrame.

        Useful for downstream analysis and export.
        
        Alternative data representation to segments. In dataframe rather than dict
        """

        df = pd.DataFrame(self.segments)
        df = df.set_index('time_index')
        self.segments_df = df

    def filter_segments(self, direction:str='Any', sort_by:str='time_index', format='df'):
        """
        Filters and sorts segments based on direction and ranking.

        Args:
            direction (str, optional):
                Filter by trend direction. Options: `'Any'`, `'Up/Down'`, `'Up'`, `'Down'`, `'Flat'`, `'Noise'`.
            sort_by (str, optional):
                Sort segments by `'time_index'` (ascending) or `'change_rank'` (descending).
            format (str, optional):
                Output format. `'df'` returns a DataFrame, `'dict'` returns a list of dictionaries.

        Returns:
            Union[list, pd.DataFrame]:
                Filtered and sorted segments in the specified format.
        """
        segments = self.segments

        # Sort segments by index/rank
        if sort_by == 'change_rank':
            segments = sorted(segments, key=lambda x: abs(x.get('total_change', 0)), reverse=True) # descending
        elif sort_by == 'time_index':
            segments = sorted(segments, key=lambda x: abs(x.get('time_index', 0))) # ascending
        else:
            print(f'{sort_by} is not a valid sort_by. Please try one of [\'time_index\', \'change_rank\']')

        # Filter segments by direction
        options = ['Any', 'Up/Down', 'Up', 'Down', 'Flat', 'Noise']
        if direction != 'Any' and direction in options:
            allowed_directions = {'Up', 'Down'} if direction == 'Up/Down' else {direction}
            segments = [seg for seg in segments if seg['direction'] in allowed_directions]
        if direction not in options:
            print(f'{direction} is not a valid direction. Please try one of [\'Any\', \'Up/Down\', \'Up\', \'Down\', \'Flat\', \'Noise\']')

        if len(segments) == 0:
            return []

        if format not in ['dict', 'df']:
            print(f'{format} is not a valid format. Please try one of [\'dict\', \'df\']')
        if format=='dict':
            return segments
        elif format == 'df':
            df = pd.DataFrame(segments)
            df = df.set_index('time_index')
            return df
        
        return segments
