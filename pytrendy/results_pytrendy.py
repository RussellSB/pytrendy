import pandas as pd
import numpy as np
from collections import Counter

class PyTrendyResults: 
    """Wrapper for segment results."""

    def __init__(self, segments):
        self.segments = segments
        self.set_best()
        self.set_summary()
        self.set_segments_df()

    def set_best(self):
        """
        results.best returns best based on total_change (cumulative sum of differences). 
        This prioritises both longest segment length (days) and steepness of trend.
        """
        best = self.segments[0]
        if best['direction'] not in ['Up', 'Down']:
            best = None
        self.best = best

    def set_summary(self):
        summary = {}

        direction_counts = Counter(seg["direction"] for seg in self.segments)
        summary["direction_counts"] = dict(direction_counts)
        
        trend_class_counts = Counter(seg["trend_class"] for seg in self.segments if "trend_class" in seg)
        summary["trend_class_counts"] = dict(trend_class_counts)

        changes = [seg.get("total_change", 0) for seg in self.segments if "total_change" in seg]
        summary['highest_total_change'] = np.max(changes)

        df = pd.DataFrame(self.segments)
        df = df[['direction', 'start', 'end', 'days', 'total_change', 'change_rank', 'time_index']]
        df = df.set_index('time_index')
        summary['df']  = df

        self.summary = summary

    def print_summary(self):

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
        """Alternative data representation to segments. In dataframe rather than dict"""
        df = pd.DataFrame(self.segments)
        df = df.set_index('time_index')
        self.segments_df = df

    def filter_segments(self, direction:str='Any', sort_by:str='change_rank', format='df'):
        """
        Simple helper for getting segments 
        - filtered by direction ['Any', 'Up/Down', 'Up', 'Down', 'Flat', 'Noise']
        - sorted by time_index (ascending) or change_rank (descending)
        - return format, either of ['dict', 'df']
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
            print('No segments found...')
            return

        if format not in ['dict', 'df']:
            print(f'{format} is not a valid format. Please try one of [\'dict\', \'df\']')
        if format=='dict':
            return segments
        elif format == 'df':
            df = pd.DataFrame(segments)
            df = df.set_index('time_index')
            return df
        
        return segments
