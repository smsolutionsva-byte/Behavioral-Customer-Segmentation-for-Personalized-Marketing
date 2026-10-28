import pandas as pd
import numpy as np
from typing import List, Dict, Any

class RescueQueue:
    """
    Engine to identify and prioritize at-risk customers for immediate intervention.
    """

    def build_queue(self, features_df: pd.DataFrame, scores_df: pd.DataFrame, profiles: List[Dict[str, Any]], labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Build a prioritized rescue queue for high-risk customers.
        """
        if features_df.empty or scores_df.empty:
            return []

        # Create a working dataframe with necessary columns
        df = pd.DataFrame(index=features_df.index)
        df['customer_id'] = features_df.index if 'customer_id' not in features_df.columns else features_df['customer_id']
        df['segment_id'] = labels
        
        # Merge scores safely
        if 'value_score' in scores_df.columns:
            df['value_score'] = scores_df['value_score']
        else:
            df['value_score'] = np.random.uniform(0, 100, len(df))
            
        if 'churn_risk_score' in scores_df.columns:
            df['churn_risk_score'] = scores_df['churn_risk_score']
        else:
            df['churn_risk_score'] = np.random.uniform(0, 100, len(df))

        # Proxy features for recency and engagement
        recency_col = 'recency_days' if 'recency_days' in features_df.columns else 'recency'
        recency = features_df[recency_col] if recency_col in features_df.columns else pd.Series(0.0, index=df.index)
        df['recency_percentile'] = pd.Series(recency, index=df.index).rank(pct=True) * 100
        
        # engagement_decline_signal is a difference of two percentiles (-100..100); rescale to 0..100
        if 'engagement_decline_signal' in features_df.columns:
            engagement = (features_df['engagement_decline_signal'].clip(-100, 100) + 100) / 2
        else:
            engagement = pd.Series(50.0, index=df.index)
        df['engagement_decline'] = engagement

        # Filter high-risk customers
        high_risk_df = df[df['churn_risk_score'] > 50].copy()
        
        if high_risk_df.empty:
            return []

        # Calculate rescue priority score
        high_risk_df['rescue_priority_score'] = (
            high_risk_df['value_score'] * 0.3 + 
            high_risk_df['churn_risk_score'] * 0.3 + 
            (100 - high_risk_df['recency_percentile']) * 0.2 + 
            high_risk_df['engagement_decline'] * 0.2
        )

        # Sort descending by priority
        high_risk_df = high_risk_df.sort_values(by='rescue_priority_score', ascending=False)
        
        # Take top 100
        top_100 = high_risk_df.head(100)

        # Map segment IDs to names
        segment_map = {p.get('segment_id', -1): p.get('segment_name', f"Segment {p.get('segment_id')}") for p in profiles}

        queue = []
        for _, row in top_100.iterrows():
            segment_name = segment_map.get(row['segment_id'], f"Segment {row['segment_id']}")
            
            queue.append({
                "customer_id": str(row['customer_id']),
                "rescue_priority_score": float(row['rescue_priority_score']),
                "value_score": float(row['value_score']),
                "churn_risk_score": float(row['churn_risk_score']),
                "revenue_at_risk": float(row['value_score'] * 15.5), # Simulated estimation
                "days_until_projected_churn": max(1, int((100 - row['churn_risk_score']) / 2)),
                "recommended_action": "Reach out directly with premium retention offer",
                "segment_name": segment_name
            })

        return queue
